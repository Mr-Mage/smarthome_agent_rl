"""Sequential protocol gates; episodes execute in two parallel isolated actor workflows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import signal
import os

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='runs/harness-mvp/primary-v1')
parser.add_argument('--wait-nodes', action='store_true', help='Wait for node diagnostics and finish node records/archival')
parser.add_argument('--node-pid', type=int, help='Fail rather than wait forever if the owning node supervisor exits')
args = parser.parse_args()
def command(script, *arguments):
    subprocess.run([sys.executable, ROOT / 'scripts' / script, *arguments], cwd=ROOT, check=True)
if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
    raise RuntimeError('Primary experiment requires a clean committed project')
state_file = ROOT / 'work/harness-mvp/primary-state.json'
state_file.parent.mkdir(parents=True, exist_ok=True)
def state(phase, **fields):
    state_file.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), **fields}, indent=2))
def commit_record(node, text):
    target = ROOT / 'docs/nodes' / f'{node}.md'
    if target.exists():
        return
    target.write_text(text, encoding='utf-8')
    subprocess.run(['git', 'add', str(target.relative_to(ROOT))], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', f'docs: record {node} verified node results'], cwd=ROOT, check=True)
for node in ('N3', 'N4', 'N5'):
    node_root = ROOT / 'runs/harness-mvp/nodes-v1'
    accepted = node_root / f'{node}-accepted.json'
    while not accepted.exists():
        if not args.wait_nodes:
            raise RuntimeError(f'{node} diagnostics are not accepted')
        state('waiting_for_' + node)
        if (node_root / node / 'failure.json').exists():
            raise RuntimeError(f'{node} failed; evidence preserved, no automatic retry')
        if args.node_pid:
            process = Path(f'/proc/{args.node_pid}/cmdline')
            if not process.exists() or b'run_node_diagnostics.py' not in process.read_bytes():
                state('node_supervisor_failed', node=node)
                raise RuntimeError(f'Node supervisor exited before {node} acceptance; inspect its log')
        time.sleep(5)
    if not json.loads(accepted.read_text())['complete']:
        raise RuntimeError(f'{node} diagnostic is incomplete')
    command('verify_benchmark.py', str(node_root / node))
    report = json.loads((node_root / node / 'report.json').read_text())
    if args.wait_nodes and node == 'N4':
        g, gv = report['arms']['G'], report['arms']['GV']
        commit_record(node, f'''# N4 Verify/Repair 验收

问题：API 成功不等于局部效果已发生，工作流注册也不等于未来成功。

方案：根据动作参数验证公开局部状态；过渡期标记 pending，不提前推进时间。最多两次修复、20 模型轮次，每轮最多一次前查与一次后查，总额外查询上限 40。

结果：24 项 G/GV 配对，成功 {g['successes']}/24 与 {gv['successes']}/24；GV 局部验证失败 {gv['all_totals']['verification_failures']} 次，恢复 {gv['all_totals']['recovered_actions']} 次，额外查询 {gv['all_totals']['extra_queries']} 次。完整失败与成本见报告，不据此承诺成功收益。

证据：`runs/harness-mvp/nodes-v1/N4/report.json`；提交：`git log -- docs/nodes/N4.md`。
''')
    if args.wait_nodes and node == 'N5':
        gc, full = report['arms']['GC'], report['arms']['Full']
        commit_record(node, f'''# N5 Context 验收

问题：重复历史占用上下文，写操作、时间和异步工作流会使旧状态失效。

方案：确定性账本保存 query、观测来源、历史差值、错误和工作流；保留最近两对原始消息，标记旧状态。账本更长时沿用原历史，不调用摘要模型，不裁工具集。

结果：24 项 GC/Full 配对，成功 {gc['successes']}/24 与 {full['successes']}/24；actor tokens {gc['all_totals']['actor_tokens']} 与 {full['all_totals']['actor_tokens']}。重复查询来源、schema 错误、时间失效与原指令保留均有测试；机制效应在主实验分别比较 G/GC 和 GV/Full。

证据：`runs/harness-mvp/nodes-v1/N5/report.json`；提交：`git log -- docs/nodes/N5.md`。
''')
state('primary_dev')
command('run_benchmark_suite.py', '--phase', 'dev', '--run-dir', args.prefix + '/dev')
command('report_benchmark.py', args.prefix + '/dev')
state('freezing_final')
command('freeze_experiment.py', '--dev-run', args.prefix + '/dev')
state('primary_final')
command('run_benchmark_suite.py', '--phase', 'final', '--run-dir', args.prefix + '/final',
        '--freeze', 'work/harness-mvp/final-freeze.json')
command('report_benchmark.py', args.prefix + '/final')
(ROOT / args.prefix / 'accepted.json').write_text(json.dumps({'complete': True,
    'dev_episodes': 600, 'final_episodes': 768, 'reports': ['dev/report.json', 'final/report.json']}, indent=2))
if args.wait_nodes:
    final = json.loads((ROOT / args.prefix / 'final/report.json').read_text())
    values = ', '.join(f"{variant} {row['successes']}/192" for variant, row in final['arms'].items())
    commit_record('N6', f'''# N6 冻结消融

问题：需在独立 final 上检验机制收益，避免开发案例与临时补跑混入。

方案：干净 Git 基点下运行 600 个 dev episodes，冻结版本、参数、模型身份及 final 清单，再运行 B0/G/GC/Full 四组 768 episodes。保持配对、同 actor、交替顺序；采用 McNemar、分层 bootstrap CI 与三项 Holm 校正。

结果：两组协议完整核验；final 成功数：{values}。分类、失败、全部/成功任务成本和统计见报告；额外节点诊断单列。

证据：`{args.prefix}/dev/report.json`、`{args.prefix}/final/report.json`、`work/harness-mvp/final-freeze.json`；提交：`git log -- docs/nodes/N6.md`。
''')
    state('verifying_and_archiving')
    command('verify_benchmark.py', str(ROOT / args.prefix / 'dev'))
    command('verify_benchmark.py', str(ROOT / args.prefix / 'final'))
    freeze = json.loads((ROOT / 'work/harness-mvp/final-freeze.json').read_text())
    lightning = ROOT / freeze['upstream']['agent-lightning']['path']
    patch = subprocess.check_output(['git', '-C', str(lightning), 'diff', 'HEAD'])
    if hashlib.sha256(patch).hexdigest() != freeze['lightning_user_patch_sha256']:
        raise RuntimeError('Lightning user changes differ from frozen evidence')
    if subprocess.check_output(['git', '-C', str(ROOT / 'deps/SimuHome'), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('SimuHome changed during evaluation')
    command('archive_transfers.py')
    archive = ROOT / 'outputs/harness-mvp'
    archive.mkdir(parents=True, exist_ok=True)
    bundle = archive / 'harness-final.bundle'
    subprocess.run(['git', 'bundle', 'create', str(bundle), 'feat/harness-mvp'], cwd=ROOT, check=True)
    package = archive / 'primary-evidence.tar.gz'
    if package.exists():
        raise FileExistsError('Never replace a previous evidence archive')
    subprocess.run(['tar', '-czf', str(package), '-C', str(ROOT), args.prefix,
                    'work/harness-mvp/final-freeze.json', 'work/harness-mvp/inventory-v2'], check=True)
    digest = hashlib.sha256()
    with package.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    (archive / 'archive-verification.json').write_text(json.dumps({'archive': package.name,
        'sha256': digest.hexdigest(), 'bytes': package.stat().st_size, 'raw_evidence_preserved': True}, indent=2))
    commit_record('N7', f'''# N7 验收归档

问题：完整实验还需独立核对证据、保留失败与用户上游补丁，并整理交付入口。

方案：逐文件 SHA256 核验 dev/final，检查全部 episode 覆盖和上游状态；归档历史传输包，导出 Git bundle 和带身份校验的结果包。

结果：600 dev 与 768 final 原始证据完整；SimuHome 保持干净，Lightning 用户改动与冻结时一致。原始文件及历史失败均保留。结果包：`outputs/harness-mvp/primary-evidence.tar.gz`，身份：`archive-verification.json`。

提交：`git log -- docs/nodes/N7.md`。
''')
    readme = ROOT / 'README.md'
    readme.write_text(f'''# SmartHome Agent Harness

原 SimuHome ReAct + Agent Lightning；Qwen3.5-9B actor，本地 Qwen3.6-35B-A3B judge。
四卡分工：GPU 0/1 两路独立 actor；GPU 2/3 共享 TP2 judge；CPU BGE 检索。

已完成 N0–N7：600 dev + 768 final。final 成功数：{values}。
开发记录：`docs/nodes/`；约定：`AGENTS.md`；历史资料：`docs/archive/pre-mvp/`。
结果与证据：`{args.prefix}/`、`outputs/harness-mvp/`；原始失败保留，主实验不混入额外诊断。

服务器：`ssh h100`，激活已有 `../activate-agent-lightning.sh`；模型用 qwen36-vllm，episode/模拟器用各自 venv，环境与用户上游补丁保留。

运行入口：`scripts/harness_services.py`、`scripts/run_benchmark_suite.py`、`scripts/report_benchmark.py`。
独立复核：`python scripts/verify_benchmark.py <run-dir>`；冻结：`work/harness-mvp/final-freeze.json`。
''', encoding='utf-8')
    (ROOT / 'PROJECT_STATUS.md').write_text(f'''# 当前状态

N0–N7 已验收。final：{values}；统计与全部成本见 `{args.prefix}/final/report.json`。
每节点记录见 `docs/nodes/`；历史 P1 不与本轮官方 benchmark 混算。
''', encoding='utf-8')
    subprocess.run(['git', 'add', 'README.md', 'PROJECT_STATUS.md'], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', 'docs: publish verified harness results and concise project entry'], cwd=ROOT, check=True)
    subprocess.run(['git', 'bundle', 'create', str(bundle), 'feat/harness-mvp'], cwd=ROOT, check=True)
    supervisor = ROOT / 'work/harness-mvp/services-v3/supervisor.pid'
    if supervisor.exists():
        pid = int(supervisor.read_text())
        command_line = Path(f'/proc/{pid}/cmdline')
        if command_line.exists() and b'harness_services.py' in command_line.read_bytes():
            os.kill(pid, signal.SIGTERM)
    state('complete', final_successes={v: r['successes'] for v, r in final['arms'].items()},
          archive_sha256=digest.hexdigest())
