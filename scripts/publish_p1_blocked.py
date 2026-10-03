"""Publish a truthful incomplete P1 handover, without inventing paired results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/p1-20261003"
docs = ROOT / "docs/p1-20261003"
docs.mkdir(parents=True, exist_ok=True)
load = lambda root, name: json.loads((root / name).read_text(encoding="utf-8"))
diag = load(ROOT / "runs/p1-report-20261003", "diagnostics_verification.json")
assert diag["artifact_checks_passed"]
shutil.copy2(ROOT / "runs/p1-report-20261003/diagnostics_verification.json", docs / "diagnostics_verification.json")
partial = ROOT / "runs/p1-paired-round1-20261003-01"
results = load(partial, "results.json")
assert len(results) == 13 and not (partial / "comparison.json").exists()
infrastructure = []
for row in results:
    trajectory = [json.loads(line) for line in (partial / row["path"] / "episode/trajectory.jsonl").read_text().splitlines()]
    for step in trajectory:
        if step["infrastructure_error"]:
            infrastructure.append({"task_id": row["task_id"], "harness": row["harness"],
                "turn": step["turn"], "action": step["action"], "observation": step["observation"]})
historical = ROOT / "runs/b0-qwen35-9b-20261003-03"
p0_errors = []
for row in load(historical, "results.json"):
    if row["summary"]["infrastructure_errors"]:
        p0_errors.append({"task_id": row["task_id"], "errors": row["summary"]["infrastructure_errors"], "source": row["path"]})
status = {"captured_utc": datetime.now(timezone.utc).isoformat(), "p1_complete": False,
    "stability_verified": True, "fixed_requests": 72, "chain_episodes": 8,
    "complete_formal_rounds": 0, "planned_formal_episodes": 96, "incomplete_round_episodes": len(results),
    "incomplete_round": str(partial), "infrastructure_evidence": infrastructure,
    "p0_historical_infrastructure_errors": p0_errors, "p0_raw_artifacts_unchanged": True,
    "tests_passed": 23, "training": False,
    "blocked_by": "get_cluster_doc tokenizer network dependency and FAISS-compatible embedding backend configuration",
    "tokenizer": load(WORK, "doc_dependency_audit.json"),
    "resume": "Provide existing compatible embedding configuration or explicitly revise unavailable-documentation protocol; then create a new entire round1 and run round2/round3; never replace selectively rerun tasks"}
(docs / "P1-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
lines = ["# P1 执行状态：诊断完成，三轮实验被检索依赖阻塞", "",
    "**P1 尚未完成。** 72 次固定请求与 8 个完整链路诊断已完成并独立核验；正式第一轮执行了 13/32 个 episode 后因基础设施错误中止。完整正式轮数为 0/3，不能得出 H1 成功率或成本改善结论。", "",
    "原有 20 项检查与新增 3 项检查通过。已实现任务/变体排程参数、请求冻结、仅生成诊断 worker、链路核验、跨轮报告和失败证据归类；正式报告入口只接受三轮完整独立核验结果。", "",
    "阻塞发生在 dev-expanded-dimmer-1 的 get_cluster_doc：上游检索工具下载 cl100k_base.tiktoken 时 DNS 失败，返回 500。本轮保留完整未完成目录，不计作正式分数。随后通过既有代理取回官方文件，并校验 SHA256=223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7，项目工作目录缓存已离线验证。", "",
    "仍需修复 embedding 配置：工具用 OpenAIEmbeddings 对既有 FAISS 索引检索，而 worker 硬编码的 local-unused / OpenAI base 地址指向 language-model-only 的 Qwen3.5-9B vLLM。未发现项目 .env 配置。离线配置审计未向任何 embedding 服务发出请求，也未读取或显示密钥。不能通过换检索算法、伪造返回值或放宽基础设施检查满足原协议。", "",
    "恢复需要已有、与 FAISS 索引匹配的 embedding 服务配置路径；不要在聊天贴密钥。另一选择是用户明确修订协议，承认文档检索不可用，单列该限制。原计划要求基础设施故障整轮作废，故不能擅自采用后一方案。", "",
    "P0 补充纠正：其原 summary 中也记录了该检索基础设施错误，见 P1-status.json 的 p0_historical_infrastructure_errors。P0 12/16 的真实设备成功结论保留，但此前报告的“基础设施故障 0/16”不准确。原 P0 产物不覆盖，作为历史参照且不并入 P1 三轮。", "",
    "已完成的稳定性诊断见 [P1 稳定性报告](P1-稳定性报告.md)。正式配对源码与实际未完成排程、全部轨迹保留在交付包。恢复后第一轮必须新建目录从全部 16 项两组开始；第二轮反序/offset=1，第三轮复用第一轮排程；每轮重新启动服务并独立核验。", "",
    "诊断启动导入冲突和重启端口预检失败目录均保留，它们未生成推理请求。随后 72 次请求/8 个链路使用完整新目录成功完成。没有训练、新增变体或修改 SimuHome/Lightning 用户代码。"]
(docs / "P1-执行与阻塞报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
groups = diag["groups"]
stability = ["# P1：9B 稳定性与 Lightning 接入报告", "",
    "72 次固定请求与 8 个链路 episode 全部独立核验通过，均标记 diagnostic_only，不计入正式成功率。", "",
    "四个任务族各第一任务的 P0 首请求完整冻结；直接路径同样添加 return_token_ids=true。所有实际后端请求与冻结请求哈希一致。首次正序 24 次、同服务反序 24 次、重启后正序 24 次，两条路径先后交替。", "",
    "|任务|路径|重复|不同全文|不同规范化动作|不同输入 token IDs|不同回复 token IDs|",
    "|---|---|---:|---:|---:|---:|---:|"]
for g in groups:
    stability.append(f"|{g['task_id']}|{g['path']}|{g['requests']}|{g['unique_texts']}|{g['unique_actions']}|{g['unique_prompt_ids']}|{g['unique_response_ids']}|")
stability += ["", "36 对相邻 direct/Gateway 请求的全文、动作、prompt IDs 与 reply IDs 全部一致。", "",
    "|任务|direct 严格成功|Lightning 严格成功|direct reward|Lightning reward|动作/反馈首次分叉|",
    "|---|---|---|---:|---:|---|"]
for c in diag["chains"]:
    a, b = c["summaries"]["direct"], c["summaries"]["lightning"]
    stability.append(f"|{c['task_id']}|{a['success']}|{b['success']}|{a['reward']:.6f}|{b['reward']:.6f}|{c['first_divergence_turn']['action']} / {c['first_divergence_turn']['observation']}|")
stability += ["", "8 个链路 episode 的初始目标状态、首个实际后端请求、真实工具反馈、最终目标检查和 reward 均核对；没有动作/反馈分叉。两条路径均为 3/4 成功，调光任务失败保留。完整 state 的时钟差异仍在轨迹，不要求全部状态逐字相同。", "",
    "诊断开销：" + json.dumps(diag["cost"], ensure_ascii=False), "",
    "解释边界：这次冻结首请求与四条任务链的结果一致，支持本次 Lightning 接入保持预期行为；不能推广为所有输入确定性保证。顺序、热状态和重启共同变化，不声称隔离了其因果影响。采样、任务、模型配置与既有 9B 基线一致。", "",
    "完整证据见 diagnostics_verification.json、两个成功诊断目录的 service_start.json、diagnostic_schedule.json、frozen_requests.json、request_records.json、chain_results.json 和各请求/episode 子目录。"]
(docs / "P1-稳定性报告.md").write_text("\n".join(stability) + "\n", encoding="utf-8")
supplement = (WORK / "P1-learning-supplement.md").read_text(encoding="utf-8")
for key, value in {
    "GATEWAY_RESULT": "72 次冻结请求全部与后端请求哈希相同，36 对 direct/Gateway 的文本、动作、输入与输出 token IDs 完全一致。8 个链路 episode 的事件、最终状态与 reward 通过独立核验。",
    "STABILITY_RESULT": "72 次请求和 8 个完整 episode 核验通过，无动作/反馈分叉；两条路径均为 3/4 严格成功，调光失败保留。诊断总固定请求成本 311760 prompt tokens + 5742 completion tokens，链路 56 次调用、297220 tokens。",
    "HARNESS_RESULT": "正式第一轮在 13/32 episode 时被文档检索基础设施故障中止，完整轮数 0/3。H1 的成功率和成本提升尚未得到可验收结论，不使用未完成轮作为成绩。",
    "SCHEDULE_RESULT": "参数和完整展开排程已实现，新增覆盖测试通过。未完成第一轮的 32 项预定排程和 13 项实际执行轨迹全部保存；三轮 96 项与跨轮统计仍待依赖修复。",
    "ERROR_RESULT": "23 项检查通过；稳定性诊断独立核验通过。正式轮遇到 get_cluster_doc 的 500，按协议中止并保留。tokenizer 文件已下载并官方哈希验证，embedding 后端仍无配置。P0 也有该类错误；此前“基础设施 0/16”需纠正，P0 成绩仍仅作历史参照。"
}.items():
    supplement = supplement.replace("{{" + key + "}}", value)
assert "{{" not in supplement
learning_dir = ROOT / "docs/agent-learning-20261003"
learning = learning_dir / "Agent模块设计与Lightning基线学习文档.md"
base = learning.read_text(encoding="utf-8-sig")
backup = WORK / "handover-before-blocked-report"
backup.mkdir(exist_ok=True)
shutil.copy2(learning, backup / learning.name)
if "## 16. P1 学习模块" in base:
    base = base[:base.index("## 16. P1 学习模块")]
base = base.replace("| 9B 重复稳定性 / direct 与 Lightning 配对 | 尚未完成 |", "| 9B 重复稳定性 / direct 与 Lightning 配对 | P1 72 次请求 + 8 个链路核验完成 |")
base = base.replace("| 9B 的 structured harness 对照 | 尚未开展 |", "| 9B 的 structured harness 对照 | 第一轮因检索基础设施中止，0/3 完整轮 |")
learning.write_text(base.rstrip() + "\n\n" + supplement, encoding="utf-8")
(docs / learning.name).write_text(learning.read_text(encoding="utf-8").replace("(source-b0-run03/", "(../agent-learning-20261003/source-b0-run03/"), encoding="utf-8")
for name in ("README.md", "docs/项目方案与进展简报.md", "docs/B0基线对齐进展.md"):
    p = ROOT / name
    target = backup / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(p, target)
    original = p.read_text(encoding="utf-8-sig")
    link = "docs/p1-20261003/" if name == "README.md" else "p1-20261003/"
    notice = f"> P1 当前状态：72 次请求 + 8 个链路诊断已独立核验；正式第一轮因 get_cluster_doc 的检索依赖故障中止，完整轮数 0/3，无三轮成绩。需要 FAISS-compatible embedding 后端配置；本轮服务已释放，不训练。见 [P1 执行报告]({link}P1-执行与阻塞报告.md)。\n"
    if notice not in original:
        first, rest = original.split("\n", 1)
        p.write_text(first + "\n\n" + notice + rest, encoding="utf-8")
state_path = ROOT / "logs/project-direction.json"
shutil.copy2(state_path, backup / "project-direction.json")
state = load(state_path.parent, state_path.name)
state.update({"updated_utc": datetime.now(timezone.utc).isoformat(), "p1_state": "stability_verified_paired_blocked_by_doc_embeddings",
    "p1_requests": 72, "p1_diagnostic_episodes": 8, "p1_complete_formal_rounds": 0,
    "p1_incomplete_round_episodes": 13, "p1_report": "docs/p1-20261003/P1-执行与阻塞报告.md",
    "training_started": False, "owned_services_stopped": True, "tests_passed": 23,
    "next_phase": "restore_existing_doc_embedding_configuration_then_restart_entire_paired_round1"})
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
jobs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True).strip()
assert not jobs, jobs
cleanup = {"gpu": gpu, "gpu_compute_processes": jobs, "owned_services_stopped": True,
    "simuhome_status": subprocess.check_output(["git", "-C", str(ROOT / "deps/SimuHome"), "status", "--porcelain"], text=True),
    "lightning_diff_stat": subprocess.check_output(["git", "-C", str(ROOT.parent / "agent-lightning"), "diff", "--stat"], text=True)}
assert not cleanup["simuhome_status"]
assert "5 files changed, 43 insertions(+), 7 deletions(-)" in cleanup["lightning_diff_stat"]
(docs / "cleanup_audit.json").write_text(json.dumps(cleanup, indent=2), encoding="utf-8")
shutil.copy2(state_path, docs / "project-direction.json")
source = docs / "source-p1"
for directory in ("scripts", "smarthome_agent_rl", "tests", "configs"):
    shutil.copytree(ROOT / directory, source / directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
device = docs / "device-source"
shutil.copytree(ROOT / "deps/SimuHome/src/simulator/domain/clusters", device / "clusters", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
paths = sorted(ROOT.glob("runs/p1-*")) + [WORK, docs, learning]
archive = ROOT / "runs/p1-partial-handover-20261003.tar.gz"
paths = [p for p in paths if p.is_dir() or p == learning]
subprocess.run(["tar", "-czf", str(archive), *[str(p.relative_to(ROOT)) for p in paths]], cwd=ROOT, check=True)
checksum = {"archive": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size}
(WORK / "partial_handover_checksum.json").write_text(json.dumps(checksum, indent=2))
print(json.dumps({"p1_complete": False, "diagnostic_verified": True, "incomplete_episodes": 13, "archive": checksum, "gpu": gpu}))
