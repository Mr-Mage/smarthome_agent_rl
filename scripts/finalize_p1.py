"""Publish verified P1 reports and learning notes, preserving historical evidence."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/p1-20261003"

def main():
    complete = json.loads((WORK / "completed.json").read_text())
    assert complete["complete"]
    report_root = ROOT / complete["report"]
    report = json.loads((report_root / "paired_report.json").read_text())
    diagnostic = json.loads((report_root / "diagnostics_verification.json").read_text())
    assert report["artifact_checks_passed"] and diagnostic["artifact_checks_passed"]
    assert len(report["records"]) == 96
    successes = {h: report["round_statistics"][h]["success"]["values"] for h in ("upstream-react", "structured-schema")}
    invalids = {h: report["round_statistics"][h]["invalid_actions"]["values"] for h in successes}
    token_cost = {h: report["round_statistics"][h]["total_tokens"]["values"] for h in successes}
    diffs = {h: report["round_statistics"][h]["success"]["mean"] for h in successes}
    same = {field: sum(p[field] for p in diagnostic["paired_requests"])
            for field in ("same_text", "same_action", "same_prompt_ids")}
    errors = json.loads((report_root / "error_evidence.json").read_text())
    categories = Counter(e["category"] for row in errors for e in row["evidence"])
    actual = f"B0 三轮严格成功 {successes['upstream-react']} /16；H1 {successes['structured-schema']} /16。平均差 {diffs['structured-schema'] - diffs['upstream-react']:+.2f} 项。"
    template = (WORK / "P1-learning-supplement.md").read_text(encoding="utf-8")
    replacements = {
        "GATEWAY_RESULT": f"72 次固定请求的实际后端请求哈希全部与冻结输入一致；36 对相邻请求中，prompt IDs 一致 {same['same_prompt_ids']}/36、全文一致 {same['same_text']}/36、动作一致 {same['same_action']}/36。8 个链路 episode 的请求、初始目标状态、反馈、最终状态与 reward 复算通过。",
        "STABILITY_RESULT": f"72 次请求与 8 个完整 episode 已完成并独立核验。36 对请求的全文一致 {same['same_text']}/36、动作一致 {same['same_action']}/36。具体逐任务/路径变体数、第一处分叉和 reward 见 P1 稳定性报告；诊断成本为 {json.dumps(diagnostic['cost'], ensure_ascii=False)}。",
        "HARNESS_RESULT": actual + f" 非法动作 B0={invalids['upstream-react']}，H1={invalids['structured-schema']}；总 token B0={token_cost['upstream-react']}，H1={token_cost['structured-schema']}。逐任务变化和成功任务子集成本均已报告。",
        "SCHEDULE_RESULT": "96 个正式 episode、三轮各 32 个；全部原独立 verifier 通过。第 1/3 轮展开排程完全相同，第 2 轮反序和 offset=1 核对通过。" + actual,
        "ERROR_RESULT": f"原有 20 项检查和新增 5 项检查通过。三轮独立核验均通过，完整合法失败保留。错误事件分类为 {json.dumps(dict(categories), ensure_ascii=False)}；内层参数大小写/缺失可由已记录设备结构和固定模拟器方法签名证明，其他只有 HTTP 状态的仍为未确定。诊断首次启动导入冲突和重启端口预检失败各自保留，均未发推理请求。检索修复前第一轮因真实文档检索依赖故障中止，完整 13 个 episode 仍保留，不计入三轮成绩。",
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    assert "{{" not in template
    retrieval = report["retrieval_dependency_repair"]
    template += "\n\n## 21. P1 环境修复：真实本地文档检索\n\n"
    template += "**要验证的问题。** 原 get_cluster_doc 能否真正返回检索文档，而不是因 tokenizer 或 embedding 配置不可用返回 500？\n\n"
    template += "**模块设计。** 原 SimuHome 初始提交包含工具和 Ada/1536 维索引，项目未配置兼容服务；用户选择先完成真实检索，再保持原实验协议。项目用 CPU BGE-small 重建同一上游文档的独立 384 维索引，仍调用原 load_docs、1000/200 切分、FAISS 和原 get_cluster_doc；通过 ToolConfig(db=database) 注入，不改上游源码。模型文件固定 revision 并官方哈希校验，原 Ada 索引保留。两组及三轮共享这项环境修复。\n\n"
    template += "```python\nmodel = load_pinned_BGE(device='cpu')\nchunks = original_split(original_load_docs(), size=1000, overlap=200)\nindex = FAISS(chunks, embed_documents(chunks))\nset_tool_config(ToolConfig(db=index))\n# 原工具继续执行：\nhits = index.similarity_search(query, k=top_k)\nreturn original_observation_format(hits)\n```\n\n"
    template += "**真实结果。** 检索模型、索引哈希与三项真实检索 smoke 检查通过；原工具检索链路和损坏索引拒绝测试通过，总检查 25 项。索引构建成本 " + json.dumps(retrieval["indexing_cost"], ensure_ascii=False) + "。检索修复后重新完成 8 个链路核查，再执行全新三轮 96 个 episode，未沿用失败轮的任务成绩。\n\n"
    template += "**解释边界。** BGE 检索与原 Ada 检索不是同一模型，故 P0 不作为同环境配对样本。这里比较的是同样修复环境中的 B0/H1，非检索算法消融。真实返回文档不保证检索总是准确；retrieval_events.json 单列调用、embedding tokens、CPU 延迟；生成 reward 和生成预算不并入 embedding tokens。之前的 8 个链路以及失败第一轮的成本另行保留为修复前诊断/基础设施开销。\n\n"
    template += "P0 补充纠正：原 P0 summary 中记录了 get_cluster_doc 基础设施错误，此前报告的 0/16 基础设施故障不准确；原轨迹和 12/16 的真实设备成功结论保留，只作为历史参照。\n"
    backup = WORK / "handover-before-finalization"
    backup.mkdir(exist_ok=True)
    learning_directory = ROOT / "docs/agent-learning-20261003"
    learning_directory.mkdir(parents=True, exist_ok=True)
    learning = learning_directory / "Agent模块设计与Lightning基线学习文档.md"
    base = learning.read_text(encoding="utf-8-sig") if learning.exists() else (WORK / "learning-base.md").read_text(encoding="utf-8-sig")
    if learning.exists() and not (backup / learning.name).exists():
        shutil.copy2(learning, backup / learning.name)
    if "## 16. P1 学习模块" in base:
        base = base[:base.index("## 16. P1 学习模块")]
    base = base.replace("尚未完成 9B 重复研究，也没有 9B direct/Lightning 的逐轮配对验证", "P1 已完成 72 次固定请求、8 个 direct/Lightning 链路诊断和三轮 96 个正式 episode，独立核验通过")
    base = base.replace("后续先检查稳定性，再开展同模型、同任务、同预算的 harness 对照，随后才准备 SFT 和 RL。", "P1 已完成稳定性和同模型、同任务、同预算的 harness 对照；接下来根据失败证据决定设备能力或错误反馈对照，再规划正式数据划分、SFT 和 RL。")
    base = base.replace("| 9B 重复稳定性 / direct 与 Lightning 配对 | 尚未完成 |", "| 9B 重复稳定性 / direct 与 Lightning 配对 | P1 完成，72 次请求 + 8 个 episode 独立核验 |")
    base = base.replace("| 9B 的 structured harness 对照 | 尚未开展 |", "| 9B 的 structured harness 对照 | P1 完成，96 个 episode、三轮独立核验 |")
    base = base.replace("| 9B 的 structured harness 对照 | 第一轮因检索基础设施中止，0/3 完整轮 |", "| 9B 的 structured harness 对照 | P1 完成，96 个 episode、三轮独立核验 |")
    learning.write_text(base.rstrip() + "\n\n" + template, encoding="utf-8")
    docs = ROOT / "docs/p1-20261003"
    docs.mkdir(parents=True, exist_ok=True)
    for p in report_root.iterdir():
        if p.is_file():
            shutil.copy2(p, docs / p.name)
    old_status = docs / "P1-status.json"
    if old_status.exists() and not (WORK / "status-before-completion.json").exists():
        shutil.copy2(old_status, WORK / "status-before-completion.json")
    old_status.write_text(json.dumps({"p1_complete": True, "fixed_requests": 72,
        "current_environment_chain_episodes": 8, "additional_pre_retrieval_chain_episodes": 8,
        "formal_episodes": 96, "complete_formal_rounds": 3, "tests_passed": 25,
        "rounds": complete["rounds"], "successes": successes, "training": False,
        "historical_incomplete_round_episodes": 13, "historical_incomplete_included_in_statistics": False,
        "retrieval_repaired": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    blocked_report = docs / "P1-执行与阻塞报告.md"
    if blocked_report.exists():
        text = blocked_report.read_text(encoding="utf-8")
        if "修复前历史记录" not in text:
            first, rest = text.split("\n", 1)
            blocked_report.write_text("# P1 修复前历史记录\n\n> 当前 P1 已完成真实本地检索修复和三轮验收，详见 P1-三轮配对报告.md。以下内容保留修复前阻塞状态，不能作为当前状态。\n" + rest, encoding="utf-8")
    learning_copy = docs / learning.name
    learning_copy.write_text(learning.read_text(encoding="utf-8").replace("(source-b0-run03/", "(../agent-learning-20261003/source-b0-run03/"), encoding="utf-8")
    notice = f"> P1 已完成：72 次固定请求、8 个链路诊断及 96 个三轮配对 episode 均独立核验。{actual} 不训练；详见 [P1 三轮报告](p1-20261003/P1-三轮配对报告.md) 和 [稳定性报告](p1-20261003/P1-稳定性报告.md)。后文旧进度保留为历史记录。\n"
    for name in ("README.md", "docs/项目方案与进展简报.md", "docs/B0基线对齐进展.md"):
        p = ROOT / name
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(p, target)
        text = p.read_text(encoding="utf-8-sig")
        text = text.replace("> P1 当前状态：72 次请求 + 8 个链路诊断已独立核验；正式第一轮", "> P1 修复前历史状态：72 次请求 + 8 个链路诊断已独立核验；正式第一轮")
        text = text.replace("9B 重复稳定性尚待验证。", "9B 重复稳定性与三轮 harness 对照已在 P1 完成，详见顶部报告。")
        local_notice = notice.replace("(p1-20261003/", "(docs/p1-20261003/") if name == "README.md" else notice
        first, rest = text.split("\n", 1)
        if local_notice not in text:
            text = first + "\n\n" + local_notice + rest
        text = text.replace("下一阶段为重复稳定性检查与最小 harness 对照，尚未启动。", "P1 稳定性与最小 schema 对照已完成，见本文顶部的新报告。")
        text = text.replace("本轮未启动 P1、SFT 或 GRPO。", "P1 已完成；未启动 SFT 或 GRPO。")
        p.write_text(text, encoding="utf-8")
    state_file = ROOT / "logs/project-direction.json"
    if not (backup / "project-direction.json").exists():
        shutil.copy2(state_file, backup / "project-direction.json")
    state = json.loads(state_file.read_text(encoding="utf-8-sig"))
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    gpu_jobs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True).strip()
    assert not gpu_jobs, gpu_jobs
    processes = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid,ppid,args"], text=True)
    active = [line for line in processes.splitlines() if any(marker in line for marker in (
        "vllm.entrypoints.openai.api_server", "uvicorn src.simulator.api.app:app", "agl-controller", "agl-server", "run_harness_suite.py", "serve_doc_embeddings.py"))]
    assert not active, active
    state.update({"updated_utc": datetime.now(timezone.utc).isoformat(), "p1_state": "completed_and_verified",
        "p1_requests": 72, "p1_diagnostic_episodes": 8, "p1_formal_episodes": 96,
        "p1_runs": complete["rounds"], "p1_successes": successes, "p1_report": "docs/p1-20261003/P1-三轮配对报告.md",
        "tests_passed": 25, "training_started": False, "owned_services_stopped": True,
        "next_phase": "decide_device_capability_or_error_feedback_comparison_then_plan_data_split_and_SFT"})
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    cleanup = {"gpu": gpu, "gpu_compute_processes": gpu_jobs, "owned_services_stopped": True, "active_project_services": active,
        "simuhome_status": subprocess.check_output(["git", "-C", str(ROOT / "deps/SimuHome"), "status", "--porcelain"], text=True),
        "lightning_diff_stat": subprocess.check_output(["git", "-C", str(ROOT.parent / "agent-lightning"), "diff", "--stat"], text=True),
        "lightning_status": subprocess.check_output(["git", "-C", str(ROOT.parent / "agent-lightning"), "status", "--porcelain"], text=True)}
    assert not cleanup["simuhome_status"]
    assert "5 files changed, 43 insertions(+), 7 deletions(-)" in cleanup["lightning_diff_stat"]
    (docs / "cleanup_audit.json").write_text(json.dumps(cleanup, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(state_file, docs / "project-direction.json")
    project_documents = docs / "server-project-documents"
    for name in ("README.md", "docs/项目方案与进展简报.md", "docs/B0基线对齐进展.md"):
        target = project_documents / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    source = docs / "source-p1"
    for directory in ("smarthome_agent_rl", "scripts", "tests", "configs"):
        shutil.copytree(ROOT / directory, source / directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    gateway = docs / "gateway-source"
    if not gateway.exists():
        shutil.copytree(ROOT.parent / "agent-lightning/agentlightning/server", gateway / "server", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT.parent / "agent-lightning/agentlightning/controller/local_reconciler.py", gateway / "local_reconciler.py")
    device = docs / "device-source"
    if not device.exists():
        shutil.copytree(ROOT / "deps/SimuHome/src/simulator/domain/clusters", device / "clusters", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "deps/SimuHome/src/agents/tools.py", device / "tools.py")
    retrieval_docs = docs / "retrieval-source-docs"
    if not retrieval_docs.exists():
        shutil.copytree(ROOT / "deps/SimuHome/docs/clusters", retrieval_docs)
    for name in ("retrieval-model-verification.json", "embedding-venv-isolation.json"):
        shutil.copy2(WORK / name, docs / name)
    shutil.copy2(WORK / "doc-index-bge/verification.json", docs / "retrieval-index-verification.json")
    paths = sorted(p for p in ROOT.glob("runs/p1-*") if p.is_dir()) + [WORK, docs, learning]
    manifests = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for folder in paths if folder.is_dir() for p in folder.rglob("*") if p.is_file() and p != docs / "evidence_manifest.json" and "__pycache__" not in p.parts and "embedding-venv" not in p.parts}
    (docs / "evidence_manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = ROOT / "runs/p1-handover-20261003.tar.gz"
    subprocess.run(["tar", "--exclude=work/p1-20261003/embedding-venv", "-czf", str(archive), *[str(p.relative_to(ROOT)) for p in paths]], cwd=ROOT, check=True)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (WORK / "handover_checksum.json").write_text(json.dumps({"archive": str(archive), "sha256": checksum, "bytes": archive.stat().st_size}, indent=2))
    print(json.dumps({"complete": True, "successes": successes, "archive": str(archive), "sha256": checksum, "gpu": gpu}, ensure_ascii=False))

if __name__ == "__main__":
    main()
