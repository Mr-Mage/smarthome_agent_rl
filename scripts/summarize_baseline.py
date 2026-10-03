"""Summarize verified B0 artifacts, separating task outcome from worker status."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path


def summarize(run):
    load = lambda p: json.loads((run / p).read_text(encoding="utf-8"))
    verified = load("verification.json")
    if verified.get("artifact_checks_passed") is not True:
        raise ValueError("Independent artifact verification is required first")
    config, comparison = load("config.json"), load("comparison.json")
    if load("harnesses.json") != ["upstream-react"] or config.get("baseline_id") != "B0-ReAct-v1":
        raise ValueError("This report is for the frozen single-harness B0 only")
    families, outcomes, records = defaultdict(list), Counter(), []
    for row in comparison["results"]:
        folder = Path(row["path"]) / "episode"
        task, summary = load(folder / "task.json"), row["summary"]
        trajectory = [json.loads(line) for line in (run / folder / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        calls = load(folder / "model_calls.json")
        if summary["success"]:
            outcome = "success"
        elif summary["goal_success"] and not summary["finished"]:
            outcome = "goals_met_without_finish"
        elif summary["task_failure"]:
            outcome = "agent_limit_or_rejection"
        elif summary["finished"]:
            outcome = "finished_with_unsatisfied_goals"
        else:
            outcome = "execution_error"
        def action_signature(step):
            action = step["action"]
            payload = action.get("action_input")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    pass
            return json.dumps({"action": action.get("action"), "arguments": payload,
                "unparsed": action.get("unparsed")}, sort_keys=True)
        failures = Counter(action_signature(step) for step in trajectory if step["invalid"])
        record = {"task_id": task["task_id"], "family": task.get("family", "unspecified"),
            "outcome": outcome, "success": summary["success"], "goal_success": summary["goal_success"],
            "finished": summary["finished"], "lightning_state": row["lightning_state"],
            "model_calls": summary["model_calls"], "invalid_actions": summary["invalid_actions"],
            "prompt_tokens": sum(c["response"]["usage"]["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["response"]["usage"]["completion_tokens"] for c in calls),
            "tokens": summary["total_tokens"], "reward": summary["reward"],
            "abort_calls_without_feedback": summary["model_rejections_without_feedback"],
            "repeated_identical_invalid_actions": sum(max(0, n - 1) for n in failures.values())}
        records.append(record)
        families[record["family"]].append(record)
        outcomes[outcome] += 1
    family_summary = {name: {"episodes": len(items), "successes": sum(r["success"] for r in items),
        "goal_successes": sum(r["goal_success"] for r in items),
        "model_calls": sum(r["model_calls"] for r in items),
        "invalid_actions": sum(r["invalid_actions"] for r in items)} for name, items in families.items()}
    report = {"captured_utc": datetime.now(timezone.utc).isoformat(), "baseline_id": config["baseline_id"],
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_purpose": config["run_purpose"], "actual_model": config["model_path"],
        "target_model": config.get("target_model"), "generation": load("run_protocol.json")["generation"],
        "aggregation": comparison["aggregation"], "outcomes": dict(outcomes),
        "families": family_summary, "records": records, "training": False, "official_reproduction": False,
        "note": "1.5B engineering runs are not Qwen3.5-9B results; task failure and Lightning worker failure are reported separately."}
    (run / "baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# B0 自建基线记录", "", f"用途：{config['run_purpose']}。实际模型：{config['model_path']}。",
        "", "本报告只整理已独立核验的运行，不是原论文复现；1.5B 工程验证不能作为 9B 成绩。",
        "", "|任务族|成功 / episode|达到环境目标|调用|非法动作|", "|---|---:|---:|---:|---:|"]
    for name, item in family_summary.items():
        lines.append(f"|{name}|{item['successes']}/{item['episodes']}|{item['goal_successes']}|{item['model_calls']}|{item['invalid_actions']}|")
    lines += ["", "|任务|结果|Lightning 状态|调用|非法动作|tokens|", "|---|---|---|---:|---:|---:|"]
    for item in records:
        lines.append(f"|{item['task_id']}|{item['outcome']}|{item['lightning_state']}|{item['model_calls']}|{item['invalid_actions']}|{item['tokens']}|")
    lines += ["", "指标与完整配置见 baseline_report.json、run_protocol.json 和 verification.json。失败轨迹全部保留。"]
    (run / "baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_id": report["baseline_id"], "outcomes": report["outcomes"],
        "families": family_summary, "aggregation": report["aggregation"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    summarize(Path(parser.parse_args().run_dir).resolve())
