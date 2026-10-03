"""Verify actual recorded episodes, including expected agent task failures."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.reward import check_goals


def verify(run):
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    comparison, config = read("comparison.json"), read("config.json")
    events, triplets = read("lightning_events.json"), read("lightning_triplets.json")
    counts = Counter(e["event_type"] for e in events)
    assert comparison["first_request_identical"]
    assert comparison["final_goal_checks_identical"]
    assert not comparison["upstream_dirty_after"]
    summaries, tool_attempts = {}, {}
    for mode in ["direct", "lightning"]:
        summary = read(f"{mode}/summary.json")
        task = read(f"{mode}/task.json")
        steps = [json.loads(s) for s in (run / mode / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        calls = read(f"{mode}/model_calls.json")
        checks = check_goals(read(f"{mode}/final_state.json"), task["goals"])
        initial = check_goals(read(f"{mode}/initial_state.json"), task["goals"])
        assert checks == summary["checks"]
        achieved = all(c["satisfied"] for c in checks)
        assert summary.get("goal_success", achieved) == achieved
        assert summary["success"] == (achieved and summary["finished"] and summary["execution_completed"])
        assert len(calls) == summary["model_calls"] == len(steps) == summary["steps"]
        assert sum(c["response"]["usage"]["total_tokens"] for c in calls) == summary["total_tokens"]
        assert sum(s["invalid"] for s in steps) == summary["invalid_actions"]
        assert sum(s["infrastructure_error"] is not None for s in steps) == summary["infrastructure_errors"]
        tool_attempts[mode] = sum(e["event"] == "action" for e in read(f"{mode}/upstream_events.json"))
        if summary.get("metrics_version", 1) >= 2:
            assert summary["tool_calls"] == tool_attempts[mode]
        elif summary["tool_calls"] != tool_attempts[mode]:
            assert summary["tool_calls"] == sum(s["event"] == "observation" for s in steps)
        phi = lambda cs: sum(c["satisfied"] for c in cs) / len(cs)
        w = config["reward"]
        expected = (w["success"] * int(summary["finished"] and summary["success"])
            + w["progress"] * (phi(checks) - phi(initial))
            - w["invalid"] * summary["invalid_actions"]
            - w["action_cost"] * len(steps)
            - w["token_cost_per_1k"] * summary["total_tokens"] / 1000)
        assert abs(expected - summary["reward"]) < 1e-9
        summaries[mode] = summary
    migrated = summaries["lightning"]
    assert counts["model_request"] == migrated["model_calls"]
    assert counts["environment_step"] == migrated["steps"]
    assert counts["environment_reset"] == counts["reward"] == 1
    model_events = [e for e in events if e["event_type"] == "model_request"]
    calls = read("lightning/model_calls.json")
    for call, event in zip(calls, model_events):
        assert event["data"]["request"] == {**call["request"], "return_token_ids": True}
    reward = next(e["data"]["value"] for e in events if e["event_type"] == "reward")
    assert abs(reward - migrated["reward"]) < 1e-9
    for event in triplets:
        if event["event_type"] == "model_request":
            assert event["data"]["prompt_token_ids"] and event["data"]["response_token_ids"]
    verification = {"artifact_checks_passed": True, "task_success": migrated["success"],
        "lightning_state": comparison["lightning_state"], "event_counts": dict(counts),
        "reward": reward, "upstream_pristine": True, "first_request_identical": True,
        "gateway_preserves_requests_except_token_id_logging": True,
        "actual_tool_attempts_from_upstream_trace": tool_attempts,
        "note": "Artifact verification passing does not imply the agent completed the task."}
    (run / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification))


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
