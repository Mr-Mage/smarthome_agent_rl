"""Independently check saved paired trials against actual state and Gateway events."""
import ast
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.reward import check_goals
from smarthome_agent_rl.generation import generation_options


def verify(run):
    load = lambda path: json.loads((run / path).read_text(encoding="utf-8"))
    comparison, config = load("comparison.json"), load("config.json")
    generation = generation_options(config)
    if (run / "run_protocol.json").exists():
        protocol = load("run_protocol.json")
        assert protocol["generation"] == generation and protocol["training"] is False
        assert protocol["config_sha256"] == hashlib.sha256((run / "config.json").read_bytes()).hexdigest()
        assert protocol["task_sha256"] == hashlib.sha256((run / "tasks.json").read_bytes()).hexdigest()
    results = comparison["results"]
    source_tree = ast.parse((run / "code/smarthome_agent_rl/structured.py").read_text(encoding="utf-8"))
    profile_settings = next((ast.literal_eval(n.value) for n in source_tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "HARNESS_SETTINGS" for t in n.targets)), None)
    harnesses = load("harnesses.json") if (run / "harnesses.json").exists() else ["upstream-react", "structured-v1"]
    if (run / "run_protocol.json").exists():
        assert protocol["harnesses"] == harnesses
        expected_tasks = {t["task_id"]: t for t in load("tasks.json")}
        assert protocol["task_ids"] == list(expected_tasks)
        assert len(results) == len(expected_tasks) * len(harnesses)
        assert {(r["task_id"], r["harness"]) for r in results} == {
            (task_id, h) for task_id in expected_tasks for h in harnesses}
    if (run / "schedule.json").exists():
        assert load("purpose.json")["purpose"] == "paired_formal"
        actual = load("schedule.json")
        assert [(r["task_id"], r["harness"]) for r in results] == [(r["task_id"], r["harness"]) for r in actual]
        assert [r["position"] for r in actual] == list(range(len(results)))
    pairs, aggregation = {}, {}
    for record in results:
        directory = Path(record["path"])
        prefix = directory / "episode"
        summary, task = load(prefix / "summary.json"), load(prefix / "task.json")
        if (run / "run_protocol.json").exists():
            assert task == expected_tasks[record["task_id"]]
            contract = load(prefix / "upstream_contract.json")
            assert contract["generation"] == generation
            assert contract["baseline_id"] == protocol["baseline_id"] == summary["baseline_id"]
        if (run / "purpose.json").exists() and config.get("run_purpose") == "p1_paired_dev_schema":
            assert summary["infrastructure_errors"] == 0 and record["lightning_state"] == "succeeded"
            assert summary["error"] is None or summary["task_failure"]
        initial, final = load(prefix / "initial_state.json"), load(prefix / "final_state.json")
        calls, events = load(prefix / "model_calls.json"), load(directory / "lightning_events.json")
        steps = [json.loads(s) for s in (run / prefix / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        checks, before = check_goals(final, task["goals"]), check_goals(initial, task["goals"])
        assert checks == summary["checks"] and summary == record["summary"]
        achieved = all(c["satisfied"] for c in checks)
        assert summary.get("goal_success", achieved) == achieved
        assert summary["success"] == (achieved and summary["finished"] and summary["execution_completed"])
        if config.get("lightning_agent_class") == "smarthome_agent_rl.baseline:BaselineAgent":
            assert summary["task_failure_is_data"] is True
            assert record["lightning_state"] == "succeeded"
        assert len(calls) == len(steps) == summary["steps"] == summary["model_calls"]
        assert len(calls) <= config["max_steps"]
        assert summary["total_tokens"] == sum(c["response"]["usage"]["total_tokens"] for c in calls)
        assert summary["invalid_actions"] == sum(s["invalid"] for s in steps)
        traces = load(prefix / "upstream_events.json")
        assert sum(e["event"] == "finish" for e in traces) == int(summary["finished"])
        assert summary["tool_calls"] == sum(e["event"] == "action" for e in traces)
        phi = lambda cs: sum(c["satisfied"] for c in cs) / len(cs)
        w = config["reward"]
        reward = (w["success"] * int(summary["success"] and summary["finished"])
            + w["progress"] * (phi(checks) - phi(before)) - w["invalid"] * summary["invalid_actions"]
            - w["action_cost"] * len(steps) - w["token_cost_per_1k"] * summary["total_tokens"] / 1000)
        assert abs(summary["reward"] - reward) < 1e-9
        rewards = [e for e in events if e["event_type"] == "reward"]
        model_events = [e for e in events if e["event_type"] == "model_request"]
        assert len(rewards) == 1 and abs(rewards[0]["data"]["value"] - reward) < 1e-9
        assert len(model_events) == len(calls)
        assert sum(e["event_type"] == "environment_reset" for e in events) == 1
        assert sum(e["event_type"] == "environment_step" for e in events) == len(steps)
        for call, event in zip(calls, model_events):
            assert event["data"]["request"] == {**call["request"], "return_token_ids": True}
            assert call["request"]["model"] == config["served_model"]
            assert call["request"]["seed"] == config["model_seed"]
            for key, value in generation.items():
                if key == "extra_body":
                    assert all(call["request"].get(k) == v for k, v in value.items())
                else:
                    assert call["request"].get(key) == value
        triplets = load(directory / "lightning_triplets.json")
        assert sum(e["event_type"] == "model_request" for e in triplets) == len(calls)
        for event in triplets:
            if event["event_type"] == "model_request":
                assert event["data"]["prompt_token_ids"] and event["data"]["response_token_ids"]
        if record["harness"].startswith("structured-"):
            audit = load(prefix / "harness_events.json")
            assistant_events = [e for e in traces if e["event"] == "assistant"]
            assert len(audit) == len(calls)
            assert len(assistant_events) == len(calls)
            assert all(a["extra_model_calls"] == 0 for a in audit)
            for call, step, entry, assistant in zip(calls, steps, audit, assistant_events):
                body = json.loads(call["response"]["choices"][0]["message"]["content"])
                assert body == step["model_action"]
                assert entry["normalized_action"] == step["action"]
                assert json.loads(assistant["payload"]) == step["action"]
                if profile_settings is not None:
                    assert entry["settings"] == profile_settings[record["harness"]]
                allowed = {s["properties"]["tool"]["const"] for s in call["request"]["response_format"]["json_schema"]["schema"]["properties"]["call"]["anyOf"]}
                registry = set(load(prefix / "upstream_contract.json")["tool_names"])
                assert allowed == registry - ({"finish"} if not entry["finish_enabled"] else set())
                if step["action"]["action"] == "finish":
                    assert entry["finish_enabled"]
        pairs.setdefault(task["task_id"], []).append({"task": task, "initial_checks": before})
        ss = aggregation.setdefault(record["harness"], [])
        ss.append(summary)
    for pair in pairs.values():
        assert len(pair) == len(harnesses) and all(entry == pair[0] for entry in pair)
    recomputed = {}
    for harness, ss in aggregation.items():
        recomputed[harness] = {"episodes": len(ss), "successes": sum(s["success"] for s in ss),
            "success_rate": sum(s["success"] for s in ss) / len(ss),
            "model_calls": sum(s["model_calls"] for s in ss), "invalid_actions": sum(s["invalid_actions"] for s in ss),
            "tokens": sum(s["total_tokens"] for s in ss), "reward_sum": sum(s["reward"] for s in ss)}
    assert recomputed == comparison["aggregation"]
    verification = {"artifact_checks_passed": True, "tasks": len(pairs),
        "paired_tasks": len(pairs) if len(harnesses) > 1 else 0, "episodes": len(results),
        "harnesses": harnesses, "generation_settings_verified": True,
        "same_tasks_and_initial_goal_states": True, "same_model_seed_and_call_budget": True,
        "reward_and_actual_state_verified": True, "gateway_requests_and_token_ids_verified": True,
        "aggregation": recomputed, "note": "Small dev suite; no benchmark or causal isolation of individual harness changes."}
    (run / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
