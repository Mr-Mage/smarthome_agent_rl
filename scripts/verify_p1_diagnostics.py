"""Independent P1 diagnostic audit; variability is reported, not rejected."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.p1 import freeze, digest, normalize
from smarthome_agent_rl.reward import check_goals

def read(root, p):
    return json.loads((root / p).read_text(encoding="utf-8"))

def verify(initial, restart, baseline, output, chain_run=None):
    expected = freeze(baseline)
    entries = {e["task"]["task_id"]: e for e in expected["entries"]}
    records = []
    for directory, blocks, count in ((initial, [1, 2], 48), (restart, [3], 24)):
        assert read(directory, "purpose.json")["purpose"] == "diagnostic_only"
        assert not (directory / "comparison.json").exists()
        assert read(directory, "frozen_requests.json") == expected
        complete = read(directory, "diagnostic_complete.json")
        assert complete["blocks"] == blocks and complete["requests"] == count
        rows = read(directory, "request_records.json")
        assert len(rows) == count and len(read(directory, "diagnostic_schedule.json")) == count
        assert [r["position"] for r in rows] == list(range(count))
        assert all(r["service_start"] == read(directory, "service_start.json") for r in rows)
        coverage = Counter((r["block"], r["task_id"], r["path"]) for r in rows)
        assert coverage == Counter({(b, t, p): 3 for b in blocks for t in entries for p in ("direct", "lightning")})
        for b in blocks:
            actual_order = list(dict.fromkeys(r["task_id"] for r in rows if r["block"] == b))
            assert actual_order == (list(reversed(entries)) if b == 2 else list(entries))
        for i in range(0, count, 2):
            assert [r["path"] for r in rows[i:i+2]] == (["direct", "lightning"] if i // 2 % 2 == 0 else ["lightning", "direct"])
        for r in rows:
            assert r["purpose"] == "diagnostic_only"
            assert r["backend_request"] == entries[r["task_id"]]["backend_request"]
            assert r["backend_sha256"] == digest(r["backend_request"])
            assert r["normalized_action"] == normalize(r["reply_text"])
            assert r["prompt_token_ids"] and r["response_token_ids"]
            assert r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
            prefix = f"requests/{r['position']:02d}-{r['task_id']}-{r['path']}"
            assert read(directory, prefix + "/record.json") == r
            if r["path"] == "lightning":
                events = read(directory, prefix + "/lightning_events.json")
                model = [e for e in events if e["event_type"] == "model_request"]
                assert len(model) == 1 and model[0]["data"]["request"] == r["backend_request"]
                assert model[0]["data"]["response"] == r["response"]
                assert not any(e["event_type"].startswith("environment_") or e["event_type"] == "reward" for e in events)
                tokens = [e for e in read(directory, prefix + "/lightning_triplets.json") if e["event_type"] == "model_request"]
                assert len(tokens) == 1
                assert tokens[0]["data"]["prompt_token_ids"] == r["prompt_token_ids"]
                assert tokens[0]["data"]["response_token_ids"] == r["response_token_ids"]
        records += rows
    assert read(initial, "service_start.json") != read(restart, "service_start.json")
    groups = []
    for task in entries:
        for path in ("direct", "lightning"):
            rs = [r for r in records if r["task_id"] == task and r["path"] == path]
            groups.append({"task_id": task, "path": path, "requests": len(rs),
                "unique_texts": len({r["reply_text"] for r in rs}),
                "unique_actions": len({digest(r["normalized_action"]) for r in rs}),
                "unique_prompt_ids": len({digest(r["prompt_token_ids"]) for r in rs}),
                "unique_response_ids": len({digest(r["response_token_ids"]) for r in rs})})
    pairs = []
    for b in (1, 2, 3):
        for task in entries:
            for rep in range(3):
                rs = {r["path"]: r for r in records if r["block"] == b and r["task_id"] == task and r["repeat"] == rep}
                a, z = rs["direct"], rs["lightning"]
                pairs.append({"block": b, "task_id": task, "repeat": rep,
                    "same_text": a["reply_text"] == z["reply_text"],
                    "same_action": a["normalized_action"] == z["normalized_action"],
                    "same_prompt_ids": a["prompt_token_ids"] == z["prompt_token_ids"],
                    "same_response_ids": a["response_token_ids"] == z["response_token_ids"]})
    if chain_run is not None:
        original_restart = restart
        restart = chain_run
        assert read(restart, "purpose.json")["purpose"] == "diagnostic_only"
        assert read(restart, "diagnostic_complete.json")["episodes"] == 8
    episodes = read(restart, "chain_results.json")
    assert len(episodes) == 8
    assert {(r["task_id"], r["path"]) for r in episodes} == {(t, p) for t in entries for p in ("direct", "lightning")}
    chain = []
    for task_id, entry in entries.items():
        modes = {}
        for path in ("direct", "lightning"):
            row = next(r for r in episodes if r["task_id"] == task_id and r["path"] == path)
            folder = row["directory"] + "/episode/"
            load = lambda p: read(restart, folder + p)
            task, summary, calls = load("task.json"), load("summary.json"), load("model_calls.json")
            assert task == entry["task"] and summary == row["summary"]
            initial_checks = check_goals(load("initial_state.json"), task["goals"])
            checks = check_goals(load("final_state.json"), task["goals"])
            assert checks == summary["checks"]
            assert summary["goal_success"] == all(c["satisfied"] for c in checks)
            assert summary["success"] == (summary["goal_success"] and summary["finished"] and summary["execution_completed"])
            steps = [json.loads(line) for line in (restart / (folder + "trajectory.jsonl")).read_text().splitlines()]
            assert len(steps) == len(calls) == summary["model_calls"] == summary["steps"] <= 20
            assert summary["total_tokens"] == sum(c["response"]["usage"]["total_tokens"] for c in calls)
            assert summary["invalid_actions"] == sum(s["invalid"] for s in steps)
            assert not summary["infrastructure_errors"]
            assert calls[0]["request"] == (entry["backend_request"] if path == "direct" else entry["request"])
            w = read(restart, "config.json")["reward"]
            phi = lambda cs: sum(c["satisfied"] for c in cs) / len(cs)
            reward = w["success"] * int(summary["success"] and summary["finished"]) + w["progress"] * (phi(checks) - phi(initial_checks)) - w["invalid"] * summary["invalid_actions"] - w["action_cost"] * len(steps) - w["token_cost_per_1k"] * summary["total_tokens"] / 1000
            assert abs(reward - summary["reward"]) < 1e-9
            if path == "lightning":
                events = read(restart, row["directory"] + "/lightning_events.json")
                models = [e for e in events if e["event_type"] == "model_request"]
                assert len(models) == len(calls)
                for c, e in zip(calls, models):
                    assert e["data"]["request"] == {**c["request"], "return_token_ids": True}
                assert [e["data"]["value"] for e in events if e["event_type"] == "reward"] == [summary["reward"]]
                counts = Counter(e["event_type"] for e in events)
                assert counts["environment_reset"] == counts["reward"] == 1
                assert counts["environment_step"] == len(steps)
                triplets = read(restart, row["directory"] + "/lightning_triplets.json")
                assert all(e["data"]["prompt_token_ids"] and e["data"]["response_token_ids"] for e in triplets if e["event_type"] == "model_request")
            modes[path] = {"initial_checks": initial_checks, "summary": summary, "steps": steps}
        assert modes["direct"]["initial_checks"] == modes["lightning"]["initial_checks"]
        historical = read(baseline, entry["source_path"] + "/episode/initial_state.json")
        assert modes["direct"]["initial_checks"] == check_goals(historical, entry["task"]["goals"])
        divergences = {}
        for field in ("action", "model_action", "observation", "state", "checks"):
            a, z = modes["direct"]["steps"], modes["lightning"]["steps"]
            divergences[field] = next((i + 1 for i in range(max(len(a), len(z)))
                if i >= min(len(a), len(z)) or (normalize(json.dumps(a[i][field])) if field == "action" else a[i][field]) !=
                (normalize(json.dumps(z[i][field])) if field == "action" else z[i][field])), None)
        chain.append({"task_id": task_id, "initial_goal_states_equal": True,
            "first_backend_request_equal": True, "first_divergence_turn": divergences,
            "summaries": {p: modes[p]["summary"] for p in modes}})
    report = {"artifact_checks_passed": True, "purpose": "diagnostic_only", "requests": 72,
        "episodes": 8, "groups": groups, "paired_requests": pairs, "chains": chain,
        "cost": {"fixed_prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in records),
            "fixed_completion_tokens": sum(r["usage"]["completion_tokens"] for r in records),
            "fixed_latency_seconds": sum(r["duration_seconds"] for r in records),
            "chain_model_calls": sum(r["summary"]["model_calls"] for r in episodes),
            "chain_tokens": sum(r["summary"]["total_tokens"] for r in episodes),
            "chain_duration_seconds": sum(r["summary"]["duration_seconds"] for r in episodes)},
        "limits": "Order, warm state and restart co-vary; differences do not identify a causal mechanism."}
    if chain_run is not None:
        old_episodes = read(original_restart, "chain_results.json")
        report["additional_pre_retrieval_chain_cost"] = {
            "episodes": 8, "model_calls": sum(r["summary"]["model_calls"] for r in old_episodes),
            "tokens": sum(r["summary"]["total_tokens"] for r in old_episodes),
            "duration_seconds": sum(r["summary"]["duration_seconds"] for r in old_episodes)}
        report["chain_run"] = str(chain_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"diagnostics_verified": True, "requests": 72, "episodes": 8}))
    return report

if __name__ == "__main__":
    verify(*(Path(p).resolve() for p in sys.argv[1:]))
