"""P1 diagnostic workload, using run_harness_suite's verified service lifecycle."""
import json
import os
from pathlib import Path
import time
import httpx
from smarthome_agent_rl.p1 import freeze, digest, normalize

ROOT = Path(__file__).resolve().parents[1]

def perform(mode, baseline, run, config, api, launch, controller, save):
    frozen = freeze(baseline)
    save("frozen_requests.json", frozen)
    service = json.loads((run / "service_start.json").read_text())
    def rollout(input_data, cls, env_map, prefix):
        created = api("POST", "/api/rollouts", [{"input": input_data, "is_train": False,
            "metadata": {"purpose": "diagnostic_only"}, "config": {"timeout_seconds": 600,
            "local": {"agent_class": cls, "env_map": env_map}}}])
        save(prefix + "/rollout_created.json", created)
        rid = created[0]["rollout_id"]
        deadline = time.monotonic() + 630
        while time.monotonic() < deadline:
            if controller.poll() is not None:
                raise RuntimeError("Controller exited")
            detail = api("GET", f"/api/rollouts/{rid}")
            if detail["rollout"]["status"]["state"] in {"succeeded", "failed"}:
                break
            time.sleep(1)
        else:
            raise TimeoutError(rid)
        save(prefix + "/rollout.json", detail)
        events = api("GET", f"/api/rollouts/{rid}/events")
        save(prefix + "/lightning_events.json", events)
        triplets = api("GET", f"/api/rollouts/{rid}/events?format=triplet")
        save(prefix + "/lightning_triplets.json", triplets)
        assert detail["rollout"]["status"]["state"] == "succeeded", detail
        return events, triplets

    records, expanded = [], []
    blocks = [(1, False), (2, True)] if mode == "initial" else [(3, False)] if mode == "restart" else []
    for block, reverse in blocks:
        entries = list(reversed(frozen["entries"])) if reverse else frozen["entries"]
        for entry in entries:
            for repeat in range(3):
                paths = ["direct", "lightning"] if (len(records) // 2) % 2 == 0 else ["lightning", "direct"]
                for path in paths:
                    index = len(records)
                    prefix = f"requests/{index:02d}-{entry['task']['task_id']}-{path}"
                    (run / prefix).mkdir(parents=True)
                    expanded.append({"position": index, "block": block, "repeat": repeat,
                        "task_id": entry["task"]["task_id"], "path": path,
                        "backend_sha256": entry["backend_sha256"], "service_start": service})
                    save("diagnostic_schedule.json", expanded)
                    if path == "direct":
                        with httpx.Client(trust_env=False, timeout=120) as client:
                            started = time.monotonic()
                            response = client.post(config["model_endpoint"] + "/chat/completions",
                                json=entry["backend_request"])
                            elapsed = time.monotonic() - started
                            response.raise_for_status()
                            body = response.json()
                        backend_request = entry["backend_request"]
                        prompt_ids = body["prompt_token_ids"]
                        response_ids = body["choices"][0]["token_ids"]
                    else:
                        output = run / prefix / "reply.json"
                        events, triplets = rollout({"request": entry["request"], "output": str(output)},
                            "smarthome_agent_rl.p1_replay:FrozenRequestAgent",
                            {"P1_REQUEST": "input.request", "P1_OUTPUT": "input.output"}, prefix)
                        reply = json.loads(output.read_text())
                        body, elapsed = reply["response"], reply["duration_seconds"]
                        model_event = next(e for e in events if e["event_type"] == "model_request")
                        backend_request = model_event["data"]["request"]
                        token_event = next(e for e in triplets if e["event_type"] == "model_request")
                        prompt_ids = token_event["data"]["prompt_token_ids"]
                        response_ids = token_event["data"]["response_token_ids"]
                    assert digest(backend_request) == entry["backend_sha256"]
                    assert prompt_ids and response_ids
                    text = body["choices"][0]["message"]["content"]
                    record = {**expanded[-1], "backend_request": backend_request, "response": body,
                        "reply_text": text, "normalized_action": normalize(text), "usage": body["usage"],
                        "prompt_token_ids": prompt_ids, "response_token_ids": response_ids,
                        "duration_seconds": elapsed, "purpose": "diagnostic_only"}
                    save(prefix + "/record.json", record)
                    records.append(record)
                    save("request_records.json", records)
                    print(json.dumps({"diagnostic_request": index, "block": block, "path": path,
                                      "task": entry["task"]["task_id"]}), flush=True)
    if mode in {"restart", "chain"}:
        episodes = []
        for i, entry in enumerate(frozen["entries"]):
            task = entry["task"]
            for path in (["direct", "lightning"] if i % 2 == 0 else ["lightning", "direct"]):
                prefix = f"chain/{task['task_id']}/{path}"
                output = run / prefix / "episode"
                output.parent.mkdir(parents=True, exist_ok=True)
                variant = {**config, "harness": "upstream-react", "diagnostic_return_token_ids": True}
                if path == "direct":
                    proc = launch("chain-" + task["task_id"], [ROOT / ".venv-baseline/bin/python",
                        ROOT / "scripts/run_official_episode.py", "--mode", "direct", "--task-failure-is-data"],
                        ROOT, {"SMARTHOME_TASK": json.dumps(task), "SMARTHOME_CONFIG": json.dumps(variant),
                        "SMARTHOME_OUTPUT": str(output), "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "deps/SimuHome")})
                    assert proc.wait(timeout=600) == 0
                else:
                    rollout({"task": task, "config": variant, "output": str(output)},
                        config["lightning_agent_class"], {"SMARTHOME_TASK": "input.task",
                        "SMARTHOME_CONFIG": "input.config", "SMARTHOME_OUTPUT": "input.output"}, prefix)
                summary = json.loads((output / "summary.json").read_text())
                assert not summary["infrastructure_errors"]
                assert summary["error"] is None or summary["task_failure"]
                episodes.append({"task_id": task["task_id"], "path": path, "directory": prefix,
                                 "summary": summary, "purpose": "diagnostic_only"})
                save("chain_results.json", episodes)
                print(json.dumps({"chain_episode": task["task_id"], "path": path, "success": summary["success"]}), flush=True)
    save("diagnostic_complete.json", {"purpose": "diagnostic_only", "requests": len(records),
        "episodes": 8 if mode in {"restart", "chain"} else 0, "blocks": [b for b, r in blocks]})
