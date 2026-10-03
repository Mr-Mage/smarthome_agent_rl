"""Pure scheduling, request freezing and repeated paired summaries for P1."""
import copy
import hashlib
import json
import statistics

B0, H1 = "upstream-react", "structured-schema"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()

def schedule(tasks, harnesses, task_order="forward", offset=0):
    ordered = tasks if task_order == "forward" else list(reversed(tasks))
    return [{"position": len(harnesses) * i + j, "task_index": i,
             "task_id": task["task_id"], "harness": harness}
            for i, task in enumerate(ordered)
            for j, harness in enumerate(harnesses[(i + offset) % len(harnesses):]
                + harnesses[:(i + offset) % len(harnesses)])]

def freeze(baseline):
    load = lambda p: json.loads((baseline / p).read_text(encoding="utf-8"))
    assert load("verification.json")["artifact_checks_passed"]
    tasks, chosen, families = load("tasks.json"), [], set()
    results = load("results.json")
    for task in tasks:
        if task["family"] in families:
            continue
        families.add(task["family"])
        record = next(r for r in results if r["task_id"] == task["task_id"])
        request = load(record["path"] + "/episode/model_calls.json")[0]["request"]
        backend = {**copy.deepcopy(request), "return_token_ids": True}
        chosen.append({"task": task, "request": request, "backend_request": backend,
                       "backend_sha256": digest(backend), "source_path": record["path"]})
    assert len(chosen) == 4
    return {"purpose": "diagnostic_only", "source": str(baseline), "entries": chosen}

def normalize(text):
    try:
        body = json.loads(text)
        if "call" in body:
            return {"tool": body["call"]["tool"], "arguments": body["call"]["arguments"]}
        args = body["action_input"]
        return {"tool": body["action"], "arguments": json.loads(args) if isinstance(args, str) else args}
    except (ValueError, TypeError, KeyError):
        return {"unparsed": text}

def stats(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values), "values": values}

def paired_summary(rounds):
    assert len(rounds) == 3
    keys = [{(r["task_id"], r["harness"]) for r in rs} for rs in rounds]
    assert keys[0] == keys[1] == keys[2] and len(keys[0]) == 32
    assert all(len(rs) == 32 for rs in rounds)
    output = {}
    for task in sorted({t for t, h in keys[0]}):
        variants = {h: [next(r for r in rs if r["task_id"] == task and r["harness"] == h)
                        for rs in rounds] for h in (B0, H1)}
        metrics = ("success", "goal_success", "model_calls", "tool_calls", "invalid_actions",
                   "prompt_tokens", "completion_tokens", "total_tokens", "duration_seconds", "model_latency_seconds", "retrieval_calls", "retrieval_tokens", "retrieval_latency_seconds", "reward")
        output[task] = {h: {m: stats([int(r[m]) if isinstance(r[m], bool) else r[m]
                                      for r in variants[h]]) for m in metrics} for h in (B0, H1)}
        output[task]["H1_minus_B0"] = {m: stats([variants[H1][i][m] - variants[B0][i][m]
                                                for i in range(3)]) for m in metrics}
        output[task]["success_flips"] = {h: len({r["success"] for r in variants[h]}) > 1 for h in (B0, H1)}
    return output
