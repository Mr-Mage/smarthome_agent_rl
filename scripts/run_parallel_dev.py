"""Independent GPU/simulator/Gateway shards; no shared reset or controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def validate_plan(plan):
    occupied_gpus, occupied_ports, task_ids = set(), set(), set()
    for shard in plan["shards"]:
        config = json.loads((ROOT / shard["config"]).read_text())
        gpu_list = shard["gpu"].split(",")
        gpus = set(gpu_list)
        if any(not g.isdigit() for g in gpu_list) or len(gpus) != len(gpu_list):
            raise ValueError("GPU IDs must be distinct nonnegative integer indices")
        if len(gpus) != int(config.get("tensor_parallel_size", 1)):
            raise ValueError("Shard GPU allocation must match tensor_parallel_size")
        if not gpus or gpus & occupied_gpus:
            raise ValueError("Parallel shards must have disjoint GPUs")
        occupied_gpus |= gpus
        ports = set()
        for endpoint in ["simulator_url", "model_endpoint", "gateway_url"]:
            parsed = urlparse(config[endpoint])
            if parsed.hostname != "127.0.0.1" or parsed.port is None:
                raise ValueError("Parallel plan requires explicit local ports")
            if parsed.port in ports or parsed.port in occupied_ports:
                raise ValueError("Parallel shards must have disjoint service ports")
            ports.add(parsed.port)
        occupied_ports |= ports
        ids = [t["task_id"] for t in json.loads((ROOT / shard["suite"]).read_text())]
        if len(set(ids)) != len(ids) or set(ids) & task_ids:
            raise ValueError("Task shards must be disjoint")
        task_ids.update(ids)
    return {"gpus": sorted(occupied_gpus), "ports": sorted(occupied_ports), "tasks": len(task_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="configs/parallel-dev-v1.json")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    inventory = validate_plan(plan)
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (run / "isolation.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    owned, handles, commands = [], [], []
    try:
        for i, shard in enumerate(plan["shards"]):
            output = run / f"shard{i}"
            command = [sys.executable, str(ROOT / "scripts/run_harness_suite.py"), "--suite", shard["suite"],
                "--config", shard["config"], "--gpu", shard["gpu"], "--run-dir", str(output),
                "--harnesses", *plan["harnesses"]]
            log = (run / f"shard{i}.log").open("w")
            handles.append(log)
            commands.append(command)
            owned.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True))
        (run / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
        print(json.dumps({"state": "started", **inventory, "pids": [p.pid for p in owned]}), flush=True)
        while any(p.poll() is None for p in owned):
            time.sleep(2)
        status = [p.returncode for p in owned]
        (run / "exit_codes.json").write_text(json.dumps(status), encoding="utf-8")
        if any(status):
            raise RuntimeError(f"Shard failed; retain all artifacts: {status}")
        results = []
        for i in range(len(owned)):
            shard_dir = run / f"shard{i}"
            subprocess.run([sys.executable, str(ROOT / "scripts/verify_harness_suite.py"), str(shard_dir)], check=True)
            data = json.loads((shard_dir / "comparison.json").read_text())
            for record in data["results"]:
                results.append({**record, "path": f"shard{i}/{record['path']}"})
        aggregation = {}
        families = {}
        for record in results:
            s = record["summary"]
            a = aggregation.setdefault(record["harness"], {"episodes": 0, "successes": 0,
                "model_calls": 0, "invalid_actions": 0, "tokens": 0, "reward_sum": 0.0})
            a["episodes"] += 1
            a["successes"] += int(s["success"])
            for name, key in [("model_calls", "model_calls"), ("invalid_actions", "invalid_actions"), ("tokens", "total_tokens"), ("reward_sum", "reward")]:
                a[name] += s[key]
            task = json.loads((run / record["path"] / "episode/task.json").read_text())
            f = families.setdefault(task["family"], {}).setdefault(record["harness"], {"episodes": 0, "successes": 0})
            f["episodes"] += 1
            f["successes"] += int(s["success"])
        for a in aggregation.values():
            a["success_rate"] = a["successes"] / a["episodes"]
        summary = {"aggregation": aggregation, "families": families, "results": results,
            "isolation": inventory, "official_benchmark": False, "training": False,
            "note": "One execution per dev fixture/profile; GPU/shard is not a causal variable."}
        (run / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"aggregation": aggregation, "families": families}), flush=True)
    finally:
        for p in owned:
            if p.poll() is None:
                # Signal wrapper first so it can stop only the services it owns.
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=15)
        for log in handles:
            log.close()


if __name__ == "__main__":
    main()
