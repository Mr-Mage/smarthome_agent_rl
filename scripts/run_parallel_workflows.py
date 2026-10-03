"""Run pristine CLI reproduction alongside CPU preflight + Lightning dev shards."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reproduction-config", default="configs/reproduction-smoke.json")
    parser.add_argument("--dev-plan", default="configs/parallel-dev-v1.json")
    parser.add_argument("--single-gpu", action="store_true", help="CPU preflight in parallel; dev GPU work waits for official CLI exit")
    args = parser.parse_args()
    # Validate reservations across the two experiment lines before starting either.
    from run_parallel_dev import validate_plan
    plan = json.loads((ROOT / args.dev_plan).read_text())
    dev = validate_plan(plan)
    repro = json.loads((ROOT / args.reproduction_config).read_text())
    if len(repro["gpu"].split(",")) != int(repro["tensor_parallel_size"]):
        raise ValueError("Reproduction GPU allocation must match tensor_parallel_size")
    preflight_config = plan["preflight"]
    expected_ids = {t["task_id"] for t in json.loads((ROOT / preflight_config["suite"]).read_text())}
    shard_ids = {t["task_id"] for shard in plan["shards"]
        for t in json.loads((ROOT / shard["suite"]).read_text())}
    if expected_ids != shard_ids:
        raise ValueError("Preflight suite must cover exactly the development shards")
    if set(repro["gpu"].split(",")) & set(dev["gpus"]) and not args.single_gpu:
        raise ValueError("Reproduction and dev GPU assignments overlap")
    if args.single_gpu and (len(plan["shards"]) != 1 or repro["gpu"] != plan["shards"][0]["gpu"] or "," in repro["gpu"]):
        raise ValueError("Single-GPU mode requires one dev shard and the same single GPU for both lines")
    if repro["model_port"] in dev["ports"] or 20099 in dev["ports"] or repro["model_port"] == 20099:
        raise ValueError("Reproduction, dev, or preflight ports overlap")
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    commands, handles, processes = [], [], {}
    states = {"official": "pending", "preflight": "pending", "dev": "pending"}
    def save():
        (run / "workflow_status.json").write_text(json.dumps({"states": states, "commands": commands,
            "training": False, "dev_uses_benchmark": False,
            "gpu_work_sequential": args.single_gpu}, indent=2), encoding="utf-8")
    def launch(name, command):
        handle = (run / f"{name}.log").open("w")
        handles.append(handle)
        proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = proc
        commands.append({"name": name, "argv": command, "pid": proc.pid})
        states[name] = "running"
        save()
        return proc
    try:
        launch("official", [sys.executable, str(ROOT / "scripts/reproduce_official_cli.py"),
            "--config", args.reproduction_config, "--run-dir", str(run / "official")])
        preflight = launch("preflight", [sys.executable, str(ROOT / "scripts/preflight_dev_suite.py"),
            "--suite", preflight_config["suite"], "--references", preflight_config["references"],
            "--run-dir", str(run / "preflight")])
        while preflight.poll() is None:
            time.sleep(1)
        states["preflight"] = "succeeded" if preflight.returncode == 0 else "failed"
        save()
        if preflight.returncode == 0:
            if args.single_gpu:
                states["dev"] = "waiting_for_gpu"
                save()
                while processes["official"].poll() is None:
                    time.sleep(2)
                states["official"] = "succeeded" if processes["official"].returncode == 0 else "failed"
                save()
            launch("dev", [sys.executable, str(ROOT / "scripts/run_parallel_dev.py"),
                "--plan", args.dev_plan, "--run-dir", str(run / "dev")])
        else:
            states["dev"] = "blocked_by_preflight"
            save()
        while any(p.poll() is None for p in processes.values()):
            for name, proc in processes.items():
                if proc.poll() is not None and states[name] == "running":
                    states[name] = "succeeded" if proc.returncode == 0 else "failed"
                    save()
            time.sleep(2)
        for name, proc in processes.items():
            states[name] = "succeeded" if proc.returncode == 0 else "failed"
        save()
        print(json.dumps(states), flush=True)
        if states["official"] != "succeeded" or states["dev"] != "succeeded":
            raise SystemExit(1)
    finally:
        for name, proc in processes.items():
            if proc.poll() is None:
                # Let each runner's finally clean its own process groups.
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=60)
                states[name] = "interrupted"
        save()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
