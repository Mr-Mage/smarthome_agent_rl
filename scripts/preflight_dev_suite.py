"""Prove generated goal reachability with a separate owned CPU simulator."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.adapter import SimuHomeAdapter
from smarthome_agent_rl.reward import check_goals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="configs/dev-expanded-v1.json")
    p.add_argument("--references", default="configs/dev-expanded-reference-plans.json")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--port", type=int, default=20099)
    args = p.parse_args()
    tasks = json.loads((ROOT / args.suite).read_text())
    plans = json.loads((ROOT / args.references).read_text())
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    env = SimuHomeAdapter(f"http://127.0.0.1:{args.port}/api")
    results = []
    with (run / "simulator.log").open("w") as log:
        proc = subprocess.Popen([str(ROOT / ".venv-simuhome/bin/python"), "-m", "uvicorn",
            "src.simulator.api.app:app", "--host", "127.0.0.1", "--port", str(args.port)],
            cwd=ROOT / "deps/SimuHome", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(60):
                if proc.poll() is not None:
                    raise RuntimeError("Simulator exited")
                try:
                    if env.client.get("/__health__").is_success:
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Simulator startup")
            for task in tasks:
                env.reset(task)
                initial_checks = env.checks
                if all(c["satisfied"] for c in initial_checks):
                    raise ValueError(f"Trivial already complete task: {task['task_id']}")
                observations = [env.dispatch(action) for action in plans[task["task_id"]]]
                final = env.request("GET", "/home/state")
                checks = check_goals(final, task["goals"])
                record = {"task_id": task["task_id"], "family": task["family"],
                    "initial_checks": initial_checks, "actions": plans[task["task_id"]],
                    "observations": observations, "checks": checks, "reachable": all(c["satisfied"] for c in checks)}
                results.append(record)
                (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                if not record["reachable"]:
                    raise RuntimeError(f"Unreachable generated task: {task['task_id']}")
            # Also detect configuration/state leakage between repeated resets.
            for task in reversed(tasks):
                env.reset(task)
                expected = next(r["initial_checks"] for r in results if r["task_id"] == task["task_id"])
                if env.checks != expected:
                    raise RuntimeError(f"Reset does not restore goals: {task['task_id']}")
            summary = {"tasks": len(results), "reachable": sum(r["reachable"] for r in results),
                "reset_restores_initial_goal_states": True, "agent_used": False, "benchmark_used": False}
            (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)
        finally:
            env.close()
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)


if __name__ == "__main__":
    main()
