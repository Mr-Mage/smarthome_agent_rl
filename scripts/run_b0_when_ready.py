"""Finite job: verified 9B download + idle GPU -> own B0, verification and report."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-pid", type=int, required=True)
    parser.add_argument("--config", default="configs/b0-qwen35-9b.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--timeout-hours", type=float, default=24)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    destination = (ROOT / config["model_path"]).resolve()
    state_path = ROOT / "logs/b0-queued-state.json"
    started, previous = time.monotonic(), None
    def save(state, **extra):
        nonlocal previous
        value = {"state": state, "config": args.config, "run_dir": args.run_dir,
            "download_pid": args.download_pid, "elapsed_seconds": time.monotonic() - started,
            "training": False, "official_reproduction": False, **extra}
        state_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        if state != previous:
            print(json.dumps(value), flush=True)
            previous = state
    try:
        while time.monotonic() - started < args.timeout_hours * 3600:
            marker_path = destination / "download-verification.json"
            marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
            verified = marker.get("verified") is True and marker.get("repository") == config["model_repo"] and marker.get("revision") == config["model_revision"]
            if not verified:
                process = Path(f"/proc/{args.download_pid}/cmdline")
                command = process.read_bytes() if process.exists() else b""
                if b"prepare_reproduction_model.py" not in command or destination.name.encode() not in command:
                    raise RuntimeError("Model downloader stopped before full verification")
                save("waiting_for_verified_model")
            elif subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                save("waiting_for_idle_gpu")
            else:
                commands = [["bash", "scripts/run-single-h100.sh", "--config", args.config, "--run-dir", args.run_dir],
                    [sys.executable, "scripts/verify_harness_suite.py", args.run_dir],
                    [sys.executable, "scripts/summarize_baseline.py", args.run_dir]]
                save("running_own_b0", commands=commands)
                for command in commands:
                    result = subprocess.run(command, cwd=ROOT)
                    if result.returncode:
                        save("failed", command=command, returncode=result.returncode)
                        return result.returncode
                save("completed_and_verified")
                return 0
            time.sleep(30)
        save("queue_timeout")
        return 1
    except Exception as exc:
        save("queue_error", error=type(exc).__name__, detail=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
