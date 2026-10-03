"""Finite queue: verified download + completed dev run + idle GPU -> original CLI.

No training, judge, source changes, or automatic model/precision substitution.
"""
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
    parser.add_argument("--dev-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/reproduction-qwen3-32b-single-cu128.json")
    parser.add_argument("--state-file", default="logs/qwen3-queued-state.json")
    parser.add_argument("--timeout-hours", type=float, default=24)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    model = ROOT.parent / config["model_path"]
    dev = Path(args.dev_run).resolve()
    state_file = ROOT / args.state_file
    started = time.monotonic()
    previous = None
    def record(state, **extra):
        nonlocal previous
        value = {"state": state, "config": args.config, "dev_run": str(dev),
            "download_pid": args.download_pid, "elapsed_seconds": time.monotonic() - started,
            "training": False, **extra}
        state_file.write_text(json.dumps(value, indent=2), encoding="utf-8")
        if state != previous:
            print(json.dumps(value), flush=True)
            previous = state
    try:
        while time.monotonic() - started < args.timeout_hours * 3600:
            marker_path = model / "download-verification.json"
            marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
            ready = marker.get("verified") is True and marker.get("repository") == config["model_repo"] and marker.get("revision") == config["model_revision"]
            if not ready:
                cmdline_path = Path(f"/proc/{args.download_pid}/cmdline")
                cmdline = cmdline_path.read_bytes() if cmdline_path.exists() else b""
                if b"prepare_reproduction_model.py" not in cmdline:
                    record("download_stopped_before_verification", marker=marker)
                    raise RuntimeError("Download is no longer running; resume it before queueing again")
                record("waiting_for_download")
            elif not (dev / "comparison.json").exists() and not (dev / "failure.json").exists():
                record("waiting_for_dev")
            else:
                active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
                if active:
                    record("waiting_for_idle_gpu")
                else:
                    command = ["bash", "scripts/reproduce-h100.sh", "--config", args.config, "--run-dir", args.run_dir]
                    record("running_original_cli", command=command)
                    result = subprocess.run(command, cwd=ROOT)
                    record("completed" if result.returncode == 0 else "failed", returncode=result.returncode, run_dir=args.run_dir)
                    return result.returncode
            time.sleep(30)
        record("queue_timeout")
        return 1
    except Exception as exc:
        record("queue_error", error=type(exc).__name__, detail=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
