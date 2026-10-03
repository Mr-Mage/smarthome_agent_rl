"""Fail-stop P1 orchestration; an incomplete round is never selectively replaced."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    work = ROOT / "work/p1-20261003"
    work.mkdir(parents=True, exist_ok=True)
    config = "configs/p1-qwen35-9b-retrieval.json"
    suite = "configs/p1-dev-frozen.json"
    baseline = "runs/b0-qwen35-9b-20261003-03"
    initial = "runs/p1-stability-initial-20261003-02"
    restart = "runs/p1-stability-restart-20261003-02"
    destination = "runs/p1-report-20261003"
    chain = "runs/p1-chain-retrieval-20261003-01"
    python = str(ROOT / ".venv-baseline/bin/python")
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "deps/SimuHome")}
    stage = None
    commands = []
    def command(label, argv):
        nonlocal stage
        stage = label
        commands.append({"stage": label, "argv": argv, "started_at": time.time()})
        (work / "orchestration.json").write_text(json.dumps(commands, indent=2))
        print("Starting " + label, flush=True)
        with (work / (label + ".log")).open("w") as log:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        commands[-1]["completed_at"] = time.time()
        (work / "orchestration.json").write_text(json.dumps(commands, indent=2))
    try:
        for label, mode, directory in (("diagnostic-initial-02", "initial", initial), ("diagnostic-restart-01", "restart", restart)):
            if (ROOT / directory / "diagnostic_complete.json").exists():
                complete = json.loads((ROOT / directory / "diagnostic_complete.json").read_text())
                assert complete["requests"] == (48 if mode == "initial" else 24)
                print("Retaining completed frozen-request diagnostic blocks", flush=True)
                continue
            command(label, ["bash", "scripts/run-single-h100.sh", "--config", config, "--suite", suite,
                           "--diagnostic-mode", mode, "--baseline-run", baseline, "--run-dir", directory])
        command("diagnostic-chain-retrieval", ["bash", "scripts/run-single-h100.sh", "--config", config, "--suite", suite,
            "--diagnostic-mode", "chain", "--baseline-run", baseline, "--run-dir", chain])
        command("diagnostic-verification", [python, "scripts/verify_p1_diagnostics.py", initial, restart, baseline,
            destination + "/diagnostics_verification.json", chain])
        rounds = []
        for number in range(1, 4):
            directory = f"runs/p1-paired-round{number}-20261003-02"
            rounds.append(directory)
            command(f"paired-round{number}", ["bash", "scripts/run-single-h100.sh", "--config", config,
                "--suite", suite, "--harnesses", "upstream-react", "structured-schema", "--task-order",
                "reverse" if number == 2 else "forward", "--variant-order-offset", "1" if number == 2 else "0",
                "--run-dir", directory])
            command(f"verify-round{number}", [python, "scripts/verify_harness_suite.py", directory])
        command("paired-report", [python, "scripts/report_p1.py", destination,
            destination + "/diagnostics_verification.json", *rounds])
        (work / "completed.json").write_text(json.dumps({"complete": True, "rounds": rounds,
            "diagnostics": [initial, restart, chain], "report": destination, "training": False}, indent=2))
        print("P1 completed and independently verified", flush=True)
    except Exception as exc:
        (work / "orchestration_failure.json").write_text(json.dumps({"stage": stage,
            "type": type(exc).__name__, "message": str(exc), "incomplete": True}, indent=2))
        raise

if __name__ == "__main__":
    main()
