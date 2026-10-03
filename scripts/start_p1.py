"""Detach only the owned P1 orchestration, preserving original SSH sessions."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
work = ROOT / "work/p1-20261003"
with (work / "orchestrator.log").open("w") as log:
    process = subprocess.Popen([str(ROOT / ".venv-baseline/bin/python"), "-u", "scripts/run_p1.py"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (work / "orchestrator.pid").write_text(str(process.pid))
    print(process.pid)
