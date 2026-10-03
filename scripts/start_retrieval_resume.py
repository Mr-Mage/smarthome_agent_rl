from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "work/p1-20261003/retrieval-resume.log").open("w") as log:
    process = subprocess.Popen([str(ROOT / ".venv-baseline/bin/python"), "-u", "scripts/resume_p1_with_retrieval.py"], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (ROOT / "work/p1-20261003/retrieval-resume.pid").write_text(str(process.pid))
    print(process.pid)
