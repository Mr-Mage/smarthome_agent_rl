from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "work/p1-20261003/doc-preparation.log").open("w") as log:
    process = subprocess.Popen([str(ROOT / ".venv-baseline/bin/python"), "-u", "scripts/prepare_doc_backend.py"], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(process.pid)
