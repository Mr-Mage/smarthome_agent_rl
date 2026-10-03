"""Own the CPU service during immutable index preparation, then release it."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
ROOT = Path(__file__).resolve().parents[1]
work = ROOT / "work/p1-20261003"
model_python = str(work / "embedding-venv/bin/python")
environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "deps/SimuHome")}
import httpx
with (work / "doc-service-prepare.log").open("w") as log:
    proc = subprocess.Popen([model_python, "scripts/serve_doc_embeddings.py", "--model", str(ROOT.parent / "models/bge-small-en-v1.5-p1")],
        cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            for attempt in range(120):
                if proc.poll() is not None:
                    raise RuntimeError("CPU embedding service exited")
                try:
                    if client.get("http://127.0.0.1:20200/health").is_success:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("CPU embedding startup")
        subprocess.run([str(ROOT / ".venv-baseline/bin/python"), "scripts/build_doc_retrieval.py"], cwd=ROOT,
                       env=environment, check=True)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
