"""Repair shared retrieval, verify it, then restart the entire paired experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
work = ROOT / "work/p1-20261003"
python = str(ROOT / ".venv-baseline/bin/python")
def run(label, argv):
    print("Starting " + label, flush=True)
    with (work / (label + ".log")).open("w") as log:
        subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
verification = ROOT.parent / "models/bge-small-en-v1.5-p1/download-verification.json"
if not verification.exists():
    run("bge-download-resumed", [python, "-u", "scripts/download_p1_embedding.py"])
index = work / "doc-index-bge"
if not (index / "verification.json").exists():
    run("doc-backend-build", [python, "-u", "scripts/prepare_doc_backend.py"])
manifest = json.loads((index / "verification.json").read_text())
assert manifest["verified"]
config = json.loads((ROOT / "configs/p1-qwen35-9b.json").read_text())
config["retrieval"] = {"endpoint": "http://127.0.0.1:20200", "model_path": str(verification.parent),
    "python": str(work / "embedding-venv/bin/python"),
    "model_revision": manifest["model_revision"], "index_path": str(index), "device": "cpu",
    "repair_reason": "User requested real retrieval backend before continuing original P1 protocol; original Ada FAISS assets remain untouched",
    "indexing_config": {"source": "upstream_original_docs", "chunk_size": 1000, "chunk_overlap": 200}}
(ROOT / "configs/p1-qwen35-9b-retrieval.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
for filename in ("orchestrator.log", "orchestration.json", "orchestration_failure.json"):
    source = work / filename
    if source.exists():
        import shutil
        target = work / ("before-retrieval-" + filename)
        if not target.exists():
            shutil.copy2(source, target)
run("orchestrator-retrieval", [python, "-u", "scripts/run_p1.py"])
print("Retrieval repair and all paired rounds completed", flush=True)
