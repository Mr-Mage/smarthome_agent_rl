"""Reuse an installed compatible hub package in an isolated retrieval venv."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
installed = Path("/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/anaconda3/envs/agent-lightning/lib/python3.12/site-packages")
destination = ROOT / "work/p1-20261003/embedding-venv/lib/python3.12/site-packages"
paths = [installed / "huggingface_hub", installed / "huggingface_hub-0.36.2.dist-info"]
assert all(p.is_dir() for p in paths)
for source in paths:
    shutil.copytree(source, destination / source.name, dirs_exist_ok=True)
audit = {"source_environment_unchanged": True, "generation_environment_unchanged": True,
    "isolated_dependency": "huggingface-hub==0.36.2", "source": str(installed), "destination": str(destination),
    "module_sha256": {str(p.relative_to(installed)): hashlib.sha256(p.read_bytes()).hexdigest()
        for source in paths for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts}}
(ROOT / "work/p1-20261003/embedding-venv-isolation.json").write_text(json.dumps(audit, indent=2))
print(json.dumps({"isolated_dependency": "huggingface-hub==0.36.2", "base_environments_unchanged": True}))
