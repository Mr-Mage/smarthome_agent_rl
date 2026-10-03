"""Download a pinned CPU retrieval model, checking official per-file hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/p1-20261003"
repo = "BAAI/bge-small-en-v1.5"
opener = build_opener(ProxyHandler({"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}))
with opener.open("https://huggingface.co/api/models/" + repo + "?blobs=true", timeout=30) as response:
    manifest = json.load(response)
revision = manifest["sha"]
(WORK / "bge-manifest-pinned.json").write_text(json.dumps(manifest, indent=2))
destination = ROOT.parent / "models/bge-small-en-v1.5-p1"
destination.mkdir(exist_ok=True)
filenames = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"}
entries = [s for s in manifest["siblings"] if s["rfilename"] in filenames]
assert {s["rfilename"] for s in entries} == filenames
def download(entry):
    name = entry["rfilename"]
    target = destination / name
    partial = destination / (name + ".partial")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    expected = entry.get("lfs", {}).get("sha256")
    for attempt in range(20):
        if not target.exists():
            result = subprocess.run(["curl", "-x", "http://127.0.0.1:7897", "-L", "-sS", "--fail",
                "--connect-timeout", "10", "--max-time", "55", "-C", "-", "-o", str(partial), url])
            if result.returncode:
                time.sleep(1)
                continue
            partial.replace(target)
        content = target.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        valid = sha256 == expected if expected else hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == entry["blobId"]
        assert valid, name
        print(json.dumps({"file": name, "verified": True, "bytes": len(content)}), flush=True)
        return {"file": name, "bytes": len(content), "sha256": sha256, "official_hash": expected or entry["blobId"], "verified": True}
    raise RuntimeError("Download failed: " + name)
with ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(download, entries))
verification = {"repository": repo, "revision": revision, "files": records, "verified": True,
                "purpose": "CPU documentation retrieval only; generation model unchanged"}
(destination / "download-verification.json").write_text(json.dumps(verification, indent=2))
(WORK / "retrieval-model-verification.json").write_text(json.dumps(verification, indent=2))
print(json.dumps({"complete": True, "revision": revision, "destination": str(destination)}), flush=True)
