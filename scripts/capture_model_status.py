"""Capture pinned model file progress without claiming partial files are verified."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="configs/model-qwen3.5-9b.json")
parser.add_argument("--destination", default="../models/Qwen3.5-9B")
parser.add_argument("--output", default="logs/b0-model-status.json")
args = parser.parse_args()
manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8-sig"))
destination = (ROOT / args.destination).resolve()
files = []
for entry in manifest["files"]:
    target, partial = destination / entry["path"], destination / (entry["path"] + ".partial")
    files.append({"path": entry["path"], "expected_bytes": entry["bytes"],
        "complete_bytes": target.stat().st_size if target.exists() else 0,
        "partial_bytes": partial.stat().st_size if partial.exists() else 0})
marker_path = destination / "download-verification.json"
marker = json.loads(marker_path.read_text()) if marker_path.exists() else None
result = {"captured_utc": datetime.now(timezone.utc).isoformat(),
    "repository": manifest["repository"], "revision": manifest["revision"], "destination": str(destination),
    "expected_bytes": sum(e["bytes"] for e in manifest["files"]),
    "stored_bytes": sum(max(e["complete_bytes"], e["partial_bytes"]) for e in files),
    "complete_files": sum(e["complete_bytes"] == e["expected_bytes"] for e in files),
    "verification": marker, "files": files, "partial_bytes_not_verified": True}
target = ROOT / args.output
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("repository", "stored_bytes", "expected_bytes", "complete_files", "verification")}))
