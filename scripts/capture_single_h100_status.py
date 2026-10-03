"""Capture hardware and finite download queue state without exposing credentials."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "configs/reproduction-model-qwen3-32b.json").read_text())
destination = ROOT.parent / "models/Qwen3-32B"
files = []
for entry in manifest["files"]:
    complete, partial = destination / entry["path"], destination / (entry["path"] + ".partial")
    complete_bytes = complete.stat().st_size if complete.exists() else 0
    partial_bytes = partial.stat().st_size if partial.exists() else 0
    files.append({"path": entry["path"], "expected_bytes": entry["bytes"],
        "complete_bytes": complete_bytes, "partial_bytes": partial_bytes})

def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None

processes = {}
for label, pid in {"download": 1002, "queued_reproduction": 3030, "dev_runner": 2760}.items():
    path = Path(f"/proc/{pid}/cmdline")
    command = path.read_bytes().decode().replace("\0", " ").strip() if path.exists() else ""
    processes[label] = {"pid": pid, "command": command, "alive": bool(command)}
result = {
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv"], text=True),
    "active_gpu_pids": subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(),
    "upstream_status": subprocess.check_output(["git", "-C", str(ROOT / "deps/SimuHome"), "status", "--porcelain"], text=True),
    "download": {"repository": manifest["repository"], "revision": manifest["revision"],
        "expected_bytes": sum(e["bytes"] for e in manifest["files"]),
        "stored_bytes": sum(max(e["complete_bytes"], e["partial_bytes"]) for e in files),
        "complete_files": sum(e["complete_bytes"] == e["expected_bytes"] for e in files),
        "verification": read_json(destination / "download-verification.json"), "files": files},
    "queue": read_json(ROOT / "logs/qwen3-queued-state.json"),
    "processes": processes, "training": False,
}
target = ROOT / "logs/single-h100-status.json"
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"path": str(target), "stored_bytes": result["download"]["stored_bytes"],
    "expected_bytes": result["download"]["expected_bytes"], "complete_files": result["download"]["complete_files"],
    "verified": bool(result["download"]["verification"]), "gpu_pids": result["active_gpu_pids"],
    "queue": result["queue"]["state"]}))
