"""Read-only documentation tool dependency probe; never print credentials."""
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "deps/SimuHome")]
from dotenv import dotenv_values
summary = []
for p in (ROOT / ".env", ROOT.parent / ".env", ROOT / "deps/SimuHome/.env"):
    if p.exists():
        values = dotenv_values(p)
        summary.append({"path": str(p), "keys": sorted(values),
            "credential_configured": any(v and v not in {"local-unused", "your_api_key", "your-openai-api-key"} for k, v in values.items() if k in {"OPENAI_API_KEY", "EMBEDDING_API_KEY"}),
            "endpoint_hosts": {k: urlparse(v).hostname for k, v in values.items() if v and k in {"OPENAI_BASE_URL", "OPENAI_API_BASE", "EMBEDDING_BASE_URL"}}})
print(json.dumps(summary, ensure_ascii=False))
if "--run" not in sys.argv:
    raise SystemExit(0)
from src.agents.tools import tool_get_cluster_doc
reply = tool_get_cluster_doc({"query": "Matter LevelControl cluster MoveToLevel command arguments", "top_k": 3})
error = reply.get("error")
# Do not print exception strings that could contain authenticated endpoint URLs.
print(json.dumps({"status": reply.get("status"), "data_chars": len(str(reply.get("data"))),
    "error_type": error.get("type") if isinstance(error, dict) else error,
    "error_class": "tokenizer_download" if "cl100k_base.tiktoken" in str(error) else
                   "authentication" if "401" in str(error) else
                   "embedding_endpoint_not_found" if "404" in str(error) else "other" if error else None}))
