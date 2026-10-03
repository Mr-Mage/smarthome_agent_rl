"""Lightning diagnostic worker: generate one frozen reply, execute no tools."""
import json
import os
from pathlib import Path
import time
import httpx

class FrozenRequestAgent:
    def run(self):
        with httpx.Client(trust_env=False, timeout=120) as client:
            started = time.monotonic()
            response = client.post(os.environ["AGL_OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["AGL_KEY"]},
                json=json.loads(os.environ["P1_REQUEST"]))
            elapsed = time.monotonic() - started
            response.raise_for_status()
            Path(os.environ["P1_OUTPUT"]).write_text(json.dumps({"response": response.json(),
                "duration_seconds": elapsed}, ensure_ascii=False, indent=2), encoding="utf-8")
