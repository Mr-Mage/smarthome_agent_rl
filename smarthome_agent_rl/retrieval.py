"""External retrieval dependency repair; pristine tool performs original search."""
import json
import hashlib
import time
from pathlib import Path
import httpx
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

class LocalDocEmbeddings(Embeddings):
    def __init__(self, endpoint, audit_path=None):
        self.endpoint, self.audit_path = endpoint, audit_path
        self.records = []

    def encode(self, texts, query):
        started = time.monotonic()
        with httpx.Client(trust_env=False, timeout=300) as client:
            response = client.post(self.endpoint + "/embed", json={"texts": texts, "query": query})
            response.raise_for_status()
            result = response.json()
        assert result["dimension"] == 384 and len(result["vectors"]) == len(texts)
        assert all(len(v) == 384 for v in result["vectors"])
        self.records.append({"purpose": "documentation_query" if query else "documentation_indexing",
            "texts": texts, "input_tokens": result["input_tokens"], "duration_seconds": time.monotonic() - started,
            "server_duration_seconds": result["duration_seconds"], "dimension": 384, "device": result["device"]})
        if self.audit_path:
            Path(self.audit_path).write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
        return result["vectors"]

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), 128):
            vectors.extend(self.encode(texts[start:start + 128], False))
        return vectors

    def embed_query(self, text):
        return self.encode([text], True)[0]

def load_retrieval(settings, audit_path=None):
    embeddings = LocalDocEmbeddings(settings["endpoint"], audit_path)
    directory = Path(settings["index_path"])
    manifest = json.loads((directory / "verification.json").read_text())
    assert manifest["verified"] and manifest["model_revision"] == settings["model_revision"]
    for filename, expected in manifest["index_sha256"].items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
    database = FAISS.load_local(str(directory), embeddings, allow_dangerous_deserialization=True)
    assert database.index.d == 384
    return database
