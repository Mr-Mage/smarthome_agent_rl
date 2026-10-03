"""Offline effective embedding configuration audit without credentials or network."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "deps/SimuHome")]
from langchain_openai import OpenAIEmbeddings
import faiss
config = json.loads((ROOT / "configs/p1-qwen35-9b.json").read_text())
os.environ.update(OPENAI_API_KEY="local-unused", OPENAI_BASE_URL=config["model_endpoint"],
                  OPENAI_API_BASE=config["model_endpoint"])
embedding = OpenAIEmbeddings()
index = ROOT / "deps/SimuHome/data/vector_db/matter_index/index.faiss"
tokenizer = ROOT / "work/p1-20261003/cl100k_base.tiktoken"
expected = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
assert hashlib.sha256(tokenizer.read_bytes()).hexdigest() == expected
cache = ROOT / "work/p1-20261003/tiktoken-cache"
cache.mkdir(exist_ok=True)
url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
(cache / hashlib.sha1(url.encode()).hexdigest()).write_bytes(tokenizer.read_bytes())
os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
import tiktoken
assert tiktoken.get_encoding("cl100k_base").encode("test")
report = {"tokenizer_cached_and_hash_verified": True, "tokenizer_sha256": expected,
    "cache_directory": str(cache), "requires_cache_environment": "TIKTOKEN_CACHE_DIR",
    "embedding_model": embedding.model, "effective_embedding_base_url": str(embedding.client._client.base_url),
    "key_source": "worker hardcoded local-unused placeholder", "faiss_dimension": faiss.read_index(str(index)).d,
    "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
    "generation_backend": "Qwen3.5-9B vLLM language-model-only; no compatible embedding service configured",
    "configuration_files_with_credentials_found": False, "network_requests_sent": 0,
    "required_for_resume": "Existing FAISS-compatible embedding service configuration; do not change retrieval results or substitute fabricated observations"}
(ROOT / "work/p1-20261003/doc_dependency_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
