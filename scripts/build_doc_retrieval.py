"""Rebuild an isolated matching FAISS index from the pristine upstream docs."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "deps/SimuHome")]
from src.agents.tools import load_docs, tool_get_cluster_doc, ToolConfig, set_tool_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from smarthome_agent_rl.retrieval import LocalDocEmbeddings

destination = ROOT / "work/p1-20261003/doc-index-bge"
destination.mkdir(exist_ok=False)
documents = load_docs()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
embedding = LocalDocEmbeddings("http://127.0.0.1:20200", ROOT / "work/p1-20261003/index_embedding_cost.json")
database = FAISS.from_documents(chunks, embedding)
database.save_local(str(destination))
revision = json.loads((ROOT.parent / "models/bge-small-en-v1.5-p1/download-verification.json").read_text())["revision"]
set_tool_config(ToolConfig(db=database))
checks = []
for query, keywords in (("LevelControl MoveToLevel command Level TransitionTime", ["MoveToLevel", "Level"]),
                        ("FanControl FanMode attribute values", ["FanMode"]),
                        ("OnOff cluster On Off command", ["OnOff"])):
    response = tool_get_cluster_doc({"query": query, "top_k": 3})
    assert response["status"]["code"] == 200
    text = response["data"]["text"]
    assert all(keyword in text for keyword in keywords), query
    checks.append({"query": query, "response": response})
manifest = {"verified": True, "embedding_model": "BAAI/bge-small-en-v1.5", "model_revision": revision,
    "device": "cpu", "dimension": 384, "documents": len(documents), "chunks": len(chunks),
    "chunk_size": 1000, "chunk_overlap": 200, "original_docs_loader": "src.agents.tools.load_docs",
    "original_index_unchanged": True, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "deps/SimuHome/docs/clusters").glob("*.md")},
    "index_sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in ("index.faiss", "index.pkl")},
    "smoke_checks": checks, "indexing_cost": {"embedding_tokens": sum(r["input_tokens"] for r in embedding.records if r["purpose"] == "documentation_indexing"),
        "duration_seconds": sum(r["duration_seconds"] for r in embedding.records if r["purpose"] == "documentation_indexing")},
    "smoke_cost": {"embedding_tokens": sum(r["input_tokens"] for r in embedding.records if r["purpose"] == "documentation_query"),
        "duration_seconds": sum(r["duration_seconds"] for r in embedding.records if r["purpose"] == "documentation_query")}}
(destination / "verification.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"verified": True, "documents": len(documents), "chunks": len(chunks), "revision": revision}), flush=True)
