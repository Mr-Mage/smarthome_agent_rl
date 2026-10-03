import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from smarthome_agent_rl.retrieval import LocalDocEmbeddings, load_retrieval
from src.agents.tools import ToolConfig, set_tool_config, tool_get_cluster_doc

class RetrievalTests(unittest.TestCase):
    def test_original_tool_uses_injected_database_and_correct_query_mode(self):
        modes = []
        def encode(self, texts, query):
            modes.append(query)
            return [[1.0 if "MoveToLevel" in t else 0.0, 0.0 if "MoveToLevel" in t else 1.0] + [0.0] * 382 for t in texts]
        with patch.object(LocalDocEmbeddings, "encode", encode):
            embeddings = LocalDocEmbeddings("http://unused")
            database = FAISS.from_documents([Document(page_content="MoveToLevel requires Level and TransitionTime"),
                                            Document(page_content="FanMode is the fan attribute")], embeddings)
            set_tool_config(ToolConfig(db=database))
            try:
                with patch("src.agents.tools._load_shared_db", side_effect=AssertionError("Original remote backend must not load")):
                    result = tool_get_cluster_doc({"query": "MoveToLevel", "top_k": 1})
                self.assertEqual(result["status"]["code"], 200)
                self.assertIn("TransitionTime", result["data"]["text"])
                self.assertEqual(modes, [False, True])
            finally:
                set_tool_config(ToolConfig())

    def test_corrupt_index_fails_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "index.faiss").write_bytes(b"corrupted")
            (directory / "verification.json").write_text(json.dumps({"verified": True,
                "model_revision": "fixed", "index_sha256": {"index.faiss": hashlib.sha256(b"expected").hexdigest()}}))
            with self.assertRaises(AssertionError):
                load_retrieval({"endpoint": "http://unused", "index_path": temp, "model_revision": "fixed"})
