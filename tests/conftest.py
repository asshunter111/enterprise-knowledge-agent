import os
import tempfile
from pathlib import Path

test_root = Path(tempfile.gettempdir()) / f"enterprise-rag-agent-tests-{os.getpid()}"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{test_root / 'test.db'}"
os.environ["UPLOAD_DIR"] = str(test_root / "uploads")
os.environ["CHROMA_PERSIST_DIR"] = str(test_root / "chroma")
os.environ["CHROMA_COLLECTION"] = "test_collection"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["RERANK_BACKEND"] = "lexical"
os.environ["LLM_API_KEY"] = ""
