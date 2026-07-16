"""企业知识库智能问答 Agent — 应用配置"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── LLM 配置 ──
LLM_API_KEY = os.getenv("LLM_API_KEY", "your-deepseek-api-key")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ── Embedding 配置 ──
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── 向量数据库 ──
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_data"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "enterprise_docs")

# ── 数据库（MySQL / SQLite） ──
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///" + str(BASE_DIR / "app.db")
)
# MySQL 示例: "mysql+asyncmy://user:pass@localhost:3306/rag_db"

# ── 检索配置 ──
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# ── 文件上传 ──
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

# ── 服务 ──
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
