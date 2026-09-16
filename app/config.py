from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Enterprise RAG Agent"
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    upload_dir: Path = Path("./data/uploads")
    max_upload_size_mb: int = 20

    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection: str = "enterprise_docs_v2"
    chunk_size: int = 800
    chunk_overlap: int = 150

    embedding_backend: str = "hash"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 384

    rerank_backend: str = "lexical"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    min_relevance_score: float = 0.05

    # 上下文解析只喂最近 N 轮，避免历史无限增长
    context_history_turns: int = 5

    llm_api_key: str | None = None
    llm_base_url: str | None = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    app_api_key: str | None = None
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def prepare_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite"):
            Path(self.database_url.removeprefix("sqlite+aiosqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
