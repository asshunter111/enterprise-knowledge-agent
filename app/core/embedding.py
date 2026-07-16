"""Embedding — 本地 BGE 模型，无需 API Key"""
from typing import List

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# BGE 中文小模型，本地运行，首次自动下载 (~400MB)
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _model


async def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    return model.embed_documents(texts)


async def embed_query(query: str) -> List[float]:
    model = _get_model()
    return model.embed_query(query)
