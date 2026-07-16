"""Embedding 向量化服务"""
from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL


# 全局单例
_embedding_model: OpenAIEmbeddings | None = None


def get_embedding_model() -> OpenAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OpenAIEmbeddings(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
            model=EMBEDDING_MODEL,
        )
    return _embedding_model


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """对文本列表进行向量化"""
    model = get_embedding_model()
    return await model.aembed_documents(texts)


async def embed_query(query: str) -> List[float]:
    """对查询文本进行向量化"""
    model = get_embedding_model()
    return await model.aembed_query(query)
