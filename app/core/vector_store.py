"""向量数据库操作 — ChromaDB"""
import uuid
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from app.core.embedding import embed_texts

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


async def add_chunks(chunks: list, doc_id: str, doc_name: str) -> int:
    """将文档分块存入向量库，返回存入数量"""
    collection = _get_collection()
    texts = [c["content"] for c in chunks]
    embeddings = await embed_texts(texts)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            **c["metadata"],
            "doc_id": doc_id,
            "doc_name": doc_name,
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(ids)


async def search(query_embedding: List[float], top_k: int = 10) -> List[dict]:
    """向量相似度检索"""
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if not results["ids"][0]:
        return []

    return [
        {
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine距离→相似度
        }
        for i in range(len(results["ids"][0]))
    ]


async def delete_by_doc_id(doc_id: str):
    """删除指定文档的所有向量"""
    collection = _get_collection()
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


async def get_collection_stats() -> dict:
    """获取集合统计"""
    collection = _get_collection()
    return {
        "name": collection.name,
        "count": collection.count(),
    }
