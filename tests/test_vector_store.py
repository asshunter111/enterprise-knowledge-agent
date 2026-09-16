import pytest

from app.config import Settings
from app.core.embedding import EmbeddingService
from app.core.vector_store import VectorStore


@pytest.mark.asyncio
async def test_vector_store_upsert_and_count():
    settings = Settings(
        chroma_collection="test_vector_store_upsert",
        embedding_backend="hash",
    )

    vector_store = VectorStore(settings)
    embedding_service = EmbeddingService(settings)

    chunks = [
        {
            "content": "兰州是甘肃省的省会城市。",
            "metadata": {
                "chunk_index": 0,
                "chunk_total": 1,
            },
        }
    ]

    embeddings = await embedding_service.embed_documents(
        [chunk["content"] for chunk in chunks]
    )

    count = await vector_store.upsert(
        chunks=chunks,
        embeddings=embeddings,
        document_id="test-doc-1",
        document_name="test.md",
    )

    assert count == 1
    assert await vector_store.count() == 1

    
@pytest.mark.asyncio
async def test_vector_store_search_returns_relevant_chunk():
    settings = Settings(
        chroma_collection="test_vector_store_search",
        embedding_backend="hash",
    )

    vector_store = VectorStore(settings)
    embedding_service = EmbeddingService(settings)

    chunks = [
        {
            "content": "兰州是甘肃省的省会城市。",
            "metadata": {
                "chunk_index": 0,
                "chunk_total": 1,
            },
        },
        {
            "content": "企业级RAG系统通常包括文档解析、文本切分和向量检索。",
            "metadata": {
                "chunk_index": 1,
                "chunk_total": 2,
            },
        },
    ]

    embeddings = await embedding_service.embed_documents(
        [chunk["content"] for chunk in chunks]
    )

    await vector_store.upsert(
        chunks=chunks,
        embeddings=embeddings,
        document_id="test-doc-2",
        document_name="test.md",
    )

    query = "兰州是哪个省的省会？"

    query_embedding = await embedding_service.embed_query(query)

    results = await vector_store.search(
        embedding=query_embedding,
        top_k=2,
    )

    assert len(results) == 2
    assert all("id" in item for item in results)
    assert all("content" in item for item in results)
    assert all("metadata" in item for item in results)
    assert all("score" in item for item in results)

    assert results[0]["content"] == "兰州是甘肃省的省会城市。"    