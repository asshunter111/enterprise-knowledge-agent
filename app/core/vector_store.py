import asyncio
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings, get_settings


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            self.settings.prepare_directories()
            self._client = chromadb.PersistentClient(
                path=str(self.settings.chroma_persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def upsert(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        document_id: str,
        document_name: str,
    ) -> int:
        return await asyncio.to_thread(
            self._upsert_sync, chunks, embeddings, document_id, document_name
        )

    def _upsert_sync(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        document_id: str,
        document_name: str,
    ) -> int:
        if not chunks:
            return 0
        collection = self._get_collection()
        collection.upsert(
            ids=[f"{document_id}:{chunk['metadata']['chunk_index']}" for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk["content"] for chunk in chunks],
            metadatas=[
                {
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "chunk_total": chunk["metadata"]["chunk_total"],
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    async def search(self, embedding: list[float], top_k: int) -> list[dict]:
        return await asyncio.to_thread(self._search_sync, embedding, top_k)

    def _search_sync(self, embedding: list[float], top_k: int) -> list[dict]:
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": result["ids"][0][index],
                "content": result["documents"][0][index],
                "metadata": result["metadatas"][0][index],
                "score": max(0.0, 1.0 - result["distances"][0][index]),
            }
            for index in range(len(result["ids"][0]))
        ]

    async def delete_document(self, document_id: str) -> None:
        await asyncio.to_thread(self._delete_document_sync, document_id)

    def _delete_document_sync(self, document_id: str) -> None:
        self._get_collection().delete(where={"document_id": document_id})

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)

    def _count_sync(self) -> int:
        return self._get_collection().count()


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings())
