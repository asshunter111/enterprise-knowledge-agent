from functools import lru_cache

from app.config import Settings, get_settings
from app.core.embedding import EmbeddingService, get_embedding_service
from app.core.reranker import Reranker, get_reranker
from app.core.vector_store import VectorStore, get_vector_store


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        reranker: Reranker,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        query_embedding = await self.embedding_service.embed_query(query)
        return await self.vector_store.search(
            query_embedding, top_k or self.settings.retrieval_top_k
        )

    async def retrieve_and_rerank(self, query: str) -> list[dict]:
        candidates = await self.retrieve(query, self.settings.retrieval_top_k)
        return await self.reranker.rerank(query, candidates)


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_settings(), get_embedding_service(), get_vector_store(), get_reranker())
