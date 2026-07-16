import asyncio
import re
from functools import lru_cache

from app.config import Settings, get_settings


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    async def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        if not documents:
            return []
        if self.settings.rerank_backend == "cross_encoder":
            try:
                return await asyncio.to_thread(self._cross_encoder_rerank, query, documents)
            except Exception:
                pass
        return self._lexical_rerank(query, documents)

    def _cross_encoder_rerank(self, query: str, documents: list[dict]) -> list[dict]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.rerank_model, device="cpu")
        scores = self._model.predict([(query, item["content"]) for item in documents])
        reranked = []
        for item, score in zip(documents, scores, strict=True):
            reranked.append({**item, "rerank_score": float(score)})
        return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[
            : self.settings.rerank_top_k
        ]

    def _lexical_rerank(self, query: str, documents: list[dict]) -> list[dict]:
        query_tokens = self._tokens(query)
        reranked = []
        for item in documents:
            document_tokens = self._tokens(item["content"])
            overlap = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
            score = item.get("score", 0.0) * 0.65 + overlap * 0.35
            reranked.append({**item, "rerank_score": score})
        return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[
            : self.settings.rerank_top_k
        ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        chinese = re.findall(r"[\u4e00-\u9fff]", text)
        return set(words + chinese)


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings())
