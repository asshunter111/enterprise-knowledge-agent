import asyncio
import hashlib
import math
import re
from functools import lru_cache

from app.config import Settings, get_settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.settings.embedding_backend == "hash":
            return [self._hash_embedding(text) for text in texts]
        return await asyncio.to_thread(self._encode_bge, texts)

    async def embed_query(self, query: str) -> list[float]:
        if self.settings.embedding_backend == "hash":
            return self._hash_embedding(query)
        return (await asyncio.to_thread(self._encode_bge, [query]))[0]

    def _encode_bge(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model, device="cpu")
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def _hash_embedding(self, text: str) -> list[float]:
        dimension = self.settings.embedding_dimension
        vector = [0.0] * dimension
        tokens = self._tokens(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = text.lower()
        words = re.findall(r"[a-z0-9_]+", normalized)
        chinese = re.findall(r"[\u4e00-\u9fff]", normalized)
        chinese_bigrams = ["".join(chinese[index : index + 2]) for index in range(len(chinese) - 1)]
        return words + chinese + chinese_bigrams or [normalized]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())
