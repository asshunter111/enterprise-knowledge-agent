"""检索器 — 混合检索 + 重排序"""
from typing import List

from app.config import RETRIEVAL_TOP_K, RERANK_TOP_K, USE_RERANKER
from app.core.embedding import embed_query
from app.core.vector_store import search


async def retrieve(query: str, top_k: int = None) -> List[dict]:
    """检索相关文档段落"""
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    # 1. 向量检索
    q_embedding = await embed_query(query)
    results = await search(q_embedding, top_k=top_k)
    return results


async def retrieve_with_rerank(query: str) -> List[dict]:
    """检索 + 重排序"""
    # 1. 初检（取 top_k * 2 做粗筛）
    candidates = await retrieve(query, top_k=RETRIEVAL_TOP_K * 2)

    if not USE_RERANKER or len(candidates) <= RERANK_TOP_K:
        return candidates[:RERANK_TOP_K]

    # 2. Reranker 精排（Cross-Encoder）
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        pairs = [[query, c["content"]] for c in candidates]
        scores = reranker.compute_score(pairs)
        # 按分数降序
        for i, s in enumerate(scores):
            candidates[i]["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    except ImportError:
        pass  # 没装 FlagEmbedding 则跳过重排

    return candidates[:RERANK_TOP_K]
