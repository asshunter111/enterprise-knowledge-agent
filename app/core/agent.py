import logging
import operator
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.core.context_resolver import ContextResolver
from app.core.context_router import (
    CONTINUE,
    DEFAULT_REPLY,
    DIRECT,
    INDEPENDENT,
    NEW_INTENT,
    ContextRouter,
    direct_reply,
)
from app.core.generator import AnswerGenerator, get_answer_generator
from app.core.retriever import Retriever, get_retriever

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "资料中未找到相关信息。"


class AgentState(TypedDict, total=False):
    query: str
    history: list[dict]

    context_mode: str
    intent: str | None
    slots: dict
    active_context: dict | None

    retrieval_query: str

    raw_candidates: list[dict]
    rewrite_candidates: list[dict]
    candidates: list[dict]
    documents: list[dict]

    answer: str
    citations: list[dict]
    trace: Annotated[list[str], operator.add]


def merge_candidates(raw_candidates: list[dict], rewrite_candidates: list[dict]) -> list[dict]:
    """按 chunk 唯一 ID 合并两路召回，保留各自得分。

    同一个 chunk 被两路同时召回时不做覆盖，raw_score 和 rewrite_score 都留下，
    重排序使用的基础分取两者较高值：只要有一路认可该 chunk，就不应该因为另一路
    没召回而被压低。
    """

    merged: dict[str, dict] = {}
    for item in raw_candidates:
        merged[item["id"]] = dict(item)
    for item in rewrite_candidates:
        existing = merged.get(item["id"])
        if existing is None:
            merged[item["id"]] = dict(item)
        else:
            existing["rewrite_score"] = item.get("rewrite_score")

    candidates = [
        {
            **item,
            "score": max(item.get("raw_score") or 0.0, item.get("rewrite_score") or 0.0),
        }
        for item in merged.values()
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


class EnterpriseRAGAgent:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        generator: AnswerGenerator,
        resolver: ContextResolver | None = None,
        router: ContextRouter | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.resolver = resolver if resolver is not None else ContextResolver(settings)
        self.router = router if router is not None else ContextRouter()
        self.graph = self._build_graph(with_generate=True)
        self.context_graph = self._build_graph(with_generate=False)

    def _build_graph(self, with_generate: bool):
        graph = StateGraph(AgentState)
        graph.add_node("context_router", self._context_router)
        graph.add_node("direct_response", self._direct_response)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("context_resolve", self._context_resolve)
        graph.add_node("query_assembly", self._query_assembly)
        graph.add_node("raw_retrieve", self._raw_retrieve)
        graph.add_node("rewrite_retrieve", self._rewrite_retrieve)
        graph.add_node("merge", self._merge)
        graph.add_node("verify", self._verify)
        graph.add_node("rerank", self._rerank)
        graph.add_node("no_context", self._no_context)

        graph.add_edge(START, "context_router")
        graph.add_conditional_edges(
            "context_router",
            self._route_after_router,
            {
                "direct": "direct_response",
                "independent": "retrieve",
                "contextual": "context_resolve",
            },
        )
        graph.add_edge("direct_response", END)
        graph.add_edge("retrieve", "verify")
        graph.add_edge("context_resolve", "query_assembly")
        graph.add_edge("query_assembly", "raw_retrieve")
        graph.add_conditional_edges(
            "raw_retrieve",
            self._route_after_raw_retrieve,
            {"rewrite": "rewrite_retrieve", "merge": "merge"},
        )
        graph.add_edge("rewrite_retrieve", "merge")
        graph.add_edge("merge", "verify")
        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"rerank": "rerank", "no_context": "no_context"},
        )
        graph.add_edge("no_context", END)

        if with_generate:
            graph.add_node("generate", self._generate)
            graph.add_edge("rerank", "generate")
            graph.add_edge("generate", END)
        else:
            graph.add_edge("rerank", END)
        return graph.compile()

    async def run(
        self,
        query: str,
        history: list[dict] | None = None,
        active_context: dict | None = None,
    ) -> dict:
        state = await self.graph.ainvoke(self._initial_state(query, history, active_context))
        return {
            "answer": state.get("answer") or FALLBACK_ANSWER,
            "citations": state.get("citations", []),
            "trace": state.get("trace", []),
            "retrieved_count": len(state.get("documents", [])),
            "active_context": state.get("active_context"),
            "retrieval_query": state.get("retrieval_query") or query,
            "context_mode": state.get("context_mode", ""),
        }

    async def prepare_stream(
        self,
        query: str,
        history: list[dict] | None = None,
        active_context: dict | None = None,
    ) -> dict:
        state = await self.context_graph.ainvoke(
            self._initial_state(query, history, active_context)
        )
        documents = state.get("documents", [])
        direct_answer = state.get("answer") if state.get("context_mode") == DIRECT else None
        return {
            "documents": documents,
            "citations": self.generator.build_citations(documents),
            "trace": state.get("trace", []),
            "direct_answer": direct_answer or None,
            "active_context": state.get("active_context"),
        }

    async def _context_router(self, state: AgentState) -> dict:
        query = state["query"]
        mode = self.router.route(query, state.get("history", []), state.get("active_context"))

        if mode == DIRECT:
            return {
                "context_mode": DIRECT,
                "retrieval_query": query,
                "trace": ["context_router: direct response"],
            }
        if mode == INDEPENDENT:
            # 单轮独立问题不经过上下文层，trace 与改造前保持一致
            return {"context_mode": INDEPENDENT, "retrieval_query": query}

        update: dict = {
            "context_mode": mode,
            "retrieval_query": query,
            "trace": [f"context_router: {mode}"],
        }
        if mode == NEW_INTENT:
            # 只切断旧意图，历史消息仍然保留
            update["active_context"] = None
        return update

    async def _direct_response(self, state: AgentState) -> dict:
        return {
            "documents": [],
            "answer": direct_reply(state["query"]) or DEFAULT_REPLY,
            "citations": [],
            "trace": ["direct_response: completed"],
        }

    async def _retrieve(self, state: AgentState) -> dict:
        candidates = await self.retriever.retrieve(state["query"])
        return {
            "candidates": candidates,
            "trace": [f"retrieve: {len(candidates)} candidates"],
        }

    async def _context_resolve(self, state: AgentState) -> dict:
        query = state["query"]
        active_context = state.get("active_context")
        try:
            resolved = await self.resolver.resolve(
                query, self._resolver_history(state.get("history", [])), active_context
            )
        except Exception as exc:
            # 解析失败不能让整个问答挂掉：退回原始 Query，单路召回照常工作
            logger.warning("context resolution failed: %s", exc)
            return {
                "retrieval_query": query,
                "trace": [f"context_resolve: fallback ({exc})"],
            }

        if state.get("context_mode") == CONTINUE:
            previous_slots = (active_context or {}).get("slots") or {}
            slots = {**previous_slots, **resolved["slots"]}
        else:
            slots = dict(resolved["slots"])

        intent = resolved["intent"]
        return {
            "intent": intent,
            "slots": slots,
            "active_context": {"intent": intent, "slots": slots},
            "retrieval_query": resolved["retrieval_query"],
            "trace": [f"context_resolve: {intent}"],
        }

    async def _query_assembly(self, state: AgentState) -> dict:
        # Resolver 已经给出可靠改写时直接使用，这里只做兜底，不再为形式多调一次 LLM
        retrieval_query = (state.get("retrieval_query") or "").strip() or state["query"]
        return {
            "retrieval_query": retrieval_query,
            "trace": [f"query_assembly: {retrieval_query}"],
        }

    async def _raw_retrieve(self, state: AgentState) -> dict:
        candidates = await self.retriever.retrieve(state["query"])
        trace = [f"raw_retrieve: {len(candidates)} candidates"]
        if not self._needs_rewrite_retrieval(state):
            trace.append("rewrite_retrieve: skipped (identical to raw query)")
        return {
            "raw_candidates": [
                {**item, "raw_score": item.get("score", 0.0), "rewrite_score": None}
                for item in candidates
            ],
            "trace": trace,
        }

    async def _rewrite_retrieve(self, state: AgentState) -> dict:
        candidates = await self.retriever.retrieve(state["retrieval_query"])
        return {
            "rewrite_candidates": [
                {**item, "raw_score": None, "rewrite_score": item.get("score", 0.0)}
                for item in candidates
            ],
            "trace": [f"rewrite_retrieve: {len(candidates)} candidates"],
        }

    async def _merge(self, state: AgentState) -> dict:
        candidates = merge_candidates(
            state.get("raw_candidates", []), state.get("rewrite_candidates", [])
        )
        return {
            "candidates": candidates,
            "trace": [f"merge: {len(candidates)} unique candidates"],
        }

    async def _verify(self, state: AgentState) -> dict:
        candidates = [
            item
            for item in state.get("candidates", [])
            if item.get("score", 0.0) >= self.settings.min_relevance_score
        ]
        return {
            "candidates": candidates,
            "trace": [f"verify: {len(candidates)} candidates passed threshold"],
        }

    async def _rerank(self, state: AgentState) -> dict:
        # 用组装后的检索问题排序，生成阶段仍然使用用户原始问题
        documents = await self.retriever.reranker.rerank(
            state["retrieval_query"], state.get("candidates", [])
        )
        return {
            "documents": documents,
            "trace": [f"rerank: selected {len(documents)} chunks"],
        }

    async def _generate(self, state: AgentState) -> dict:
        result = await self.generator.generate(
            state["query"], state.get("documents", []), state.get("history", [])
        )
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "trace": ["generate: answer completed"],
        }

    @staticmethod
    async def _no_context(_: AgentState) -> dict:
        return {
            "documents": [],
            "answer": FALLBACK_ANSWER,
            "citations": [],
            "trace": ["no_context: no reliable evidence"],
        }

    def _resolver_history(self, history: list[dict]) -> list[dict]:
        limit = self.settings.context_history_turns * 2
        if limit <= 0:
            return []
        return history[-limit:]

    @staticmethod
    def _needs_rewrite_retrieval(state: AgentState) -> bool:
        retrieval_query = (state.get("retrieval_query") or "").strip()
        return bool(retrieval_query) and retrieval_query != (state.get("query") or "").strip()

    @staticmethod
    def _route_after_router(state: AgentState) -> str:
        mode = state.get("context_mode")
        if mode == DIRECT:
            return "direct"
        if mode == INDEPENDENT:
            return "independent"
        return "contextual"

    @staticmethod
    def _route_after_raw_retrieve(state: AgentState) -> str:
        return "rewrite" if EnterpriseRAGAgent._needs_rewrite_retrieval(state) else "merge"

    @staticmethod
    def _route_after_verify(state: AgentState) -> str:
        return "rerank" if state.get("candidates") else "no_context"

    @staticmethod
    def _initial_state(
        query: str,
        history: list[dict] | None,
        active_context: dict | None = None,
    ) -> AgentState:
        return {
            "query": query,
            "history": history or [],
            "context_mode": "",
            "intent": None,
            "slots": {},
            "active_context": active_context,
            "retrieval_query": query,
            "raw_candidates": [],
            "rewrite_candidates": [],
            "candidates": [],
            "documents": [],
            "answer": "",
            "citations": [],
            "trace": [],
        }


@lru_cache
def get_agent() -> EnterpriseRAGAgent:
    return EnterpriseRAGAgent(get_settings(), get_retriever(), get_answer_generator())
