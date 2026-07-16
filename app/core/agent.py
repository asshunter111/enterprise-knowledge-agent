import operator
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.core.generator import AnswerGenerator, get_answer_generator
from app.core.retriever import Retriever, get_retriever


class AgentState(TypedDict, total=False):
    query: str
    history: list[dict]
    candidates: list[dict]
    documents: list[dict]
    answer: str
    citations: list[dict]
    trace: Annotated[list[str], operator.add]


class EnterpriseRAGAgent:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        generator: AnswerGenerator,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.graph = self._build_graph()
        self.context_graph = self._build_context_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("verify", self._verify)
        graph.add_node("rerank", self._rerank)
        graph.add_node("generate", self._generate)
        graph.add_node("no_context", self._no_context)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "verify")
        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"rerank": "rerank", "no_context": "no_context"},
        )
        graph.add_edge("rerank", "generate")
        graph.add_edge("generate", END)
        graph.add_edge("no_context", END)
        return graph.compile()

    def _build_context_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("verify", self._verify)
        graph.add_node("rerank", self._rerank)
        graph.add_node("no_context", self._no_context)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "verify")
        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"rerank": "rerank", "no_context": "no_context"},
        )
        graph.add_edge("rerank", END)
        graph.add_edge("no_context", END)
        return graph.compile()

    async def run(self, query: str, history: list[dict] | None = None) -> dict:
        state = await self.graph.ainvoke(self._initial_state(query, history))
        return {
            "answer": state.get("answer", "资料中未找到相关信息。"),
            "citations": state.get("citations", []),
            "trace": state.get("trace", []),
            "retrieved_count": len(state.get("documents", [])),
        }

    async def prepare_stream(self, query: str, history: list[dict] | None = None) -> dict:
        state = await self.context_graph.ainvoke(self._initial_state(query, history))
        return {
            "documents": state.get("documents", []),
            "citations": self.generator.build_citations(state.get("documents", [])),
            "trace": state.get("trace", []),
        }

    async def _retrieve(self, state: AgentState) -> dict:
        candidates = await self.retriever.retrieve(state["query"])
        return {
            "candidates": candidates,
            "trace": [f"retrieve: {len(candidates)} candidates"],
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
        documents = await self.retriever.reranker.rerank(
            state["query"], state.get("candidates", [])
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
            "answer": "资料中未找到相关信息。",
            "citations": [],
            "trace": ["no_context: no reliable evidence"],
        }

    @staticmethod
    def _route_after_verify(state: AgentState) -> str:
        return "rerank" if state.get("candidates") else "no_context"

    @staticmethod
    def _initial_state(query: str, history: list[dict] | None) -> AgentState:
        return {
            "query": query,
            "history": history or [],
            "candidates": [],
            "documents": [],
            "answer": "",
            "citations": [],
            "trace": [],
        }


@lru_cache
def get_agent() -> EnterpriseRAGAgent:
    return EnterpriseRAGAgent(get_settings(), get_retriever(), get_answer_generator())
