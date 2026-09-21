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
from app.core.mcp_client import MCPClientError, MCPStdioClient
from app.core.permissions import filter_documents
from app.core.retriever import Retriever, get_retriever
from app.core.tools import (
    BusinessToolRegistry,
    NativeToolCall,
    ProviderToolCaller,
    ToolCallingError,
    ToolExecutionError,
    build_default_tool_registry,
)

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
    user_id: str
    role: str
    tool_name: str | None
    tool_source: str | None
    tool_arguments: dict
    tool_result: dict | None
    native_tool_call: NativeToolCall | None
    abstained: bool
    memory_context: list[dict]
    diagnostics: dict
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
        tool_registry: BusinessToolRegistry | None = None,
        memory_service=None,
        tool_caller: ProviderToolCaller | None = None,
        mcp_client_factory=None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.resolver = resolver if resolver is not None else ContextResolver(settings)
        self.router = router if router is not None else ContextRouter()
        self.tool_registry = tool_registry or build_default_tool_registry()
        self.memory_service = memory_service
        self.tool_caller = tool_caller or ProviderToolCaller(settings, self.tool_registry)
        self.mcp_client_factory = mcp_client_factory or self._default_mcp_client
        self.graph = self._build_graph(with_generate=True)
        self.context_graph = self._build_graph(with_generate=False)

    def _build_graph(self, with_generate: bool):
        graph = StateGraph(AgentState)
        graph.add_node("context_router", self._context_router)
        graph.add_node("memory_context", self._memory_context)
        graph.add_node("tool_execute", self._tool_execute)
        graph.add_node("tool_response", self._tool_response)
        graph.add_node("direct_response", self._direct_response)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("context_resolve", self._context_resolve)
        graph.add_node("query_assembly", self._query_assembly)
        graph.add_node("raw_retrieve", self._raw_retrieve)
        graph.add_node("rewrite_retrieve", self._rewrite_retrieve)
        graph.add_node("merge", self._merge)
        graph.add_node("verify", self._verify)
        graph.add_node("rerank", self._rerank)
        graph.add_node("evidence_decision", self._evidence_decision)
        graph.add_node("no_context", self._no_context)

        graph.add_edge(START, "memory_context")
        graph.add_edge("memory_context", "context_router")
        graph.add_conditional_edges(
            "context_router",
            self._route_after_router,
            {
                "direct": "direct_response",
                "tool": "tool_execute",
                "independent": "retrieve",
                "contextual": "context_resolve",
            },
        )
        graph.add_edge("direct_response", END)
        graph.add_edge("tool_execute", "tool_response")
        graph.add_edge("tool_response", END)
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

        graph.add_edge("rerank", "evidence_decision")
        graph.add_conditional_edges(
            "evidence_decision",
            self._route_after_evidence,
            {"generate": "generate" if with_generate else END, "abstain": "no_context"},
        )

        if with_generate:
            graph.add_node("generate", self._generate)
            graph.add_edge("generate", END)
        else:
            graph.add_edge("no_context", END)
        return graph.compile()

    async def run(
        self,
        query: str,
        history: list[dict] | None = None,
        active_context: dict | None = None,
        user_id: str = "anonymous",
        role: str = "employee",
    ) -> dict:
        state = await self.graph.ainvoke(
            self._initial_state(query, history, active_context, user_id, role)
        )
        return {
            "answer": state.get("answer") or FALLBACK_ANSWER,
            "citations": state.get("citations", []),
            "trace": state.get("trace", []),
            "retrieved_count": len(state.get("documents", [])),
            "active_context": state.get("active_context"),
            "retrieval_query": state.get("retrieval_query") or query,
            "context_mode": state.get("context_mode", ""),
            "tool": state.get("tool_name"),
            "tool_source": state.get("tool_source"),
            "memory_used": bool(state.get("memory_context")),
            "diagnostics": state.get("diagnostics", {}),
            "abstained": state.get("abstained", False),
        }

    async def prepare_stream(
        self,
        query: str,
        history: list[dict] | None = None,
        active_context: dict | None = None,
        user_id: str = "anonymous",
        role: str = "employee",
    ) -> dict:
        state = await self.context_graph.ainvoke(
            self._initial_state(query, history, active_context, user_id, role)
        )
        documents = state.get("documents", [])
        direct_answer = (
            state.get("answer")
            if state.get("context_mode") in {DIRECT, "tool"}
            else None
        )
        return {
            "documents": documents,
            "citations": self.generator.build_citations(documents),
            "trace": state.get("trace", []),
            "direct_answer": direct_answer or None,
            "active_context": state.get("active_context"),
            "memory_context": state.get("memory_context", []),
        }

    async def _context_router(self, state: AgentState) -> dict:
        query = state["query"]
        fallback_trace: list[str] = []
        if self.settings.llm_api_key:
            try:
                native_call = await self.tool_caller.propose(
                    query, state.get("history", []), state.get("user_id", "anonymous")
                )
            except ToolCallingError as exc:
                logger.warning("provider tool calling unavailable: %s", exc)
                fallback_trace.append(f"tool_call: deterministic fallback ({exc})")
            else:
                if native_call is not None:
                    return {
                        "context_mode": "tool",
                        "tool_name": native_call.name,
                        "tool_source": "local",
                        "tool_arguments": native_call.arguments,
                        "native_tool_call": native_call,
                        "trace": [f"context_router: native tool ({native_call.name})"],
                    }

        tool_intent = self._detect_tool_intent(query, state.get("user_id", "anonymous"))
        if tool_intent is not None:
            tool_name, tool_arguments = tool_intent
            tool_source = "local"
            if tool_name == "query_meeting_rooms":
                tool_source = "mcp"
            route_label = "mcp tool" if tool_source == "mcp" else "tool"
            return {
                "context_mode": "tool",
                "tool_name": tool_name,
                "tool_source": tool_source,
                "tool_arguments": tool_arguments,
                "trace": fallback_trace + [f"context_router: {route_label} ({tool_name})"],
            }
        mode = self.router.route(query, state.get("history", []), state.get("active_context"))

        if mode == DIRECT:
            return {
                "context_mode": DIRECT,
                "retrieval_query": query,
                "trace": fallback_trace + ["context_router: direct response"],
            }
        if mode == INDEPENDENT:
            # 单轮独立问题不经过上下文层，trace 与改造前保持一致
            return {"context_mode": INDEPENDENT, "retrieval_query": query}

        update: dict = {
            "context_mode": mode,
            "retrieval_query": query,
            "trace": fallback_trace + [f"context_router: {mode}"],
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

    async def _memory_context(self, state: AgentState) -> dict:
        if self.memory_service is None or not state.get("user_id"):
            return {"memory_context": []}
        try:
            memories = await self.memory_service.retrieve(
                state["user_id"], state["query"]
            )
        except Exception as exc:
            logger.warning("memory retrieval failed: %s", exc)
            return {"memory_context": []}
        return {
            "memory_context": [
                {"key": item.key, "value": item.value, "memory_type": item.memory_type}
                for item in memories
            ]
        }

    async def _tool_execute(self, state: AgentState) -> dict:
        name = state.get("tool_name") or ""
        if state.get("tool_source") == "mcp":
            try:
                result = await self._execute_mcp_tool(
                    "query_meeting_rooms", state.get("tool_arguments") or {}
                )
                return {"tool_result": result, "trace": [f"mcp: executed {name}"]}
            except MCPClientError as exc:
                logger.warning("MCP tool execution failed: %s", exc)
                return {
                    "tool_result": None,
                    "answer": "MCP 业务工具当前不可用，请稍后重试。",
                    "trace": [f"mcp: failed {name} ({exc})"],
                }
        if name == "get_reimbursement_status" and state.get("role") not in {
            "finance",
            "admin",
            "manager",
        }:
            if state.get("native_tool_call") is not None:
                return {
                    "tool_result": {"error": "当前角色无权查询报销单状态。"},
                    "trace": [f"tool: denied {name}"],
                }
            return {
                "tool_result": None,
                "answer": "当前角色无权查询报销单状态。",
                "trace": [f"tool: denied {name}"],
            }
        try:
            result = await self.tool_registry.execute(
                name, **(state.get("tool_arguments") or {})
            )
            return {"tool_result": result, "trace": [f"tool: executed {name}"]}
        except ToolExecutionError as exc:
            logger.warning("tool execution failed: %s", exc)
            if state.get("native_tool_call") is not None:
                return {
                    "tool_result": {"error": "当前无法完成该业务查询。"},
                    "trace": [f"tool: failed {name}"],
                }
            return {
                "tool_result": None,
                "answer": "当前无法完成该业务查询。",
                "trace": [f"tool: failed {name}"],
            }

    async def _tool_response(self, state: AgentState) -> dict:
        result = state.get("tool_result")
        native_call = state.get("native_tool_call")
        if native_call is not None:
            try:
                answer = await self.tool_caller.complete(native_call, result or {})
                if answer.strip():
                    return {
                        "answer": answer,
                        "citations": [],
                        "trace": ["tool: native response completed"],
                    }
            except ToolCallingError as exc:
                logger.warning("provider tool response unavailable: %s", exc)
                return {
                    "answer": self._format_tool_result(state.get("tool_name"), result),
                    "citations": [],
                    "trace": [f"tool: response fallback ({exc})"],
                }
        if not result:
            return {"answer": state.get("answer") or "当前无法完成该业务查询。"}
        return {
            "answer": self._format_tool_result(state.get("tool_name"), result),
            "citations": [],
            "trace": ["tool: response completed"],
        }

    @staticmethod
    def _format_tool_result(name: str | None, result: dict | None) -> str:
        if not result or "error" in result:
            if result:
                return result.get("error", "当前无法完成该业务查询。")
            return "当前无法完成该业务查询。"
        if name == "get_leave_balance":
            answer = f"员工 {result['employee_id']} 还剩 {result['annual_leave_days']} 天年假。"
        elif name == "get_reimbursement_status":
            answer = f"报销单 {result['reimbursement_id']} 当前状态是：{result['status']}。"
        else:
            rooms = "、".join(result.get("rooms", [])) or "暂无空闲会议室"
            answer = f"{result.get('date')} {result.get('time')} 可用会议室：{rooms}。"
        return answer

    async def _execute_mcp_tool(self, name: str, arguments: dict) -> dict:
        async with self.mcp_client_factory() as client:
            available = {tool.name for tool in await client.list_tools()}
            if name not in available:
                raise MCPClientError(f"MCP tool not found: {name}")
            return await client.call_tool(name, arguments)

    def _default_mcp_client(self) -> MCPStdioClient:
        return MCPStdioClient(
            command=self.settings.mcp_server_command,
            args=["-m", self.settings.mcp_server_module],
            timeout_seconds=self.settings.mcp_timeout_seconds,
        )

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
        permitted = filter_documents(state.get("candidates", []), state.get("role", "employee"))
        scores = [float(item.get("score", 0.0)) for item in permitted]
        candidates = [
            item
            for item in permitted
            if item.get("score", 0.0) >= self.settings.min_relevance_score
        ]
        return {
            "candidates": candidates,
            "diagnostics": {
                "query": state["query"],
                "raw_candidate_count": len(state.get("candidates", [])),
                "permission_allowed_count": len(permitted),
                "raw_scores": scores,
                "verified_count": len(candidates),
                "threshold": self.settings.min_relevance_score,
            },
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

    async def _evidence_decision(self, state: AgentState) -> dict:
        documents = state.get("documents", [])
        top_score = max(
            (float(item.get("rerank_score", item.get("score", 0.0))) for item in documents),
            default=0.0,
        )
        sufficient = top_score >= self.settings.evidence_min_rerank_score
        diagnostics = {
            **state.get("diagnostics", {}),
            "evidence_top_rerank_score": round(top_score, 4),
            "evidence_threshold": self.settings.evidence_min_rerank_score,
            "evidence_sufficient": sufficient,
        }
        if sufficient:
            return {
                "diagnostics": diagnostics,
            }
        return {
            "documents": [],
            "citations": [],
            "abstained": True,
            "diagnostics": diagnostics,
            "trace": [f"evidence: insufficient ({top_score:.4f})"],
        }

    async def _generate(self, state: AgentState) -> dict:
        history = list(state.get("history", []))
        if state.get("memory_context"):
            memory_text = "；".join(
                f"{item['key']}={item['value']}" for item in state["memory_context"]
            )
            history.append(
                {"role": "system", "content": f"相关用户记忆（非知识库事实）：{memory_text}"}
            )
        result = await self.generator.generate(
            state["query"], state.get("documents", []), history
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
            "abstained": True,
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
        if mode == "tool":
            return "tool"
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
    def _route_after_evidence(state: AgentState) -> str:
        return "generate" if not state.get("abstained") else "abstain"

    @staticmethod
    def _initial_state(
        query: str,
        history: list[dict] | None,
        active_context: dict | None = None,
        user_id: str = "anonymous",
        role: str = "employee",
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
            "user_id": user_id,
            "role": role,
            "tool_name": None,
            "tool_source": None,
            "tool_arguments": {},
            "tool_result": None,
            "native_tool_call": None,
            "abstained": False,
            "memory_context": [],
            "diagnostics": {},
        }

    @staticmethod
    def _detect_tool_intent(query: str, user_id: str) -> tuple[str, dict] | None:
        if any(token in query for token in ("年假", "年休假", "假期余额")):
            return "get_leave_balance", {"employee_id": user_id}
        import re

        reimbursement = re.search(r"(?:报销单|申请)[\s#号：:]*([A-Za-z0-9-]+)", query)
        if "报销" in query and ("状态" in query or "到哪" in query or reimbursement):
            return "get_reimbursement_status", {
                "reimbursement_id": reimbursement.group(1) if reimbursement else "latest"
            }
        if "会议室" in query and any(token in query for token in ("有没有", "可用", "空闲")):
            return "query_meeting_rooms", {"date": "明天", "time": "下午"}
        return None


@lru_cache
def get_agent() -> EnterpriseRAGAgent:
    return EnterpriseRAGAgent(get_settings(), get_retriever(), get_answer_generator())
