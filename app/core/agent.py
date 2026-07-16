"""Agent workflow — LangGraph orchestration"""
from typing import List, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from app.core.retriever import retrieve_with_rerank
from app.core.generator import generate_answer


class AgentState(TypedDict):
    query: str
    history: List[dict]
    retrieved_docs: List[dict]
    answer: str
    citations: List[dict]
    steps: Annotated[List[str], operator.add]


async def step_retrieve(state: AgentState) -> AgentState:
    docs = await retrieve_with_rerank(state["query"])
    return {
        "retrieved_docs": docs,
        "steps": [f"检索到 {len(docs)} 个相关片段"],
    }


async def step_generate(state: AgentState) -> AgentState:
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {
            "answer": "抱歉，知识库中暂无相关信息，请上传更多文档后重试。",
            "citations": [],
            "steps": ["未找到相关文档"],
        }

    result = await generate_answer(
        query=state["query"],
        retrieved_docs=docs,
        history=state.get("history"),
    )
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "steps": ["回答生成完成"],
    }


def build_graph():
    """构建检索→生成两阶段工作流，保持简单"""
    wf = StateGraph(AgentState)

    wf.add_node("retrieve", step_retrieve)
    wf.add_node("generate", step_generate)

    wf.set_entry_point("retrieve")
    wf.add_edge("retrieve", "generate")
    wf.add_edge("generate", END)

    return wf.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_graph()
    return _agent


async def run_agent(query: str, history: List[dict] = None) -> dict:
    agent = get_agent()
    result = await agent.ainvoke({
        "query": query,
        "history": history or [],
        "retrieved_docs": [],
        "answer": "",
        "citations": [],
        "steps": [],
    })
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "steps": result["steps"],
        "retrieved_count": len(result.get("retrieved_docs", [])),
    }
