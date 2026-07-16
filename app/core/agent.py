"""Agent 工作流 — LangGraph 多步骤推理 + 工具调用"""
from typing import List, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from app.core.retriever import retrieve_with_rerank
from app.core.generator import generate_answer, extract_citations


class AgentState(TypedDict):
    query: str
    history: List[dict]
    retrieved_docs: List[dict]
    answer: str
    citations: List[dict]
    steps: Annotated[List[str], operator.add]  # 思考步骤记录


async def step_retrieve(state: AgentState) -> AgentState:
    """步骤1：检索相关文档"""
    docs = await retrieve_with_rerank(state["query"])
    return {
        "retrieved_docs": docs,
        "steps": [f"检索完成，找到 {len(docs)} 个相关段落"],
    }


async def step_verify(state: AgentState) -> dict:
    """步骤2：验证检索质量，决定是否需要扩大检索"""
    docs = state.get("retrieved_docs", [])
    # 如果最高分太低，标记为低质量
    if not docs or docs[0].get("score", 0) < 0.3:
        return {"steps": ["检索质量较低，可能无法给出准确回答"]}
    return {"steps": ["检索质量良好，开始生成回答"]}


async def step_generate(state: AgentState) -> AgentState:
    """步骤3：生成回答 + 引用"""
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {
            "answer": "抱歉，未在知识库中找到相关信息。请尝试上传更多相关文档。",
            "citations": [],
            "steps": ["无相关文档，无法生成回答"],
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


# ── 构建 Agent 图 ──

def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", step_retrieve)
    workflow.add_node("verify", step_verify)
    workflow.add_node("generate", step_generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "verify")
    workflow.add_edge("verify", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# 全局 Agent 实例
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent_graph()
    return _agent


async def run_agent(query: str, history: List[dict] = None) -> dict:
    """运行 Agent 工作流"""
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
