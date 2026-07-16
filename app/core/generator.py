"""Answer generation with citation support"""
from typing import List, AsyncIterator

from langchain_openai import ChatOpenAI

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

SYSTEM_PROMPT = """你是一个企业内部知识库助手，只能基于下方【参考资料】回答问题。

要求：
- 资料里有答案 → 直接引用回答，末尾标注来源
- 资料里没有 → 说"资料中未找到相关信息"，严禁编造
- 使用 Markdown 排版
- 引用格式：[来源: {文档名}]

【参考资料】
{context}"""


def _get_llm():
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0.3,
    )


def _build_context(docs: List[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        name = doc["metadata"].get("doc_name", "未知")
        parts.append(f"[{i}] {name}:\n{doc['content']}")
    return "\n\n---\n\n".join(parts)


def _extract_citations(docs: List[dict]) -> List[dict]:
    citations = []
    for i, d in enumerate(docs):
        citations.append({
            "index": i + 1,
            "doc_name": d["metadata"].get("doc_name", "未命名文档"),
            "content_preview": d["content"][:200],
            "score": round(d.get("score", d.get("rerank_score", 0)), 4),
        })
    return citations


async def generate_answer(
    query: str,
    retrieved_docs: List[dict],
    history: List[dict] = None,
) -> dict:
    llm = _get_llm()
    context = _build_context(retrieved_docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})

    response = await llm.ainvoke(messages)
    return {
        "answer": response.content,
        "citations": _extract_citations(retrieved_docs),
    }


async def generate_answer_stream(
    query: str,
    retrieved_docs: List[dict],
    history: List[dict] = None,
) -> AsyncIterator[str]:
    llm = _get_llm()
    context = _build_context(retrieved_docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content
