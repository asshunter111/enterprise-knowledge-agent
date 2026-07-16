"""LLM 答案生成器 — 基于检索内容生成带引用的回答"""
from typing import List, AsyncIterator

from langchain_openai import ChatOpenAI

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

SYSTEM_PROMPT = """你是一个企业知识库智能助手。请严格基于以下【参考资料】回答用户问题。

规则：
1. 如果资料中有答案，直接基于资料回答，并在回答末尾标注引用来源。
2. 如果资料中没有答案，明确说"根据现有资料无法回答"，不要编造。
3. 回答简洁专业，使用 Markdown 格式。
4. 引用格式：[来源: {文档名} 段落{段落号}]

【参考资料】
{context}"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0.3,
    )


def build_context(docs: List[dict]) -> str:
    """用检索结果构建上下文"""
    parts = []
    for i, doc in enumerate(docs, 1):
        name = doc["metadata"].get("doc_name", "未知文档")
        chunk_idx = doc["metadata"].get("chunk_index", i)
        parts.append(f"[{i}] 来源: {name} 段落{chunk_idx}\n{doc['content']}")
    return "\n\n---\n\n".join(parts)


def extract_citations(docs: List[dict]) -> List[dict]:
    """提取引用信息"""
    return [
        {
            "index": i + 1,
            "doc_name": d["metadata"].get("doc_name", "未知"),
            "chunk_index": d["metadata"].get("chunk_index", i),
            "content_preview": d["content"][:200],
            "score": round(d.get("score", d.get("rerank_score", 0)), 4),
        }
        for i, d in enumerate(docs)
    ]


async def generate_answer(
    query: str,
    retrieved_docs: List[dict],
    history: List[dict] = None,
) -> dict:
    """生成完整回答（非流式）"""
    llm = _get_llm()
    context = build_context(retrieved_docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    if history:
        messages.extend(history[-6:])  # 只保留最近6条
    messages.append({"role": "user", "content": query})

    response = await llm.ainvoke(messages)
    return {
        "answer": response.content,
        "citations": extract_citations(retrieved_docs),
    }


async def generate_answer_stream(
    query: str,
    retrieved_docs: List[dict],
    history: List[dict] = None,
) -> AsyncIterator[str]:
    """流式生成回答"""
    llm = _get_llm()
    context = build_context(retrieved_docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content
