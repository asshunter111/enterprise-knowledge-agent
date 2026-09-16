import json
import re
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

RESOLVER_PROMPT = """你是多轮对话的上下文解析器，只负责还原问题的完整语义，不负责回答。

规则：
1. 只根据给出的历史对话和已有上下文进行恢复，不得引入历史中不存在的信息。
2. 保留用户当前问题的真实意图，不要替换成历史中的其他意图。
3. 识别指代关系，判断“他”“这个”“那条”等指向历史中的哪个对象。
4. 识别省略内容，补全被省略的主语、宾语、时间或条件。
5. 无法可靠恢复时，retrieval_query 使用用户当前问题的原文。
6. 只输出一个 JSON 对象，不要输出解释、Markdown 或代码块。

JSON 结构：
{
  "intent": "当前问题的业务意图，简短英文短语，例如 schedule_query、finance_policy",
  "slots": {"槽位名": "槽位值"},
  "retrieval_query": "还原后的完整中文检索问题"
}"""


class ContextResolutionError(RuntimeError):
    """上下文解析失败，调用方应回落到用户原始问题。"""


class ContextResolver:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm = None

    async def resolve(
        self, query: str, history: list[dict], active_context: dict | None
    ) -> dict:
        if not self.settings.llm_api_key:
            raise ContextResolutionError("llm api key is not configured")
        try:
            response = await self._get_llm().ainvoke(
                self._messages(query, history, active_context)
            )
        except Exception as exc:
            raise ContextResolutionError(f"resolver llm call failed: {exc}") from exc
        return self._parse(_content_to_text(response.content))

    def _messages(
        self, query: str, history: list[dict], active_context: dict | None
    ) -> list[dict]:
        lines: list[str] = []
        if active_context:
            lines.append(f"已有上下文：{json.dumps(active_context, ensure_ascii=False)}")
        if history:
            lines.append("最近对话：")
            for message in history:
                role = "用户" if message.get("role") == "user" else "助手"
                lines.append(f"{role}：{message.get('content', '')}")
        lines.append(f"当前问题：{query}")
        return [
            {"role": "system", "content": RESOLVER_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]

    @staticmethod
    def _parse(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ContextResolutionError("resolver did not return a json object")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ContextResolutionError("resolver returned invalid json") from exc
        if not isinstance(payload, dict):
            raise ContextResolutionError("resolver returned invalid json")

        retrieval_query = str(payload.get("retrieval_query") or "").strip()
        if not retrieval_query:
            raise ContextResolutionError("resolver returned empty retrieval_query")

        slots = payload.get("slots")
        return {
            "intent": str(payload.get("intent") or "").strip() or "unknown",
            "slots": slots if isinstance(slots, dict) else {},
            "retrieval_query": retrieval_query,
        }

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                temperature=0.0,
            )
        return self._llm


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


@lru_cache
def get_context_resolver() -> ContextResolver:
    return ContextResolver(get_settings())
