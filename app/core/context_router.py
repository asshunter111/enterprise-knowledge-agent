import re
from functools import lru_cache

DIRECT = "direct"
INDEPENDENT = "independent"
CONTINUE = "continue"
NEW_INTENT = "new_intent"

DEFAULT_REPLY = "你好，有什么可以帮你？"

# 打招呼、致谢、告别这类请求既不需要检索也不需要生成，直接回固定话术
_SMALL_TALK: dict[str, tuple[str, ...]] = {
    "你好，有什么可以帮你？": ("你好", "您好", "hi", "hello", "在吗"),
    "不客气，还有其他问题随时问我。": ("谢谢", "感谢", "多谢", "thanks", "thank you"),
    "再见，有问题随时回来找我。": ("再见", "拜拜", "bye", "回头见"),
}
_SMALL_TALK_MAX_LENGTH = 12
_SMALL_TALK_TOLERANCE = 4

# 指代、省略和序数引用离开历史后无法独立成立，命中这些特征的请求需要恢复上下文
_CONTINUATION_PATTERNS: tuple[str, ...] = (
    r"^那么",
    r"^那(?![些么])",
    r"^还有",
    r"^继续",
    r"呢[?？]?$",
    r"^这个",
    r"^这些",
    r"^它",
    r"^他",
    r"^她",
    r"第[一二三四五六七八九十百\d]+[条项个点次]",
    r"上[一条面述]",
    r"^前面",
    r"^刚才",
    r"^上述",
)

_PUNCTUATION = re.compile(r"[\s,，。.!！?？~～、;；:：\"'“”‘’()（）]")
_CONTINUATION = re.compile("|".join(_CONTINUATION_PATTERNS))


def direct_reply(query: str) -> str | None:
    """纯寒暄返回固定回复，其余返回 None。"""

    compact = _PUNCTUATION.sub("", query).lower()
    if not compact or len(compact) > _SMALL_TALK_MAX_LENGTH:
        return None
    for reply, keywords in _SMALL_TALK.items():
        for keyword in keywords:
            if keyword in compact and len(compact) <= len(keyword) + _SMALL_TALK_TOLERANCE:
                return reply
    return None


class ContextRouter:
    """判断当前请求该走直接回答、独立检索，还是先恢复上下文再检索。"""

    def route(self, query: str, history: list[dict], active_context: dict | None) -> str:
        text = query.strip()
        if direct_reply(text) is not None:
            return DIRECT
        if not history and not active_context:
            return INDEPENDENT
        if _CONTINUATION.search(_PUNCTUATION.sub("", text).lower()):
            return CONTINUE
        return NEW_INTENT


@lru_cache
def get_context_router() -> ContextRouter:
    return ContextRouter()
