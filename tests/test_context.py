import pytest

from app.config import Settings
from app.core.agent import EnterpriseRAGAgent, merge_candidates
from tests.fakes import RecordingGenerator, RecordingRetriever, ScriptedResolver

SCHEDULE_HISTORY = [
    {"role": "user", "content": "A组的排班情况怎么样？"},
    {"role": "assistant", "content": "A组本周负责上午班次。"},
]

SCHEDULE_CROSS_TURN_HISTORY = [
    {"role": "user", "content": "A组的排班情况怎么样？"},
    {"role": "assistant", "content": "A组本周负责上午班次。"},
    {"role": "user", "content": "那么B组呢？"},
    {
        "role": "assistant",
        "content": "1. B组本周负责下午班次。\n2. 周一B组安排陈晨、刘洋负责下午值班。",
    },
]


def chunk(chunk_index: int, content: str, score: float) -> dict:
    return {
        "id": f"eval-schedule:{chunk_index}",
        "content": content,
        "metadata": {
            "document_id": "eval-schedule",
            "document_name": "schedule.md",
            "chunk_index": chunk_index,
        },
        "score": score,
    }


class DocumentEchoGenerator(RecordingGenerator):
    """按实际传入的 documents 生成引用，用于验证改写召回命中的是哪篇文档。"""

    def build_citations(self, documents: list[dict]) -> list[dict]:
        return [
            {
                "document_name": item["metadata"]["document_name"],
                "chunk_index": item["metadata"]["chunk_index"],
            }
            for item in documents
        ]


def build_agent(retriever, generator=None, resolver=None) -> EnterpriseRAGAgent:
    return EnterpriseRAGAgent(
        Settings(min_relevance_score=0.05),
        retriever,
        generator or RecordingGenerator(),
        resolver=resolver,
    )


@pytest.mark.asyncio
async def test_direct_response_skips_retrieval_and_generation():
    retriever = RecordingRetriever()
    generator = RecordingGenerator()
    agent = build_agent(retriever, generator)

    result = await agent.run("你好")

    assert result["answer"] == "你好，有什么可以帮你？"
    assert retriever.queries == []
    assert generator.calls == []
    assert result["citations"] == []
    assert result["retrieved_count"] == 0
    assert result["trace"] == ["context_router: direct response", "direct_response: completed"]


@pytest.mark.asyncio
async def test_continue_mode_resolves_context_and_merges_both_retrievals():
    retriever = RecordingRetriever(
        {
            "那么B组呢？": [chunk(0, "A组负责本周一至周五的上午班次。", 0.60)],
            "B组本周的排班情况怎么样？": [
                chunk(1, "B组负责本周一至周五的下午班次。", 0.70),
                chunk(2, "周一B组安排陈晨、刘洋负责下午值班。", 0.55),
            ],
        }
    )
    resolver = ScriptedResolver(
        {
            "intent": "schedule_query",
            "slots": {"group": "B组", "period": "本周"},
            "retrieval_query": "B组本周的排班情况怎么样？",
        }
    )
    agent = build_agent(retriever, resolver=resolver)

    result = await agent.run("那么B组呢？", SCHEDULE_HISTORY)

    assert "B组" in result["retrieval_query"]
    assert result["retrieval_query"] == "B组本周的排班情况怎么样？"
    assert result["context_mode"] == "continue"
    assert result["active_context"] == {
        "intent": "schedule_query",
        "slots": {"group": "B组", "period": "本周"},
    }
    assert retriever.queries == ["那么B组呢？", "B组本周的排班情况怎么样？"]
    assert result["trace"] == [
        "context_router: continue",
        "context_resolve: schedule_query",
        "query_assembly: B组本周的排班情况怎么样？",
        "raw_retrieve: 1 candidates",
        "rewrite_retrieve: 2 candidates",
        "merge: 3 unique candidates",
        "verify: 3 candidates passed threshold",
        "rerank: selected 3 chunks",
        "generate: answer completed",
    ]


@pytest.mark.asyncio
async def test_rewrite_retrieval_recovers_chunk_that_raw_retrieval_misses():
    retriever = RecordingRetriever(
        {
            "第二条的对象是谁？": [],
            "排班规则中第二条的对象是谁？": [
                chunk(3, "各小组原则上按照既定班次执行工作安排。", 0.66)
            ],
        }
    )
    resolver = ScriptedResolver(
        {
            "intent": "schedule_query",
            "slots": {"section": "第二条"},
            "retrieval_query": "排班规则中第二条的对象是谁？",
        }
    )
    agent = build_agent(retriever, DocumentEchoGenerator(), resolver)

    result = await agent.run("第二条的对象是谁？", SCHEDULE_HISTORY)

    assert result["trace"][3] == "raw_retrieve: 0 candidates"
    assert result["trace"][4] == "rewrite_retrieve: 1 candidates"
    assert result["retrieved_count"] == 1
    assert result["citations"][0]["document_name"] == "schedule.md"


@pytest.mark.asyncio
async def test_cross_turn_reference_recovers_slots_from_several_turns():
    retriever = RecordingRetriever(
        {
            "第二条具体是谁负责？": [],
            "B组周一下午班由哪些人负责？": [chunk(4, "周一B组安排陈晨、刘洋负责下午值班。", 0.74)],
        }
    )
    resolver = ScriptedResolver(
        {
            "intent": "schedule_query",
            "slots": {"group": "B组", "period": "周一", "shift": "下午班"},
            "retrieval_query": "B组周一下午班由哪些人负责？",
        }
    )
    agent = build_agent(retriever, resolver=resolver)

    result = await agent.run("第二条具体是谁负责？", SCHEDULE_CROSS_TURN_HISTORY)

    for keyword in ("B组", "周一", "下午班"):
        assert keyword in result["retrieval_query"]
    assert result["active_context"]["slots"] == {
        "group": "B组",
        "period": "周一",
        "shift": "下午班",
    }


@pytest.mark.asyncio
async def test_new_intent_clears_active_context_but_keeps_history():
    retriever = RecordingRetriever({"员工报销需要在费用发生后多久提交？": []})
    resolver = ScriptedResolver(
        {
            "intent": "finance_policy",
            "slots": {"policy_type": "expense_reimbursement", "period": "十个工作日"},
            "retrieval_query": "员工报销需要在费用发生后多久提交？",
        }
    )
    agent = build_agent(retriever, resolver=resolver)

    result = await agent.run(
        "报销期限是多少？",
        SCHEDULE_HISTORY,
        {"intent": "schedule_query", "slots": {"group": "A组"}},
    )

    assert result["context_mode"] == "new_intent"
    assert result["active_context"]["intent"] == "finance_policy"
    assert "group" not in result["active_context"]["slots"]
    # 新意图只切断 active_context，历史消息照常传给解析器
    assert resolver.calls[0]["history"] == SCHEDULE_HISTORY
    assert resolver.calls[0]["active_context"] is None


@pytest.mark.asyncio
async def test_resolver_failure_falls_back_to_original_query():
    retriever = RecordingRetriever(
        {"那么B组呢？": [chunk(0, "A组负责本周一至周五的上午班次。", 0.60)]}
    )
    resolver = ScriptedResolver(error=RuntimeError("resolver exploded"))
    agent = build_agent(retriever, resolver=resolver)

    result = await agent.run("那么B组呢？", SCHEDULE_HISTORY)

    assert result["retrieval_query"] == "那么B组呢？"
    assert retriever.queries == ["那么B组呢？"]
    assert "context_resolve: fallback (resolver exploded)" in result["trace"]
    assert "rewrite_retrieve: skipped (identical to raw query)" in result["trace"]
    assert result["retrieved_count"] == 1


@pytest.mark.asyncio
async def test_missing_llm_api_key_falls_back_without_raising():
    retriever = RecordingRetriever(
        {"那么B组呢？": [chunk(0, "A组负责本周一至周五的上午班次。", 0.60)]}
    )
    agent = build_agent(retriever)

    result = await agent.run("那么B组呢？", SCHEDULE_HISTORY)

    assert result["retrieval_query"] == "那么B组呢？"
    assert "context_resolve: fallback (llm api key is not configured)" in result["trace"]


@pytest.mark.asyncio
async def test_duplicate_candidates_are_deduplicated_by_chunk_id():
    retriever = RecordingRetriever(
        {
            "那么B组呢？": [chunk(0, "B组负责本周一至周五的下午班次。", 0.40)],
            "B组本周的排班情况怎么样？": [chunk(0, "B组负责本周一至周五的下午班次。", 0.90)],
        }
    )
    resolver = ScriptedResolver(
        {
            "intent": "schedule_query",
            "slots": {"group": "B组"},
            "retrieval_query": "B组本周的排班情况怎么样？",
        }
    )
    agent = build_agent(retriever, resolver=resolver)

    result = await agent.run("那么B组呢？", SCHEDULE_HISTORY)

    assert "merge: 1 unique candidates" in result["trace"]
    assert "verify: 1 candidates passed threshold" in result["trace"]
    assert result["retrieved_count"] == 1


def test_merge_keeps_both_scores_and_uses_the_higher_one():
    raw_only = chunk(0, "原始召回命中。", 0.42)
    rewrite_only = chunk(1, "改写召回命中。", 0.38)
    both_raw = chunk(2, "两路都命中。", 0.31)
    both_rewrite = chunk(2, "两路都命中。", 0.77)

    merged = merge_candidates(
        [
            {**raw_only, "raw_score": 0.42, "rewrite_score": None},
            {**both_raw, "raw_score": 0.31, "rewrite_score": None},
        ],
        [
            {**rewrite_only, "raw_score": None, "rewrite_score": 0.38},
            {**both_rewrite, "raw_score": None, "rewrite_score": 0.77},
        ],
    )

    assert [item["id"] for item in merged] == [
        "eval-schedule:2",
        "eval-schedule:0",
        "eval-schedule:1",
    ]
    both = merged[0]
    assert both["raw_score"] == 0.31
    assert both["rewrite_score"] == 0.77
    assert both["score"] == 0.77
    assert merged[1]["raw_score"] == 0.42
    assert merged[1]["rewrite_score"] is None
    assert merged[1]["score"] == 0.42
    assert merged[2]["score"] == 0.38
