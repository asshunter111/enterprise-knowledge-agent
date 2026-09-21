import pytest

from app.config import Settings
from app.core.agent import EnterpriseRAGAgent
from tests.fakes import RecordingGenerator, RecordingRetriever


class ScoredReranker:
    def __init__(self, score: float):
        self.score = score

    async def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        return [{**item, "rerank_score": self.score} for item in documents]


class ScoredRetriever(RecordingRetriever):
    def __init__(self, score: float, results: dict[str, list[dict]] | None = None):
        super().__init__(results)
        self.reranker = ScoredReranker(score)


class RewriteResolver:
    async def resolve(self, query: str, history: list[dict], active_context: dict | None) -> dict:
        return {
            "intent": "schedule_query",
            "slots": {"group": "B组"},
            "retrieval_query": "B组排班情况",
        }


def document(content: str = "员工报销需要在费用发生后十个工作日内提交申请。") -> dict:
    return {
        "id": "doc:0",
        "content": content,
        "metadata": {"document_id": "doc", "document_name": "policy.md", "chunk_index": 0},
        "score": 0.8,
    }


def build_agent(retriever, generator=None) -> tuple[EnterpriseRAGAgent, RecordingGenerator]:
    generator = generator or RecordingGenerator()
    return (
        EnterpriseRAGAgent(
            Settings(min_relevance_score=0.05, evidence_min_rerank_score=0.20),
            retriever,
            generator,
        ),
        generator,
    )


@pytest.mark.asyncio
async def test_clear_relevant_evidence_reaches_generator():
    agent, generator = build_agent(ScoredRetriever(0.9, {"报销期限是多少": [document()]}))

    result = await agent.run("报销期限是多少")

    assert result["abstained"] is False
    assert result["retrieved_count"] == 1
    assert generator.calls == ["报销期限是多少"]


@pytest.mark.asyncio
async def test_no_documents_abstains_without_generator():
    agent, generator = build_agent(ScoredRetriever(0.9, {"未知问题": []}))

    result = await agent.run("未知问题")

    assert result["abstained"] is True
    assert result["answer"] == "资料中未找到相关信息。"
    assert generator.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [0.05, 0.19])
async def test_low_relevance_evidence_abstains_without_generator(score: float):
    agent, generator = build_agent(ScoredRetriever(score, {"收入是多少": [document()]}))

    result = await agent.run("收入是多少")

    assert result["abstained"] is True
    assert result["answer"] == "资料中未找到相关信息。"
    assert generator.calls == []
    assert result["diagnostics"]["evidence_top_rerank_score"] == score


@pytest.mark.asyncio
async def test_borderline_score_is_rejected_at_documented_boundary():
    agent, generator = build_agent(ScoredRetriever(0.20, {"报销期限是多少": [document()]}))

    result = await agent.run("报销期限是多少")

    assert result["abstained"] is False
    assert generator.calls == ["报销期限是多少"]


@pytest.mark.asyncio
async def test_rerank_evidence_can_rescue_a_low_vector_score():
    item = {**document(), "score": 0.06}
    agent, generator = build_agent(ScoredRetriever(0.4, {"报销期限是多少": [item]}))

    result = await agent.run("报销期限是多少")

    assert result["abstained"] is False
    assert generator.calls == ["报销期限是多少"]


@pytest.mark.asyncio
async def test_rewritten_query_without_evidence_abstains():
    retriever = ScoredRetriever(
        0.9,
        {"那么B组呢？": [], "B组排班情况": []},
    )
    agent, generator = build_agent(retriever)
    agent.resolver = RewriteResolver()

    result = await agent.run(
        "那么B组呢？",
        history=[{"role": "user", "content": "A组排班怎么样？"}],
    )

    assert result["abstained"] is True
    assert generator.calls == []
