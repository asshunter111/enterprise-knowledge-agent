import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.core.agent import EnterpriseRAGAgent, merge_candidates  # noqa: E402
from app.core.document_parser import chunk_text, parse_document  # noqa: E402
from app.core.embedding import EmbeddingService  # noqa: E402
from app.core.reranker import Reranker  # noqa: E402
from app.core.retriever import Retriever  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402

DATASET_PATH = BASE_DIR / "dataset.json"
DOCUMENT_DIR = BASE_DIR / "documents"
TOP_K = 3

# 需要恢复上下文才能检索的模式，用于统计改写成功率和回落率
REWRITE_REQUIRED_MODES = {"continue", "cross_turn", "dedup", "new_intent", "rewrite_failure"}


@dataclass
class Stats:
    answerable: int = 0
    unanswerable: int = 0

    vector_hit_at_1: int = 0
    vector_hit_at_3: int = 0
    rerank_hit_at_1: int = 0
    rerank_hit_at_3: int = 0

    verify_pass: int = 0
    verify_reject: int = 0
    correct_abstention: int = 0

    raw_hit: int = 0
    raw_evaluated: int = 0
    rewrite_hit: int = 0
    rewrite_evaluated: int = 0
    merged_hit: int = 0
    merged_evaluated: int = 0

    rewrite_required: int = 0
    rewrite_success: int = 0

    dedup_checked: int = 0
    dedup_effective: int = 0

    behaviour_passed: int = 0
    behaviour_failed: int = 0
    failures: list[str] = field(default_factory=list)

    baseline_answerable_correct: int = 0
    baseline_correct_abstention: int = 0
    baseline_false_refusal: int = 0
    baseline_false_answer: int = 0
    baseline_predicted_abstentions: int = 0
    improved_answerable_correct: int = 0
    improved_correct_abstention: int = 0
    improved_false_refusal: int = 0
    improved_false_answer: int = 0
    improved_predicted_abstentions: int = 0

    def record(self, case_id: str, message: str, passed: bool) -> None:
        if passed:
            self.behaviour_passed += 1
        else:
            self.behaviour_failed += 1
            self.failures.append(f"{case_id}: {message}")


class DatasetResolver:
    """用 dataset 标注的 rewritten_query 代替真实 LLM 解析器。

    这里评测的是检索链路如何处理改写结果，不是解析器本身的质量。
    """

    def __init__(self, item: dict) -> None:
        self.item = item

    async def resolve(self, query: str, history: list[dict], active_context: dict | None) -> dict:
        error = self.item.get("resolver_error")
        if error:
            raise RuntimeError(error)
        rewritten_query = self.item.get("rewritten_query")
        if not rewritten_query:
            raise RuntimeError("dataset does not provide rewritten_query")
        return {
            "intent": self.item.get("expected_intent") or "context_query",
            "slots": self.item.get("expected_context_slots") or {},
            "retrieval_query": rewritten_query,
        }


class EvalGenerator:
    """评测只关心检索结果，生成阶段直接回传候选文档来源。"""

    async def generate(self, query: str, documents: list[dict], history: list[dict]) -> dict:
        return {"answer": "", "citations": self.build_citations(documents)}

    def build_citations(self, documents: list[dict]) -> list[dict]:
        return [
            {
                "document_name": item["metadata"]["document_name"],
                "chunk_index": int(item["metadata"]["chunk_index"]),
            }
            for item in documents
        ]


class CountingRetriever:
    """包一层调用计数，用来验证 Direct Response 没有触碰检索。"""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.reranker = retriever.reranker
        self.calls = 0

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        self.calls += 1
        return await self.retriever.retrieve(query, top_k)


def print_results(results: list[dict], limit: int = TOP_K) -> None:
    if not results:
        print("  No results")
        return
    for index, result in enumerate(results[:limit], start=1):
        score = result.get("rerank_score", result.get("score", 0.0))
        chunk_index = result["metadata"].get("chunk_index", 0)
        print(
            f"  {index}. {result['metadata']['document_name']} "
            f"| chunk={chunk_index} "
            f"| score={score:.4f}"
        )


async def build_index(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    settings: Settings,
) -> None:
    """使用项目实际的文档解析和Chunk切分逻辑构建评测索引。"""

    for document_path in DOCUMENT_DIR.glob("*.md"):
        text = parse_document(document_path)
        chunks = chunk_text(text, settings)

        embeddings = await embedding_service.embed_documents(
            [chunk["content"] for chunk in chunks]
        )

        await vector_store.upsert(
            chunks=chunks,
            embeddings=embeddings,
            document_id=f"eval-{document_path.stem}",
            document_name=document_path.name,
        )

        print(f"Indexed: {document_path.name} ({len(chunks)} chunks)")


async def evaluate_single_turn(retriever: Retriever, item: dict, stats: Stats) -> None:
    question = item["question"]
    expected_document = item["expected_document"]

    vector_results = await retriever.retrieve(query=question, top_k=TOP_K)
    vector_documents = [result["metadata"]["document_name"] for result in vector_results]

    min_score = retriever.settings.min_relevance_score
    verified_results = [
        result for result in vector_results if result["score"] >= min_score
    ]

    if verified_results:
        stats.verify_pass += 1
        rerank_results = await retriever.reranker.rerank(question, verified_results)
        rerank_documents = [result["metadata"]["document_name"] for result in rerank_results]
    else:
        stats.verify_reject += 1
        rerank_results = []
        rerank_documents = []

    print()
    print(f"[{item['id']}] {question}")

    if expected_document is None:
        stats.unanswerable += 1
        is_correctly_rejected = not verified_results
        if is_correctly_rejected:
            stats.correct_abstention += 1

        print("Expected : None (unanswerable)")
        print("Vector:")
        print_results(vector_results)
        print(
            f"Verify : {len(verified_results)}/{len(vector_results)} candidates passed "
            f"(threshold={min_score:.4f})"
        )
        print(
            "Result : "
            f"{'PASS Correct abstention' if is_correctly_rejected else 'FAIL False positive'}"
        )
    else:
        stats.answerable += 1

    vector_hit1 = bool(vector_documents) and vector_documents[0] == expected_document
    vector_hit3 = expected_document in vector_documents[:TOP_K]
    rerank_hit1 = bool(rerank_documents) and rerank_documents[0] == expected_document
    rerank_hit3 = expected_document in rerank_documents[:TOP_K]

    stats.vector_hit_at_1 += int(vector_hit1)
    stats.vector_hit_at_3 += int(vector_hit3)
    stats.rerank_hit_at_1 += int(rerank_hit1)
    stats.rerank_hit_at_3 += int(rerank_hit3)

    if expected_document is not None:
        print(f"Expected : {expected_document}")
        print("Vector:")
        print_results(vector_results)
        print(
            f"Verify : {len(verified_results)}/{len(vector_results)} candidates passed "
            f"(threshold={min_score:.4f})"
        )
        if rerank_results:
            print("Rerank:")
            print_results(rerank_results)
        else:
            print("Rerank : skipped")
        print(
            f"Result : Vector@1={'PASS' if vector_hit1 else 'FAIL'}, "
            f"Vector@3={'PASS' if vector_hit3 else 'FAIL'}, "
            f"Rerank@1={'PASS' if rerank_hit1 else 'FAIL'}, "
            f"Rerank@3={'PASS' if rerank_hit3 else 'FAIL'}"
        )

    baseline_abstained = not verified_results
    baseline_correct = expected_document is not None and not baseline_abstained and rerank_hit3
    if expected_document is None:
        stats.baseline_correct_abstention += int(baseline_abstained)
        stats.baseline_false_answer += int(not baseline_abstained)
    else:
        stats.baseline_answerable_correct += int(baseline_correct)
        stats.baseline_false_refusal += int(baseline_abstained)
    stats.baseline_predicted_abstentions += int(baseline_abstained)

    agent = EnterpriseRAGAgent(
        settings=retriever.settings,
        retriever=retriever,
        generator=EvalGenerator(),
    )
    improved = await agent.run(question)
    improved_abstained = improved["abstained"]
    if expected_document is None:
        print(
            "Improved evidence : "
            f"{'abstain' if improved_abstained else 'answer'}; "
            f"top_rerank={improved['diagnostics'].get('evidence_top_rerank_score', 0.0):.4f}; "
            f"threshold={improved['diagnostics'].get('evidence_threshold', 0.0):.4f}"
        )
    improved_correct = (
        expected_document is not None
        and not improved_abstained
        and expected_document in {item["document_name"] for item in improved["citations"]}
    )
    if expected_document is None:
        stats.improved_correct_abstention += int(improved_abstained)
        stats.improved_false_answer += int(not improved_abstained)
    else:
        stats.improved_answerable_correct += int(improved_correct)
        stats.improved_false_refusal += int(improved_abstained)
    stats.improved_predicted_abstentions += int(improved_abstained)


async def evaluate_direct(
    settings: Settings, retriever: Retriever, item: dict, stats: Stats
) -> None:
    counted = CountingRetriever(retriever)
    agent = EnterpriseRAGAgent(settings, counted, EvalGenerator())

    result = await agent.run(item["question"], item.get("history", []))

    print()
    print("=" * 90)
    print(f"[{item['id']}] Direct Response Test")
    print("=" * 90)
    print(f"Query  : {item['question']}")
    print(f"Answer : {result['answer']}")
    print()
    print(f"Retriever calls : {counted.calls}")
    print(f"Rerank chunks   : {result['retrieved_count']}")
    print(f"Citations       : {len(result['citations'])}")
    print(f"Trace           : {' | '.join(result['trace'])}")

    stats.record(item["id"], "retriever was called", counted.calls == 0)
    stats.record(item["id"], "reranker produced chunks", result["retrieved_count"] == 0)
    stats.record(item["id"], "citations were returned", not result["citations"])
    stats.record(
        item["id"],
        "unexpected trace",
        result["trace"] == ["context_router: direct response", "direct_response: completed"],
    )


async def evaluate_multi_turn(
    settings: Settings, retriever: Retriever, item: dict, stats: Stats
) -> None:
    question = item["question"]
    history = item.get("history", [])
    rewritten_query = item.get("rewritten_query")
    expected_document = item["expected_document"]
    mode = item.get("mode", "continue")

    print()
    print("=" * 90)
    print(f"[{item['id']}] Multi-turn Test ({mode})")
    print("=" * 90)
    print(f"Current query   : {question}")
    print(f"Rewritten query : {rewritten_query or '(无，回落到原始 Query)'}")
    print()
    print("History:")
    for message in history:
        print(f"  {message['role']}: {message['content']}")

    raw_results = await retriever.retrieve(query=question, top_k=TOP_K)
    rewrite_results = (
        await retriever.retrieve(query=rewritten_query, top_k=TOP_K) if rewritten_query else []
    )

    merged_results = merge_candidates(
        [{**item_, "raw_score": item_["score"], "rewrite_score": None} for item_ in raw_results],
        [
            {**item_, "raw_score": None, "rewrite_score": item_["score"]}
            for item_ in rewrite_results
        ],
    )

    raw_documents = [result["metadata"]["document_name"] for result in raw_results]
    rewrite_documents = [result["metadata"]["document_name"] for result in rewrite_results]
    merged_documents = [result["metadata"]["document_name"] for result in merged_results]

    print()
    print("A. Raw Query Retrieval")
    print_results(raw_results)
    print()
    print("B. Rewrite Query Retrieval")
    print_results(rewrite_results)
    print()
    print("C. Merged / Dedup")
    print_results(merged_results)

    raw_hit = expected_document in raw_documents[:TOP_K]
    rewrite_hit = expected_document in rewrite_documents[:TOP_K] if rewritten_query else None
    merged_hit = expected_document in merged_documents[:TOP_K]

    if expected_document is not None:
        stats.raw_evaluated += 1
        stats.raw_hit += int(raw_hit)
        stats.merged_evaluated += 1
        stats.merged_hit += int(merged_hit)
        if rewritten_query:
            stats.rewrite_evaluated += 1
            stats.rewrite_hit += int(rewrite_hit)

    if mode in REWRITE_REQUIRED_MODES:
        stats.rewrite_required += 1
        if rewritten_query:
            stats.rewrite_success += 1

    if mode == "dedup":
        stats.dedup_checked += 1
        if len(merged_results) < len(raw_results) + len(rewrite_results):
            stats.dedup_effective += 1

    print()
    print(f"Raw Query Hit@{TOP_K}     : {'PASS' if raw_hit else 'FAIL'}")
    if rewrite_hit is None:
        print(f"Rewrite Query Hit@{TOP_K} : - (无改写)")
    else:
        print(f"Rewrite Query Hit@{TOP_K} : {'PASS' if rewrite_hit else 'FAIL'}")
    print(f"Merged Hit@{TOP_K}        : {'PASS' if merged_hit else 'FAIL'}")

    if mode == "dedup":
        unique = len(merged_results)
        total = len(raw_results) + len(rewrite_results)
        print(f"Dedup                  : {total} -> {unique} unique candidates")

    await check_agent_behaviour(settings, retriever, item, stats)


async def check_agent_behaviour(
    settings: Settings, retriever: Retriever, item: dict, stats: Stats
) -> None:
    """跑一遍真实的 Agent 图，验证路由、改写与回落行为。"""

    mode = item.get("mode", "continue")
    agent = EnterpriseRAGAgent(
        settings, retriever, EvalGenerator(), resolver=DatasetResolver(item)
    )
    result = await agent.run(
        item["question"], item.get("history", []), item.get("active_context")
    )

    print()
    print("Agent:")
    for line in result["trace"]:
        print(f"  {line}")

    question = item["question"]
    rewritten_query = item.get("rewritten_query")

    if mode == "rewrite_failure":
        stats.record(
            item["id"],
            "did not fall back to raw query",
            result["retrieval_query"] == question,
        )
        stats.record(
            item["id"],
            "fallback was not recorded in trace",
            any("fallback" in line for line in result["trace"]),
        )
        stats.record(item["id"], "agent did not return an answer", bool(result["answer"]))
        return

    if mode == "new_intent":
        active_context = result["active_context"] or {}
        stats.record(
            item["id"],
            f"intent did not switch to {item.get('expected_intent')}",
            active_context.get("intent") == item.get("expected_intent"),
        )
        stats.record(
            item["id"],
            "old slots leaked into the new intent",
            active_context.get("slots") == (item.get("expected_context_slots") or {}),
        )
        stats.record(
            item["id"],
            "new intent was not routed as new_intent",
            result["context_mode"] == "new_intent",
        )

    if rewritten_query:
        stats.record(
            item["id"],
            "retrieval query is not the resolved rewrite",
            result["retrieval_query"] == rewritten_query,
        )

    stats.record(
        item["id"],
        "merged candidates never reached the reranker",
        bool(result["citations"]) or item["expected_document"] is None,
    )


async def evaluate(settings: Settings, retriever: Retriever) -> Stats:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    stats = Stats()

    print("=" * 90)
    print("RAG Retrieval + Verify + Reranker Evaluation")
    print("=" * 90)
    print("Embedding backend : hash (离线演示用，不代表生产语义检索效果)")
    print("Rerank backend    : lexical")
    print("Production chunks : 800/150")
    print("Experiment chunks : 200/30")

    for item in dataset:
        mode = item.get("mode")
        if mode == "direct":
            await evaluate_direct(settings, retriever, item, stats)
        elif item.get("history"):
            await evaluate_multi_turn(settings, retriever, item, stats)
        else:
            await evaluate_single_turn(retriever, item, stats)

    return stats


def percent(hit: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{hit / total:.2%}"


def metric(label: str, hit: int, total: int) -> str:
    return f"{label:<20} : {hit}/{total} = {percent(hit, total)}"


def print_summary(stats: Stats) -> None:
    print()
    print("=" * 90)
    print("Evaluation Summary")
    print("=" * 90)

    print(f"Answerable Questions   : {stats.answerable}")
    print(f"Unanswerable Questions : {stats.unanswerable}")

    print()
    print("Single-turn Retrieval:")
    print(metric("Vector Hit@1", stats.vector_hit_at_1, stats.answerable))
    print(metric("Vector Hit@3", stats.vector_hit_at_3, stats.answerable))
    print(metric("Rerank Hit@1", stats.rerank_hit_at_1, stats.answerable))
    print(metric("Rerank Hit@3", stats.rerank_hit_at_3, stats.answerable))

    print()
    print("Multi-turn Retrieval:")
    print(metric(f"Raw Query Hit@{TOP_K}", stats.raw_hit, stats.raw_evaluated))
    print(metric(f"Rewrite Query Hit@{TOP_K}", stats.rewrite_hit, stats.rewrite_evaluated))
    print(metric(f"Merged Hit@{TOP_K}", stats.merged_hit, stats.merged_evaluated))

    print()
    print("Context Resolution:")
    print(metric("Rewrite Success Rate", stats.rewrite_success, stats.rewrite_required))
    fallback = stats.rewrite_required - stats.rewrite_success
    print(metric("Fallback Rate", fallback, stats.rewrite_required))
    print(metric("Dedup Effective", stats.dedup_effective, stats.dedup_checked))

    print()
    print("Verify / Abstention:")
    print(f"Questions with verified candidates : {stats.verify_pass}")
    print(f"Questions rejected by Verify       : {stats.verify_reject}")
    print(metric("Correct abstention", stats.correct_abstention, stats.unanswerable))
    print(
        metric(
            "Abstention recall",
            stats.correct_abstention,
            stats.unanswerable,
        )
    )
    predicted_abstentions = stats.verify_reject
    if predicted_abstentions:
        print(
            metric(
                "Abstention precision",
                stats.correct_abstention,
                predicted_abstentions,
            )
        )
    else:
        print("Abstention precision  : n/a (no questions were rejected)")

    print()
    print("Abstention Decision Comparison (single-turn, n=12):")
    print("Baseline:")
    print(metric("Answerable correct", stats.baseline_answerable_correct, stats.answerable))
    print(metric("Correct abstention", stats.baseline_correct_abstention, stats.unanswerable))
    print(metric("False refusal", stats.baseline_false_refusal, stats.answerable))
    print(metric("False answer", stats.baseline_false_answer, stats.unanswerable))
    print(
        metric(
            "Abstention precision",
            stats.baseline_correct_abstention,
            stats.baseline_predicted_abstentions,
        )
    )
    print(metric("Abstention recall", stats.baseline_correct_abstention, stats.unanswerable))
    print("Improved:")
    print(metric("Answerable correct", stats.improved_answerable_correct, stats.answerable))
    print(metric("Correct abstention", stats.improved_correct_abstention, stats.unanswerable))
    print(metric("False refusal", stats.improved_false_refusal, stats.answerable))
    print(metric("False answer", stats.improved_false_answer, stats.unanswerable))
    print(
        metric(
            "Abstention precision",
            stats.improved_correct_abstention,
            stats.improved_predicted_abstentions,
        )
    )
    print(metric("Abstention recall", stats.improved_correct_abstention, stats.unanswerable))

    print()
    print("Agent Behaviour:")
    total = stats.behaviour_passed + stats.behaviour_failed
    print(f"Passed : {stats.behaviour_passed}/{total}")
    print(f"Failed : {stats.behaviour_failed}/{total}")

    if stats.failures:
        print()
        print("Failed Cases:")
        for failure in stats.failures:
            print(f"  FAIL {failure}")
    else:
        print()
        print("Failed Cases: none")

    print()
    print("Note: hash embedding 与 lexical reranker 只用于离线演示，")
    print("      以上指标反映检索链路行为，不代表生产语义模型性能。")


async def main() -> None:
    settings = Settings(
        chroma_collection="evaluation_hash_experiment_chunk200",
        chroma_persist_dir=BASE_DIR / ".chroma",
        embedding_backend="hash",
        rerank_backend="lexical",
        retrieval_top_k=TOP_K,
        chunk_size=200,
        chunk_overlap=30,
    )

    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings)
    reranker = Reranker(settings)

    retriever = Retriever(
        settings=settings,
        embedding_service=embedding_service,
        vector_store=vector_store,
        reranker=reranker,
    )

    await build_index(
        vector_store=vector_store,
        embedding_service=embedding_service,
        settings=settings,
    )

    stats = await evaluate(settings, retriever)
    print_summary(stats)


if __name__ == "__main__":
    asyncio.run(main())
