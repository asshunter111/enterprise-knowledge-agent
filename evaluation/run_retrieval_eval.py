import asyncio
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.core.embedding import EmbeddingService
from app.core.document_parser import parse_document, chunk_text
from app.core.reranker import Reranker
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore


DATASET_PATH = BASE_DIR / "dataset.json"
DOCUMENT_DIR = BASE_DIR / "documents"


async def build_index(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
) -> None:
    """使用项目实际的文档解析和Chunk切分逻辑构建评测索引。"""

    for document_path in DOCUMENT_DIR.glob("*.md"):
        text = parse_document(document_path)
        chunks = chunk_text(text)

        embeddings = await embedding_service.embed_documents(
            [chunk["content"] for chunk in chunks]
        )

        await vector_store.upsert(
            chunks=chunks,
            embeddings=embeddings,
            document_id=f"eval-{document_path.stem}",
            document_name=document_path.name,
        )

        print(
            f"Indexed: {document_path.name} "
            f"({len(chunks)} chunks)"
        )


async def evaluate(retriever):
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    total = len(dataset)
    answerable_total = sum(
        1 for item in dataset
        if item["expected_document"] is not None
    )
    unanswerable_total = total - answerable_total

    vector_hit_at_1 = 0
    vector_hit_at_3 = 0
    rerank_hit_at_1 = 0
    rerank_hit_at_3 = 0

    verify_pass_total = 0
    verify_reject_total = 0
    correct_abstention = 0

    print("=" * 90)
    print("RAG Retrieval + Verify + Reranker Evaluation")
    print("=" * 90)

    for item in dataset:
        question = item["question"]
        expected_document = item["expected_document"]

        # ==================================================
        # Multi-turn Retrieval Experiment
        # ==================================================
        history = item.get("history", [])
        rewritten_query = item.get("rewritten_query")

        if history and rewritten_query:
            print()
            print("=" * 90)
            print(f"[{item['id']}] Multi-turn Retrieval Test")
            print("=" * 90)

            print(f"Current query   : {question}")
            print(f"Rewritten query : {rewritten_query}")

            print()
            print("History:")
            for message in history:
                print(
                    f"  {message['role']}: "
                    f"{message['content']}"
                )

            # ----------------------------------------------
            # A. Raw Query Retrieval
            # ----------------------------------------------
            raw_results = await retriever.retrieve(
                query=question,
                top_k=3,
            )

            # ----------------------------------------------
            # B. Rewritten Query Retrieval
            # ----------------------------------------------
            rewritten_results = await retriever.retrieve(
                query=rewritten_query,
                top_k=3,
            )

            print()
            print("A. Raw Query Retrieval")

            if raw_results:
                for index, result in enumerate(
                    raw_results,
                    start=1,
                ):
                    print(
                        f"  {index}. "
                        f"{result['metadata']['document_name']} "
                        f"| chunk="
                        f"{result['metadata']['chunk_index']} "
                        f"| score="
                        f"{result['score']:.4f}"
                    )
            else:
                print("  No results")

            print()
            print("B. Rewritten Query Retrieval")

            if rewritten_results:
                for index, result in enumerate(
                    rewritten_results,
                    start=1,
                ):
                    print(
                        f"  {index}. "
                        f"{result['metadata']['document_name']} "
                        f"| chunk="
                        f"{result['metadata']['chunk_index']} "
                        f"| score="
                        f"{result['score']:.4f}"
                    )
            else:
                print("  No results")

            # ----------------------------------------------
            # Compare Hit@3
            # ----------------------------------------------
            raw_documents = [
                result["metadata"]["document_name"]
                for result in raw_results
            ]

            rewritten_documents = [
                result["metadata"]["document_name"]
                for result in rewritten_results
            ]

            raw_hit = expected_document in raw_documents[:3]
            rewritten_hit = (
                expected_document
                in rewritten_documents[:3]
            )

            print()
            print(
                f"Raw Query Hit@3       : "
                f"{'✓' if raw_hit else '✗'}"
            )

            print(
                f"Rewritten Query Hit@3 : "
                f"{'✓' if rewritten_hit else '✗'}"
            )

            # 这个测试已经完成，不再进入下面的普通评测流程
            continue

        # ==================================================
        # Normal Single-turn RAG Evaluation
        # ==================================================

        # --------------------------------------------------
        # 1. Vector Retrieval
        # --------------------------------------------------
        vector_results = await retriever.retrieve(
            query=question,
            top_k=3,
        )

        vector_documents = [
            result["metadata"]["document_name"]
            for result in vector_results
        ]

        # --------------------------------------------------
        # 2. Verify
        # --------------------------------------------------
        min_score = retriever.settings.min_relevance_score

        verified_results = [
            result
            for result in vector_results
            if result["score"] >= min_score
        ]

        verify_pass_count = len(verified_results)
        verify_reject_count = (
            len(vector_results)
            - verify_pass_count
        )

        if verify_pass_count > 0:
            verify_pass_total += 1
        else:
            verify_reject_total += 1

        # --------------------------------------------------
        # 3. Reranker
        # --------------------------------------------------
        if verified_results:
            rerank_results = await retriever.reranker.rerank(
                question,
                verified_results,
            )

            rerank_documents = [
                result["metadata"]["document_name"]
                for result in rerank_results
            ]
        else:
            rerank_results = []
            rerank_documents = []

        # --------------------------------------------------
        # 4. Answerable Question
        # --------------------------------------------------
        if expected_document is not None:

            vector_hit1 = (
                len(vector_documents) >= 1
                and vector_documents[0]
                == expected_document
            )

            vector_hit3 = (
                expected_document
                in vector_documents[:3]
            )

            rerank_hit1 = (
                len(rerank_documents) >= 1
                and rerank_documents[0]
                == expected_document
            )

            rerank_hit3 = (
                expected_document
                in rerank_documents[:3]
            )

            if vector_hit1:
                vector_hit_at_1 += 1

            if vector_hit3:
                vector_hit_at_3 += 1

            if rerank_hit1:
                rerank_hit_at_1 += 1

            if rerank_hit3:
                rerank_hit_at_3 += 1

            print()
            print(f"[{item['id']}] {question}")
            print(f"Expected : {expected_document}")

            print("Vector:")

            for index, result in enumerate(
                vector_results,
                start=1,
            ):
                print(
                    f"  {index}. "
                    f"{result['metadata']['document_name']} "
                    f"| score={result['score']:.4f}"
                )

            print(
                f"Verify : {verify_pass_count}/"
                f"{len(vector_results)} "
                f"candidates passed "
                f"(threshold={min_score:.4f})"
            )

            if rerank_results:
                print("Rerank:")

                for index, result in enumerate(
                    rerank_results,
                    start=1,
                ):
                    print(
                        f"  {index}. "
                        f"{result['metadata']['document_name']} "
                        f"| vector_score="
                        f"{result['score']:.4f} "
                        f"| rerank_score="
                        f"{result.get('rerank_score', 0.0):.4f}"
                    )
            else:
                print("Rerank : skipped")

            print(
                f"Result : "
                f"Vector@1="
                f"{'✓' if vector_hit1 else '✗'}, "
                f"Vector@3="
                f"{'✓' if vector_hit3 else '✗'}, "
                f"Rerank@1="
                f"{'✓' if rerank_hit1 else '✗'}, "
                f"Rerank@3="
                f"{'✓' if rerank_hit3 else '✗'}"
            )

        # --------------------------------------------------
        # 5. Unanswerable Question
        # --------------------------------------------------
        else:
            is_correctly_rejected = (
                verify_pass_count == 0
            )

            if is_correctly_rejected:
                correct_abstention += 1

            print()
            print(f"[{item['id']}] {question}")
            print("Expected : None (unanswerable)")

            print("Vector:")

            for index, result in enumerate(
                vector_results,
                start=1,
            ):
                print(
                    f"  {index}. "
                    f"{result['metadata']['document_name']} "
                    f"| score={result['score']:.4f}"
                )

            print(
                f"Verify : {verify_pass_count}/"
                f"{len(vector_results)} "
                f"candidates passed "
                f"(threshold={min_score:.4f})"
            )

            if rerank_results:
                print("Rerank : SHOULD NOT BE USED")
                print(
                    f"Rerank candidates: "
                    f"{len(rerank_results)}"
                )
            else:
                print("Rerank : skipped")

            print(
                "Result : "
                f"{'✓ Correct abstention' if is_correctly_rejected else '✗ False positive'}"
            )

    # ======================================================
    # Final Statistics
    # ======================================================

    print()
    print("=" * 90)
    print("Evaluation Summary")
    print("=" * 90)

    print(
        f"Answerable Questions   : "
        f"{answerable_total}"
    )

    print(
        f"Unanswerable Questions : "
        f"{unanswerable_total}"
    )

    print()
    print("Retrieval:")

    print(
        f"Vector Hit@1 : "
        f"{vector_hit_at_1}/{answerable_total} "
        f"= {vector_hit_at_1 / answerable_total:.2%}"
    )

    print(
        f"Vector Hit@3 : "
        f"{vector_hit_at_3}/{answerable_total} "
        f"= {vector_hit_at_3 / answerable_total:.2%}"
    )

    print()
    print("Reranker:")

    print(
        f"Rerank Hit@1 : "
        f"{rerank_hit_at_1}/{answerable_total} "
        f"= {rerank_hit_at_1 / answerable_total:.2%}"
    )

    print(
        f"Rerank Hit@3 : "
        f"{rerank_hit_at_3}/{answerable_total} "
        f"= {rerank_hit_at_3 / answerable_total:.2%}"
    )

    print()
    print("Verify / Abstention:")

    print(
        f"Questions with verified candidates : "
        f"{verify_pass_total}/{total}"
    )

    print(
        f"Questions rejected by Verify        : "
        f"{verify_reject_total}/{total}"
    )

    if unanswerable_total > 0:
        print(
            f"Correct abstention                 : "
            f"{correct_abstention}/"
            f"{unanswerable_total} "
            f"= "
            f"{correct_abstention / unanswerable_total:.2%}"
        )

async def main() -> None:
    settings = Settings(
        chroma_collection="rag_evaluation_v2",
        embedding_backend="hash",
        rerank_backend="lexical",
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
    )

    await evaluate(retriever)


if __name__ == "__main__":
    asyncio.run(main())