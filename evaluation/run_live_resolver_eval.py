import asyncio
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.core.context_resolver import ContextResolver  # noqa: E402
from app.core.document_parser import chunk_text, parse_document  # noqa: E402
from app.core.embedding import EmbeddingService  # noqa: E402
from app.core.reranker import Reranker  # noqa: E402
from app.core.retriever import Retriever  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402

DATASET_PATH = BASE_DIR / "dataset.json"
DOCUMENT_DIR = BASE_DIR / "documents"
TOP_K = 3


async def build_index(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    settings: Settings,
) -> None:
    """使用 evaluation 数据集重新构建评测索引。"""

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


def print_results(results: list[dict]) -> None:
    if not results:
        print("  No results")
        return

    for index, result in enumerate(results[:TOP_K], start=1):
        score = result.get("rerank_score", result.get("score", 0.0))
        metadata = result.get("metadata", {})

        print(
            f"  {index}. "
            f"{metadata.get('document_name')} "
            f"| chunk={metadata.get('chunk_index')} "
            f"| score={score:.4f}"
        )


async def evaluate_live_case(
    resolver: ContextResolver,
    retriever: Retriever,
    item: dict,
) -> None:
    question = item["question"]
    history = item.get("history", [])
    active_context = item.get("active_context")
    expected_document = item.get("expected_document")

    print()
    print("=" * 90)
    print(f"[{item['id']}] Live Resolver Evaluation")
    print("=" * 90)

    print(f"Current query    : {question}")
    print(f"Expected document: {expected_document}")

    if history:
        print()
        print("History:")
        for message in history:
            print(f"  {message['role']}: {message['content']}")

    print()
    print("Calling real ContextResolver...")

    try:
        resolved = await resolver.resolve(
            query=question,
            history=history,
            active_context=active_context,
        )
    except Exception as exc:
        print(f"Resolver ERROR: {exc}")
        return

    retrieval_query = resolved["retrieval_query"]

    print()
    print("Resolver Result:")
    print(json.dumps(resolved, ensure_ascii=False, indent=2))

    print()
    print(f"Retrieval Query: {retrieval_query}")

    results = await retriever.retrieve(
        query=retrieval_query,
        top_k=TOP_K,
    )

    print()
    print("Retrieval Result:")
    print_results(results)

    documents = [
        result.get("metadata", {}).get("document_name")
        for result in results
    ]

    hit = expected_document is not None and expected_document in documents

    print()
    print(
        f"Rewrite Query Hit@{TOP_K}: "
        f"{'PASS' if hit else 'FAIL'}"
    )


async def main() -> None:
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    settings = Settings(
        chroma_collection="evaluation_live_resolver_bge_v1",
        chroma_persist_dir=BASE_DIR / ".chroma",
        embedding_backend="bge",
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

    resolver = ContextResolver(settings)

    print("=" * 90)
    print("Live Context Resolver + Retrieval Evaluation")
    print("=" * 90)
    print("Resolver          : Real DeepSeek")
    print(f"Embedding backend : {settings.embedding_backend}")
    print("Rerank backend    : lexical")
    print(f"Collection        : {settings.chroma_collection}")

    print()
    print("Building evaluation index...")

    await build_index(
        vector_store=vector_store,
        embedding_service=embedding_service,
        settings=settings,
    )

    print()
    print("Starting live evaluation...")

    for item in dataset:
        if item.get("history"):
            await evaluate_live_case(
                resolver=resolver,
                retriever=retriever,
                item=item,
            )


if __name__ == "__main__":
    asyncio.run(main())