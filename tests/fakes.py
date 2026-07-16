from collections.abc import AsyncIterator

DOCUMENT = {
    "id": "doc-1:0",
    "content": "员工报销需要在费用发生后十个工作日内提交申请。",
    "metadata": {
        "document_id": "doc-1",
        "document_name": "财务制度.md",
        "chunk_index": 0,
    },
    "score": 0.82,
}


class FakeReranker:
    async def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        return [{**item, "rerank_score": 0.91} for item in documents]


class FakeRetriever:
    def __init__(self) -> None:
        self.reranker = FakeReranker()

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        return [] if "missing" in query else [DOCUMENT]


class FakeGenerator:
    async def generate(self, query: str, documents: list[dict], history: list[dict]) -> dict:
        return {
            "answer": "报销应在十个工作日内提交。[来源: 财务制度.md]",
            "citations": self.build_citations(documents),
        }

    async def stream(
        self, query: str, documents: list[dict], history: list[dict]
    ) -> AsyncIterator[str]:
        yield "报销应在"
        yield "十个工作日内提交。"

    def build_citations(self, documents: list[dict]) -> list[dict]:
        if not documents:
            return []
        return [
            {
                "index": 1,
                "document_id": "doc-1",
                "document_name": "财务制度.md",
                "chunk_index": 0,
                "content_preview": DOCUMENT["content"],
                "score": 0.91,
            }
        ]
