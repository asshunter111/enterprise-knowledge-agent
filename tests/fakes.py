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


class RecordingRetriever(FakeRetriever):
    """按 Query 返回预设候选，并记录实际发起的检索请求。"""

    def __init__(self, results: dict[str, list[dict]] | None = None) -> None:
        super().__init__()
        self.queries: list[str] = []
        self._results = results or {}

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        self.queries.append(query)
        if query in self._results:
            return [dict(item) for item in self._results[query]]
        return []


class ScriptedResolver:
    """按脚本返回上下文解析结果，或按脚本抛出异常。"""

    def __init__(self, result: dict | None = None, error: Exception | None = None) -> None:
        self.result = result or {}
        self.error = error
        self.calls: list[dict] = []

    async def resolve(
        self, query: str, history: list[dict], active_context: dict | None
    ) -> dict:
        self.calls.append(
            {"query": query, "history": history, "active_context": active_context}
        )
        if self.error is not None:
            raise self.error
        return dict(self.result)


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


class RecordingGenerator(FakeGenerator):
    """记录生成调用，用于验证 Direct Response 不触碰生成阶段。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, query: str, documents: list[dict], history: list[dict]) -> dict:
        self.calls.append(query)
        return await super().generate(query, documents, history)
