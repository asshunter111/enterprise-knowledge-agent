import asyncio

from app.config import get_settings
from app.core.context_resolver import ContextResolver
from app.core.retriever import get_retriever


async def main():
    settings = get_settings()

    resolver = ContextResolver(settings)
    retriever = get_retriever()

    history = [
        {
            "role": "user",
            "content": "A组和B组的排班情况怎么样？",
        },
        {
            "role": "assistant",
            "content": "A组周一上午由张三负责；B组周一下午由李四负责。",
        },
    ]

    active_context = {
        "intent": "schedule_query",
        "slots": {
            "group": "B组",
            "time": "周一下午",
            "person": "李四",
        },
    }

    query = "这个需要提前申请吗？"

    print("\n=== 1. 原始问题 ===")
    print(query)

    resolved = await resolver.resolve(
        query=query,
        history=history,
        active_context=active_context,
    )

    retrieval_query = resolved["retrieval_query"]

    print("\n=== 2. Resolver Result ===")
    print(resolved)

    print("\n=== 3. Retrieval Query ===")
    print(retrieval_query)

    candidates = await retriever.retrieve(
        retrieval_query,
        settings.retrieval_top_k,
    )

    print("\n=== 4. Retrieval Result ===")

    for index, doc in enumerate(candidates, start=1):
        print(f"\n--- Top {index} ---")
        print("score:", doc.get("score"))
        print("document_id:", doc.get("document_id"))
        print("chunk_index:", doc.get("chunk_index"))
        print("content:", doc.get("content"))


if __name__ == "__main__":
    asyncio.run(main())