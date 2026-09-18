import asyncio

from app.config import get_settings
from app.core.context_resolver import ContextResolver


async def main():
    resolver = ContextResolver(get_settings())

    result = await resolver.resolve(
        query="第二条的对象是谁？",
        history=[
            {
                "role": "user",
                "content": "A组和B组的排班情况怎么样？",
            },
            {
                "role": "assistant",
                "content": (
                    "A组周一上午由张三负责；"
                    "B组周一下午由李四负责。"
                ),
            },
        ],
        active_context={
            "intent": "schedule_query",
            "slots": {
                "group": "A组",
            },
        },
    )

    print("\n=== Resolver Result ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())