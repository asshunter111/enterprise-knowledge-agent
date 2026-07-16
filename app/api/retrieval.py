from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.agent import EnterpriseRAGAgent
from app.dependencies import get_rag_agent, verify_api_key
from app.schemas import RetrievalRequest, RetrievalResult

router = APIRouter(
    prefix="/api/retrieval", tags=["retrieval"], dependencies=[Depends(verify_api_key)]
)
AgentDep = Annotated[EnterpriseRAGAgent, Depends(get_rag_agent)]


@router.post("/search", response_model=list[RetrievalResult])
async def search(body: RetrievalRequest, agent: AgentDep) -> list[dict]:
    documents = await agent.retriever.retrieve(body.query, top_k=body.top_k)
    return [
        {
            "content": item["content"],
            "score": item.get("score", 0.0),
            "metadata": item.get("metadata", {}),
        }
        for item in documents
    ]
