from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.core.agent import EnterpriseRAGAgent
from app.core.permissions import filter_documents
from app.dependencies import get_rag_agent, verify_api_key
from app.schemas import RetrievalRequest, RetrievalResult

router = APIRouter(
    prefix="/api/retrieval", tags=["retrieval"], dependencies=[Depends(verify_api_key)]
)
AgentDep = Annotated[EnterpriseRAGAgent, Depends(get_rag_agent)]


@router.post("/search", response_model=list[RetrievalResult])
async def search(
    body: RetrievalRequest,
    agent: AgentDep,
    x_role: str = Header(default="employee"),
) -> list[dict]:
    documents = await agent.retriever.retrieve(body.query, top_k=body.top_k)
    documents = filter_documents(documents, x_role)
    return [
        {
            "content": item["content"],
            "score": item.get("score", 0.0),
            "metadata": item.get("metadata", {}),
        }
        for item in documents
    ]
