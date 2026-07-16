"""企业知识库智能问答 Agent — FastAPI 入口"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.database import init_db
from app.api.documents import router as doc_router
from app.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化数据库"""
    await init_db()
    yield


app = FastAPI(
    title="企业知识库智能问答 Agent",
    description="基于 RAG + LangGraph 的企业文档智能问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(doc_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "service": "企业知识库智能问答 Agent",
        "version": "1.0.0",
        "docs": "/docs",
    }
