import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.chat import router as chat_router
from app.api.documents import router as document_router
from app.api.retrieval import router as retrieval_router
from app.config import get_settings
from app.models.database import AsyncSessionLocal, engine, init_db

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    description="基于 LangGraph 的企业文档检索增强问答服务",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.include_router(document_router)
app.include_router(chat_router)
app.include_router(retrieval_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled request error", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def readiness() -> dict[str, str]:
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ready"}
