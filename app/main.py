from contextlib import asynccontextmanager
import time
import uuid
from typing import List
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.rag.document_loader import KNOWN_DESTINATIONS
from app.rag.vector_store import get_vector_store
from app.schemas.chat_schema import ChatRequest, ChatResponse, HealthResponse
from app.services.chat_service import chat_service
from app.utils.logger import TraceLogger, logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing TripMate AI application...")
    logger.info(f"Configured Gemini Model: {settings.gemini_model}")
    logger.info(f"Configured Embedding Model: {settings.embedding_model}")

    # Pre-initialize vector store
    try:
        store = get_vector_store()
        store.initialize()
        logger.info(
            f"FAISS Destination Vector Store initialized with {len(store.documents)} sections."
        )
    except Exception as e:
        logger.warning(
            f"Vector store lazy initialization deferred due to startup warning: {e}"
        )

    yield

    logger.info("TripMate AI application shutting down.")


app = FastAPI(
    title="TripMate - AI Travel Assistant",
    description="Agentic AI Travel Assistant powered by LangGraph, Google Gemini, and Local FAISS RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_correlation_id_and_timing(request: Request, call_next):
    """Middleware to inject correlation Request ID and execution duration."""
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    request.state.request_id = req_id

    start_time = time.time()
    response: Response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to inspect service and vector store status."""
    store = get_vector_store()
    is_ready = store.index is not None and store.index.ntotal > 0
    loaded_cities = [dest["city"] for dest in KNOWN_DESTINATIONS.values()]

    return HealthResponse(
        status="healthy",
        gemini_model=settings.gemini_model,
        vector_store_ready=is_ready,
        destinations_loaded=loaded_cities,
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
)
async def chat_endpoint(
    request: ChatRequest,
    x_request_id: str = Header(default=None, alias="X-Request-ID"),
):
    """Primary chat endpoint for TripMate AI Travel Assistant.

    Receives natural language queries, dynamically orchestrates tools via Gemini + LangGraph,
    and returns a synthesized answer with observable execution trace.
    """
    req_id = x_request_id or str(uuid.uuid4())[:8]

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query field cannot be empty.",
        )

    response = await chat_service.process_chat(request=request, request_id=req_id)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to return clean JSON error responses."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error encountered while processing travel query.",
            "error": str(exc),
        },
    )

