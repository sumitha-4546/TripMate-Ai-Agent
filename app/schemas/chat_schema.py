from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming user chat query."""

    query: str = Field(
        ...,
        description="Natural language user query about travel, visa, weather, or packing",
        min_length=1,
        examples=["What should I pack for Tokyo in December?"],
    )


class ToolTraceItem(BaseModel):
    """Observable tool execution trace record."""

    tool: str = Field(..., description="Name of the executed tool")
    reason: str = Field(
        ..., description="Short explanation for why the tool was selected or called"
    )
    status: str = Field(
        ..., description="Execution status: 'success', 'error', or 'fallback'"
    )
    arguments: Optional[Dict[str, Any]] = Field(
        default=None, description="Arguments passed to the tool"
    )
    duration_ms: Optional[float] = Field(
        default=None, description="Tool execution duration in milliseconds"
    )
    summary: Optional[str] = Field(
        default=None, description="Brief summary of tool results"
    )


class ChatResponse(BaseModel):
    """Chat API response schema."""

    answer: str = Field(..., description="Synthesized natural language response")
    tools_used: List[str] = Field(
        default_factory=list,
        description="List of tool names called during the execution",
    )
    trace: List[ToolTraceItem] = Field(
        default_factory=list,
        description="Structured execution trace of tool decisions and calls",
    )
    request_id: Optional[str] = Field(
        default=None, description="Unique correlation ID for tracing the request"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service health status, e.g., 'healthy'")
    gemini_model: str = Field(..., description="Configured Gemini model")
    vector_store_ready: bool = Field(
        ..., description="Whether the FAISS destination RAG vector store is loaded"
    )
    destinations_loaded: List[str] = Field(
        default_factory=list, description="List of loaded destination cities"
    )

