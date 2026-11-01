import uuid
from typing import Optional
from langchain_core.messages import HumanMessage

from app.agents.graph import agent_graph
from app.schemas.chat_schema import ChatRequest, ChatResponse, ToolTraceItem
from app.utils.logger import TraceLogger, logger


class ChatService:
    """Service layer managing user chat execution through LangGraph."""

    def __init__(self, graph=None) -> None:
        self.graph = graph or agent_graph

    def process_chat_sync(
        self, request: ChatRequest, request_id: Optional[str] = None
    ) -> ChatResponse:
        """Synchronously process a chat request through the LangGraph workflow and return structured response."""
        rid = request_id or str(uuid.uuid4())[:8]
        query = request.query.strip()

        TraceLogger.agent_event(f"User requested: \"{query}\"", rid)

        # Initialize agent state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "trace": [],
            "tools_used": [],
            "request_id": rid,
            "query": query,
            "iteration": 0,
        }

        try:
            # Execute graph
            final_state = self.graph.invoke(initial_state)

            # Extract final answer
            messages = final_state.get("messages", [])
            answer = "I'm sorry, I could not generate a response. Please try again."

            # Find the last AIMessage with text content
            for msg in reversed(messages):
                content = getattr(msg, "content", None)
                if content and isinstance(content, str) and not getattr(msg, "tool_calls", None):
                    answer = content
                    break
                elif content and isinstance(content, str) and msg.__class__.__name__ == "AIMessage":
                    answer = content

            tools_used = final_state.get("tools_used", [])
            raw_trace = final_state.get("trace", [])

            trace_items = [
                ToolTraceItem(
                    tool=item.get("tool", "unknown"),
                    reason=item.get("reason", "Tool execution"),
                    status=item.get("status", "success"),
                    arguments=item.get("arguments"),
                    duration_ms=item.get("duration_ms"),
                    summary=item.get("summary"),
                )
                for item in raw_trace
            ]

            TraceLogger.agent_event("Final response generated.", rid)

            return ChatResponse(
                answer=answer,
                tools_used=tools_used,
                trace=trace_items,
                request_id=rid,
            )

        except Exception as e:
            TraceLogger.error(f"Chat processing failed: {str(e)}", rid)
            return ChatResponse(
                answer=(
                    f"An error occurred while processing your travel query: {str(e)}. "
                    "Please verify your query and network connection."
                ),
                tools_used=[],
                trace=[],
                request_id=rid,
            )

    async def process_chat(
        self, request: ChatRequest, request_id: Optional[str] = None
    ) -> ChatResponse:
        """Asynchronous wrapper for FastAPI endpoints."""
        return self.process_chat_sync(request, request_id)


chat_service = ChatService()

