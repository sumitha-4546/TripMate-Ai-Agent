from typing import Annotated, Any, Dict, List, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """LangGraph agent state containing conversational message history and observable trace."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    trace: List[Dict[str, Any]]
    tools_used: List[str]
    request_id: str
    query: str
    iteration: int

