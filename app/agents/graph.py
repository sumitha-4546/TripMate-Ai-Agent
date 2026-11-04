import json
import time
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.agents.prompts import SYSTEM_PROMPT
from app.agents.state import AgentState
from app.config import settings
from app.tools.destination_tool import search_destination_guide
from app.tools.weather_tool import get_weather_forecast
from app.utils.logger import TraceLogger, logger

MAX_ITERATIONS = 5

# Register available tools
TOOLS = [search_destination_guide, get_weather_forecast]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def get_llm(model_name: Optional[str] = None):
    model = model_name or settings.gemini_model
    api_key = settings.gemini_api_key

    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. Gemini API calls will fail unless mocked.")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key or "DUMMY_KEY",
        temperature=0.2,
        max_retries=2,
        timeout=60.0,
    )


def agent_node(state: AgentState) -> Dict[str, Any]:
    messages = list(state.get("messages", []))
    request_id = state.get("request_id", "")
    iteration = state.get("iteration", 0) + 1

    # Ensure system prompt is present at the beginning
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    TraceLogger.agent_event(f"Invoking Gemini model ({settings.gemini_model})...", request_id)

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        error_str = str(e)
        TraceLogger.error(f"Gemini API invocation failed: {error_str}", request_id)

        # Fallback response if quota or network fails
        if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            fallback_msg = (
                "TripMate is currently experiencing high demand (Gemini API quota reached). "
                "Please try again shortly or check your Gemini API plan."
            )
        else:
            fallback_msg = (
                f"TripMate encountered an issue communicating with the AI model: {error_str}. "
                "Please ensure your GEMINI_API_KEY is valid and has sufficient quota."
            )
        response = AIMessage(content=fallback_msg)

    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        tool_names = [tc.get("name", "tool") for tc in tool_calls]
        TraceLogger.agent_event(
            f"Gemini selected {len(tool_calls)} tool(s): {', '.join(tool_names)}",
            request_id,
        )
    else:
        TraceLogger.agent_event("Gemini generated direct response without tools.", request_id)

    return {
        "messages": [response],
        "iteration": iteration,
    }


def execute_tools_node(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    request_id = state.get("request_id", "")
    existing_trace = list(state.get("trace", []))
    existing_tools = list(state.get("tools_used", []))

    tool_calls = getattr(last_message, "tool_calls", []) or []
    tool_messages: List[ToolMessage] = []

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_call_id = tc.get("id", f"call_{int(time.time()*1000)}")

        tool_fn = TOOLS_BY_NAME.get(tool_name)
        if not tool_fn:
            TraceLogger.error(f"Requested tool '{tool_name}' not registered.", request_id)
            err_msg = f"Error: Tool '{tool_name}' is not recognized."
            tool_messages.append(ToolMessage(content=err_msg, tool_call_id=tool_call_id))
            existing_trace.append({
                "tool": tool_name,
                "reason": "Unrecognized tool requested by model",
                "status": "error",
                "arguments": tool_args,
                "summary": err_msg,
            })
            continue

        if tool_name not in existing_tools:
            existing_tools.append(tool_name)

        # Generate a descriptive reason for the trace
        reason = "Execute tool to assist user query"
        if tool_name == "search_destination_guide":
            reason = f"Retrieve destination guide information for '{tool_args.get('query', '')}'"
        elif tool_name == "get_weather_forecast":
            reason = f"Retrieve weather information for {tool_args.get('city', '')}"

        t_start = time.time()
        try:
            # Execute tool
            output = tool_fn.invoke(tool_args)
            duration_ms = (time.time() - t_start) * 1000

            # Generate summary for trace
            if isinstance(output, list):
                summary = f"Retrieved {len(output)} relevant section(s)"
                content_str = "\n\n".join(output)
            elif isinstance(output, dict):
                cond = output.get("conditions", "Normal")
                temps = output.get("temp_range_c", [])
                summary = f"{cond}, Temp: {temps}"
                content_str = json.dumps(output)
            else:
                summary = str(output)[:100]
                content_str = str(output)

            tool_messages.append(ToolMessage(content=content_str, tool_call_id=tool_call_id))
            existing_trace.append({
                "tool": tool_name,
                "reason": reason,
                "status": "success",
                "arguments": tool_args,
                "duration_ms": round(duration_ms, 2),
                "summary": summary,
            })

        except Exception as e:
            duration_ms = (time.time() - t_start) * 1000
            error_str = f"Tool execution failed: {str(e)}"
            TraceLogger.error(error_str, request_id)
            tool_messages.append(ToolMessage(content=error_str, tool_call_id=tool_call_id))
            existing_trace.append({
                "tool": tool_name,
                "reason": reason,
                "status": "error",
                "arguments": tool_args,
                "duration_ms": round(duration_ms, 2),
                "summary": error_str,
            })

    TraceLogger.agent_event("Tool results passed to Gemini.", request_id)

    return {
        "messages": tool_messages,
        "trace": existing_trace,
        "tools_used": existing_tools,
    }


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    iteration = state.get("iteration", 0)

    # Loop guard safeguard against runaway execution
    if iteration >= MAX_ITERATIONS:
        TraceLogger.agent_event(f"Loop guard triggered after {iteration} iterations. Ending cycle.")
        return END

    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls and len(tool_calls) > 0:
        return "tools"

    return END


def create_agent_graph():
    workflow = StateGraph(AgentState)

    # Add agent decision node and tool execution node
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", execute_tools_node)

    # Entry point
    workflow.set_entry_point("agent")

    # Conditional edge from agent: route to tools if tool_calls present, else end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # From tools, loop back to agent for final synthesis
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Global agent graph instance
agent_graph = create_agent_graph()

