import asyncio
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from app.agents.graph import create_agent_graph
from app.services.chat_service import ChatService
from app.schemas.chat_schema import ChatRequest


def build_mock_llm(responses_sequence):
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.invoke.side_effect = responses_sequence
    mock_llm.bind_tools.return_value = mock_bound
    return mock_llm


def test_visa_question_routes_to_rag_tool():
    mock_responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "search_destination_guide",
                "args": {"query": "visa requirements Japan tourist stays"},
                "id": "call_visa_001",
            }],
        ),
        AIMessage(
            content="Most tourists can visit Japan visa-free for up to 90 days with a valid passport.",
        ),
    ]

    mock_llm = build_mock_llm(mock_responses)

    with patch("app.agents.graph.get_llm", return_value=mock_llm):
        graph = create_agent_graph()
        service = ChatService(graph=graph)

        req = ChatRequest(query="What are the visa requirements for Japan?")
        res = service.process_chat_sync(req)

        assert "search_destination_guide" in res.tools_used
        assert "get_weather_forecast" not in res.tools_used
        assert len(res.trace) == 1
        assert res.trace[0].tool == "search_destination_guide"
        assert res.trace[0].status == "success"
        assert "visa-free" in res.answer


def test_weather_question_routes_to_weather_tool():
    mock_responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_weather_forecast",
                "args": {"city": "Tokyo", "date_or_month": "today"},
                "id": "call_weather_001",
            }],
        ),
        AIMessage(
            content="The weather in Tokyo today is cool and clear with temperatures around 12°C.",
        ),
    ]

    mock_llm = build_mock_llm(mock_responses)

    with patch("app.agents.graph.get_llm", return_value=mock_llm):
        graph = create_agent_graph()
        service = ChatService(graph=graph)

        req = ChatRequest(query="What is the weather in Tokyo?")
        res = service.process_chat_sync(req)

        assert "get_weather_forecast" in res.tools_used
        assert "search_destination_guide" not in res.tools_used
        assert len(res.trace) == 1
        assert res.trace[0].tool == "get_weather_forecast"
        assert res.trace[0].status == "success"
        assert "Tokyo" in res.answer


def test_packing_question_routes_to_both_tools():
    mock_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_destination_guide",
                    "args": {"query": "Tokyo packing tips and recommendations"},
                    "id": "call_pack_rag",
                },
                {
                    "name": "get_weather_forecast",
                    "args": {"city": "Tokyo", "date_or_month": "December"},
                    "id": "call_pack_weather",
                },
            ],
        ),
        AIMessage(
            content=(
                "For Tokyo in December, temperatures range from 4°C to 12°C with dry conditions. "
                "Pack warm layered clothing, thermal wear, and comfortable walking shoes."
            ),
        ),
    ]

    mock_llm = build_mock_llm(mock_responses)

    with patch("app.agents.graph.get_llm", return_value=mock_llm):
        graph = create_agent_graph()
        service = ChatService(graph=graph)

        req = ChatRequest(query="I am visiting Tokyo in December. What should I pack?")
        res = service.process_chat_sync(req)

        assert "search_destination_guide" in res.tools_used
        assert "get_weather_forecast" in res.tools_used
        assert len(res.trace) == 2
        trace_tools = [t.tool for t in res.trace]
        assert "search_destination_guide" in trace_tools
        assert "get_weather_forecast" in trace_tools
        assert "layered clothing" in res.answer or "December" in res.answer


def test_greeting_invokes_no_tools():
    mock_responses = [
        AIMessage(
            content="Hello! I am TripMate, your AI travel assistant. How can I help you plan your next trip?",
        )
    ]

    mock_llm = build_mock_llm(mock_responses)

    with patch("app.agents.graph.get_llm", return_value=mock_llm):
        graph = create_agent_graph()
        service = ChatService(graph=graph)

        req = ChatRequest(query="Hello")
        res = service.process_chat_sync(req)

        assert res.tools_used == []
        assert res.trace == []
        assert "TripMate" in res.answer


def test_unsupported_flight_booking_invokes_no_tools():
    mock_responses = [
        AIMessage(
            content="I am sorry, but I cannot book flights. TripMate provides travel guides, visa info, weather, and packing advice.",
        )
    ]

    mock_llm = build_mock_llm(mock_responses)

    with patch("app.agents.graph.get_llm", return_value=mock_llm):
        graph = create_agent_graph()
        service = ChatService(graph=graph)

        req = ChatRequest(query="Book a flight to Japan")
        res = service.process_chat_sync(req)

        assert res.tools_used == []
        assert res.trace == []
        assert "cannot book flights" in res.answer.lower()
