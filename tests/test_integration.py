import asyncio
from unittest.mock import MagicMock, patch
import pytest
from httpx import Response
from langchain_core.messages import AIMessage

from app.agents.graph import create_agent_graph
from app.schemas.chat_schema import ChatRequest
from app.services.chat_service import ChatService


def test_end_to_end_packing_flow_with_multi_tool_reasoning():
    
    # 1. Mock LLM sequence:
    # First invocation: calls both tools
    # Second invocation (after receiving tool messages): synthesizes final response
    mock_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_destination_guide",
                    "args": {"query": "Tokyo winter packing tips and local customs"},
                    "id": "integration_call_rag_01",
                },
                {
                    "name": "get_weather_forecast",
                    "args": {"city": "Tokyo", "date_or_month": "December"},
                    "id": "integration_call_weather_02",
                },
            ],
        ),
        AIMessage(
            content=(
                "### Packing Recommendations for Tokyo in December\n\n"
                "**Weather Overview:**\n"
                "- Expected conditions: Cold, dry, and sunny with temperatures typically ranging between 4°C and 12°C.\n\n"
                "**What to Pack:**\n"
                "- **Layered Clothing:** Warm sweaters, thermal base layers, and an insulated winter coat.\n"
                "- **Comfortable Walking Shoes:** Essential for exploring Tokyo and navigating transit stations.\n"
                "- **Indoor Layering:** Warm socks and easy-to-remove shoes for temples and traditional accommodations.\n\n"
                "*Note: Weather data is based on seasonal climate estimates.*"
            ),
        ),
    ]

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.invoke.side_effect = mock_responses
    mock_llm.bind_tools.return_value = mock_bound

    # 2. Mock weather network call just in case live API is invoked
    mock_weather_api = MagicMock()
    mock_weather_api.status_code = 200
    mock_weather_api.json.return_value = {
        "daily": {
            "time": ["2026-12-01"],
            "temperature_2m_max": [12.0],
            "temperature_2m_min": [4.0],
            "precipitation_sum": [0.0],
            "weather_code": [0],
        }
    }

    with patch("app.agents.graph.get_llm", return_value=mock_llm), \
         patch("httpx.Client.get", return_value=mock_weather_api):

        # Compile fresh graph instance
        test_graph = create_agent_graph()
        service = ChatService(graph=test_graph)

        req = ChatRequest(query="I am visiting Tokyo in December. What should I pack?")
        response = service.process_chat_sync(req, request_id="INTEG-TEST-001")

        # 3. Assertions
        assert response is not None
        assert response.request_id == "INTEG-TEST-001"

        # Check tools used
        assert "search_destination_guide" in response.tools_used
        assert "get_weather_forecast" in response.tools_used
        assert len(response.tools_used) == 2

        # Check execution trace
        assert len(response.trace) == 2
        rag_trace = next((t for t in response.trace if t.tool == "search_destination_guide"), None)
        weather_trace = next((t for t in response.trace if t.tool == "get_weather_forecast"), None)

        assert rag_trace is not None
        assert rag_trace.status == "success"
        assert rag_trace.arguments == {"query": "Tokyo winter packing tips and local customs"}

        assert weather_trace is not None
        assert weather_trace.status == "success"
        assert weather_trace.arguments == {"city": "Tokyo", "date_or_month": "December"}

        # Check final synthesized answer
        assert "Tokyo in December" in response.answer
        assert "4°C" in response.answer or "Layered" in response.answer
        assert "Walking Shoes" in response.answer or "Thermal" in response.answer
