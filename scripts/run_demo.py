
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.rag.vector_store import get_vector_store
from app.schemas.chat_schema import ChatRequest
from app.services.chat_service import chat_service

SAMPLE_QUERIES = [
    {
        "category": "A. Visa & Entry (RAG Tool Expected)",
        "query": "What are the visa requirements for Japan?",
    },
    {
        "category": "B. Weather Forecast (Weather Tool Expected)",
        "query": "What is the weather in Tokyo?",
    },
    {
        "category": "C. Multi-Tool Reasoning: Packing Advice (Both Tools Expected)",
        "query": "I am visiting Tokyo in December. What should I pack?",
    },
    {
        "category": "D. Direct Greeting (No Tools Expected)",
        "query": "Hello! What can you help me with?",
    },
    {
        "category": "E. Out-of-Scope Capability (Polite Refusal, No Tools)",
        "query": "Can you book a flight and hotel for my trip to Tokyo?",
    },
    {
        "category": "F. Unsupported Destination Handling",
        "query": "What are the customs and safety tips for visiting Paris?",
    },
]


def print_banner():
    print("=" * 80)
    print("      TRIPMATE - AGENTIC AI TRAVEL ASSISTANT DEMO")
    print(f"      Powered by Google Gemini ({settings.gemini_model}) & LangGraph")
    print("=" * 80)
    print(f"API Key Configured: {settings.is_api_key_configured}")
    print(f"Weather Base URL:   {settings.weather_api_base_url}")
    print(f"Embedding Model:    {settings.embedding_model}")
    print("=" * 80 + "\n")


async def run_query(idx: int, category: str, query: str):
    print(f"\n[{idx}/6] SCENARIO: {category}")
    print(f">>> User Query: \"{query}\"")
    print("-" * 80)

    t0 = time.time()
    req = ChatRequest(query=query)
    response = await chat_service.process_chat(req, request_id=f"DEMO-{idx:03d}")
    elapsed = time.time() - t0

    print(f"\n[Tools Selected & Called]: {response.tools_used if response.tools_used else 'None (Direct Response)'}")
    print("\n[Execution Trace]:")
    if response.trace:
        for item in response.trace:
            print(f"  • Tool:      {item.tool}")
            print(f"    Reason:    {item.reason}")
            print(f"    Status:    {item.status.upper()}")
            if item.duration_ms:
                print(f"    Duration:  {item.duration_ms:.1f}ms")
            if item.summary:
                print(f"    Summary:   {item.summary}")
            if item.arguments:
                print(f"    Arguments: {json.dumps(item.arguments)}")
            print()
    else:
        print("  • No external tools required for this request.")

    print("\n[TripMate Synthesized Response]:")
    print(response.answer)
    print(f"\n[Turn Duration]: {elapsed:.2f} seconds | [Correlation ID]: {response.request_id}")
    print("=" * 80)


async def main():
    print_banner()

    # Pre-initialize vector store
    print("Pre-loading destination knowledge vector store...")
    store = get_vector_store()
    store.initialize()
    print(f"Knowledge base ready with {store.total_vectors} vectors.\n")

    if len(sys.argv) > 1:
        # User provided custom query
        custom_query = " ".join(sys.argv[1:])
        await run_query(1, "Custom User Query", custom_query)
        return

    # Run all automated scenarios
    for i, scenario in enumerate(SAMPLE_QUERIES, 1):
        await run_query(i, scenario["category"], scenario["query"])
        time.sleep(1)

    print("\nTripMate Demo execution completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

