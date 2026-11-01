import time
from typing import List, Optional
from langchain_core.tools import tool
from app.rag.vector_store import get_vector_store
from app.utils.logger import TraceLogger, logger


@tool
def search_destination_guide(query: str) -> List[str]:
    """Search curated destination travel guides for visa requirements, local customs,
    entry guidelines, best time to visit, safety, packing tips, and cultural etiquette.

    Covers Tokyo (Japan), Bangkok (Thailand), Barcelona (Spain), and Reykjavik (Iceland).

    Args:
        query: Specific question or search phrase regarding visa rules, customs, safety,
               packing suggestions, or travel seasons for a destination.

    Returns:
        A list of relevant guide excerpts with destination, section metadata, and official notes.
    """
    t0 = time.time()
    TraceLogger.tool_call("search_destination_guide", {"query": query})

    try:
        store = get_vector_store()
        results = store.search(query=query, k=3)
        duration_ms = (time.time() - t0) * 1000

        summary = f"Retrieved {len(results)} relevant section(s)"
        TraceLogger.tool_status(
            tool_name="search_destination_guide",
            status="success",
            duration_ms=duration_ms,
            summary=summary,
        )
        return results

    except Exception as e:
        duration_ms = (time.time() - t0) * 1000
        error_msg = f"Failed to retrieve destination guide: {str(e)}"
        TraceLogger.error(error_msg)
        TraceLogger.tool_status(
            tool_name="search_destination_guide",
            status="error",
            duration_ms=duration_ms,
            summary=error_msg,
        )
        return [
            "Destination guide knowledge retrieval encountered an error. Please try again."
        ]

