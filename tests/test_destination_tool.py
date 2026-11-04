import pytest
from app.rag.vector_store import get_vector_store
from app.tools.destination_tool import search_destination_guide


class MockEmbeddings:
    def embed_documents(self, texts):
        import numpy as np
        # Return deterministic unit vectors based on hash
        vectors = []
        for t in texts:
            val = float(sum(ord(c) for c in t) % 100) / 100.0
            vec = [val] * 384
            vectors.append(vec)
        return vectors

    def embed_query(self, text):
        val = float(sum(ord(c) for c in text) % 100) / 100.0
        return [val] * 384


def test_tokyo_visa_retrieval(monkeypatch):
    store = get_vector_store()
    # Initialize with real destination documents
    store.initialize()

    results = search_destination_guide.invoke({"query": "What are the visa requirements for Tokyo Japan?"})

    assert isinstance(results, list)
    assert len(results) > 0
    combined = " ".join(results).lower()
    assert "tokyo" in combined or "japan" in combined
    assert "visa" in combined
    assert "not authoritative" in combined or "disclaimer" in combined or "reference document" in combined


def test_bangkok_customs_retrieval():
    store = get_vector_store()
    store.initialize()

    results = search_destination_guide.invoke({"query": "What are the local customs and temple etiquette in Bangkok?"})

    assert isinstance(results, list)
    assert len(results) > 0
    combined = " ".join(results).lower()
    assert "bangkok" in combined or "thailand" in combined
    assert "customs" in combined or "temple" in combined or "shoes" in combined


def test_barcelona_safety_retrieval():
    store = get_vector_store()
    store.initialize()

    results = search_destination_guide.invoke({"query": "Is Barcelona safe and what should I know about pickpocketing?"})

    assert isinstance(results, list)
    assert len(results) > 0
    combined = " ".join(results).lower()
    assert "barcelona" in combined or "spain" in combined
    assert "safe" in combined or "pickpocket" in combined


def test_unsupported_destination():
    store = get_vector_store()
    store.initialize()

    results = search_destination_guide.invoke({"query": "What are the visa requirements for visiting Paris France?"})

    assert isinstance(results, list)
    assert len(results) == 1
    msg = results[0]
    assert "not available in TripMate" in msg or "Paris" in msg
    assert "Tokyo" in msg
    assert "Bangkok" in msg
    assert "Barcelona" in msg
    assert "Reykjavik" in msg

