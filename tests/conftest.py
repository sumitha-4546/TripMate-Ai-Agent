import os
import sys
from pathlib import Path
import pytest

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure testing environment variables
os.environ["APP_ENV"] = "testing"
os.environ["GEMINI_API_KEY"] = "mock_test_gemini_key_12345"
os.environ["GEMINI_MODEL"] = "gemini-1.5-flash"
os.environ["WEATHER_API_BASE_URL"] = "https://api.open-meteo.com"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

from app.config import settings
from app.rag.document_loader import load_destination_documents
from app.rag.vector_store import DestinationVectorStore, get_vector_store


@pytest.fixture(scope="session", autouse=True)
def ensure_vector_store_initialized():
    store = get_vector_store()
    store.initialize()
