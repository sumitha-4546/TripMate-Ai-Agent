"""Ingest destination text files into local FAISS vector store."""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.rag.vector_store import get_vector_store
from app.utils.logger import logger


def main():
    logger.info("Starting destination data ingestion into FAISS vector store...")
    store = get_vector_store()
    store.initialize(force_reload=True)
    logger.info(f"Ingestion complete! Total vectors in FAISS index: {store.index.ntotal}")
    print(f"SUCCESS: Ingested {len(store.documents)} sections into FAISS vector store.")


if __name__ == "__main__":
    main()

