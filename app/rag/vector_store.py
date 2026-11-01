import os
import pickle
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
from langchain_core.documents import Document

try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, Exception):
    faiss = None
    FAISS_AVAILABLE = False

from app.config import settings
from app.rag.document_loader import (
    DEFAULT_DISCLAIMER,
    KNOWN_DESTINATIONS,
    load_destination_documents,
)
from app.rag.embeddings import get_embedding_function
from app.utils.logger import logger


class DestinationVectorStore:
    """FAISS-based destination vector store with local persistence, metadata filtering,
    and resilient cosine similarity fallback.
    """

    def __init__(self, data_dir: Path = settings.data_dir, store_dir: Path = settings.vector_store_dir) -> None:
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.embedder = get_embedding_function()
        self.index = None
        self.vectors: Optional[np.ndarray] = None
        self.documents: List[Document] = []
        self._is_initialized = False

    @property
    def total_vectors(self) -> int:
        if self.index is not None:
            return self.index.ntotal
        if self.vectors is not None:
            return len(self.vectors)
        return len(self.documents)

    def initialize(self, force_reload: bool = False) -> None:
        """Initialize or load the vector index."""
        if self._is_initialized and not force_reload:
            return

        # Check if persisted index exists
        docs_file = self.store_dir / "docs.pkl"
        vec_file = self.store_dir / "vectors.npy"
        idx_file = self.store_dir / "index.bin"

        if not force_reload and docs_file.exists() and (vec_file.exists() or idx_file.exists()):
            try:
                self._load_from_disk()
                self._is_initialized = True
                logger.info(f"Destination vector store loaded from disk ({self.total_vectors} vectors).")
                return
            except Exception as e:
                logger.warning(f"Failed loading persisted vector store: {e}. Rebuilding...")

        self._build_index()
        self._is_initialized = True

    def _build_index(self) -> None:
        """Build vector index from destination documents and persist to disk."""
        logger.info(f"Building destination index from {self.data_dir}...")
        self.documents = load_destination_documents(self.data_dir)

        if not self.documents:
            logger.warning("No destination documents loaded! Index will be empty.")
            dim = 384
            if FAISS_AVAILABLE and faiss is not None:
                self.index = faiss.IndexFlatIP(dim)
            self.vectors = np.empty((0, dim), dtype=np.float32)
            return

        texts = [doc.page_content for doc in self.documents]
        embeddings = self.embedder.embed_documents(texts)
        self.vectors = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms

        dim = self.vectors.shape[1]
        if FAISS_AVAILABLE and faiss is not None:
            try:
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(self.vectors)
                logger.info(f"FAISS index populated with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.warning(f"FAISS index construction failed: {e}. Using NumPy similarity.")
                self.index = None

        self._save_to_disk()
        logger.info(f"Vector store built with {self.total_vectors} vectors (dim={dim}).")

    def _save_to_disk(self) -> None:
        """Save vector store and documents metadata to disk."""
        try:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            with open(self.store_dir / "docs.pkl", "wb") as f:
                pickle.dump(self.documents, f)

            if self.vectors is not None:
                np.save(self.store_dir / "vectors.npy", self.vectors)

            if FAISS_AVAILABLE and faiss is not None and self.index is not None:
                try:
                    faiss.write_index(self.index, str(self.store_dir / "index.bin"))
                except Exception:
                    pass

            logger.info(f"Vector store persisted to {self.store_dir}")
        except Exception as e:
            logger.error(f"Error persisting vector store to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load index and documents metadata from disk."""
        with open(self.store_dir / "docs.pkl", "rb") as f:
            self.documents = pickle.load(f)

        vec_path = self.store_dir / "vectors.npy"
        if vec_path.exists():
            self.vectors = np.load(vec_path)

        idx_path = self.store_dir / "index.bin"
        if FAISS_AVAILABLE and faiss is not None and idx_path.exists():
            try:
                self.index = faiss.read_index(str(idx_path))
                logger.info(f"Loaded FAISS binary index with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.warning(f"Could not load FAISS binary index: {e}. Falling back to stored vectors.")
                self.index = None

    def detect_destination(self, query: str) -> Optional[str]:
        """Detect if a known destination is mentioned in the query."""
        q_lower = query.lower()
        for key, info in KNOWN_DESTINATIONS.items():
            if key in q_lower or info["country"].lower() in q_lower:
                return info["city"]
        return None

    def detect_unsupported_destination(self, query: str) -> Optional[str]:
        """Detect if the user is asking about an unsupported city."""
        common_destinations = [
            "paris", "london", "rome", "bali", "new york", "dubai", "berlin",
            "singapore", "sydney", "toronto", "amsterdam", "madrid", "seoul",
            "kyoto", "osaka", "phuket", "cancun", "hawaii", "cairo"
        ]
        q_lower = query.lower()
        for city in common_destinations:
            import re
            if re.search(rf"\b{city}\b", q_lower):
                if city not in KNOWN_DESTINATIONS:
                    return city.capitalize()
        return None

    def search(
        self,
        query: str,
        k: int = 3,
        destination_filter: Optional[str] = None,
    ) -> List[str]:
        """Search destination guide documents using semantic similarity."""
        self.initialize()

        if self.total_vectors == 0:
            return ["No destination guide data is currently available in the knowledge base."]

        # Check for explicitly unsupported destinations
        unsupported = self.detect_unsupported_destination(query)
        if unsupported:
            supported_list = ", ".join([v["city"] for v in KNOWN_DESTINATIONS.values()])
            return [
                f"Information for '{unsupported}' is not available in TripMate's curated destination knowledge base. "
                f"Currently supported destinations are: {supported_list}."
            ]

        # Auto-detect destination if not explicitly filtered
        if not destination_filter:
            destination_filter = self.detect_destination(query)

        query_emb = self.embedder.embed_query(query)
        q_np = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_np)
        if q_norm > 0:
            q_np = q_np / q_norm

        search_k = len(self.documents) if destination_filter else min(len(self.documents), k * 3)

        # Compute vector similarities
        if self.vectors is not None and len(self.vectors) > 0:
            similarities = np.dot(self.vectors, q_np)
        else:
            similarities = np.zeros(len(self.documents), dtype=np.float32)

        # Hybrid keyword relevance boost: prioritize exact section / term matches
        q_tokens = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        keyword_boosts = np.zeros(len(self.documents), dtype=np.float32)
        for i, doc in enumerate(self.documents):
            c_low = doc.page_content.lower()
            s_low = doc.metadata.get("section", "").lower()
            score = 0.0
            for t in q_tokens:
                if t in s_low:
                    score += 0.5  # high boost for section title match (e.g. visa, safety)
                elif t in c_low:
                    score += 0.2  # moderate boost for body content match
            keyword_boosts[i] = score

        final_scores = similarities + keyword_boosts
        ranked_indices = np.argsort(final_scores)[::-1][:search_k]

        results: List[str] = []
        for idx in ranked_indices:
            if idx == -1 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]

            if destination_filter:
                doc_city = doc.metadata.get("city", "")
                doc_country = doc.metadata.get("country", "")
                if destination_filter.lower() not in (doc_city.lower(), doc_country.lower()):
                    continue

            results.append(doc.page_content)
            if len(results) >= k:
                break

        # Fallback to destination documents if filter excluded all, else unfiltered top results
        if not results:
            if destination_filter:
                for doc in self.documents:
                    doc_c = doc.metadata.get("city", "").lower()
                    doc_co = doc.metadata.get("country", "").lower()
                    if destination_filter.lower() in (doc_c, doc_co):
                        results.append(doc.page_content)
                        if len(results) >= k:
                            break
            if not results:
                for idx in ranked_indices[:k]:
                    if 0 <= idx < len(self.documents):
                        results.append(self.documents[idx].page_content)

        return results


_VECTOR_STORE_INSTANCE: Optional[DestinationVectorStore] = None


def get_vector_store() -> DestinationVectorStore:
    """Singleton getter for DestinationVectorStore."""
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        _VECTOR_STORE_INSTANCE = DestinationVectorStore()
    return _VECTOR_STORE_INSTANCE
