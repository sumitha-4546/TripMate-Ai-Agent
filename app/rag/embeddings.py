import os
from typing import List
from langchain_core.embeddings import Embeddings
from app.utils.logger import logger
from app.config import settings

_EMBEDDER_INSTANCE = None


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Local Sentence Transformers embeddings wrapper conforming to LangChain's Embeddings interface.

    Uses `sentence-transformers/all-MiniLM-L6-v2` locally on CPU.
    In testing mode (APP_ENV=testing), uses fast deterministic embeddings for instantaneous test execution.
    """

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load SentenceTransformer model on first invocation."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Prefer local model path inside workspace if available
            local_model_dir = settings.base_dir / "data" / "models" / "all-MiniLM-L6-v2"
            model_target = str(local_model_dir) if local_model_dir.exists() else self.model_name
            logger.info(f"Loading SentenceTransformer model from: {model_target}...")
            self._model = SentenceTransformer(model_target)
            logger.info("SentenceTransformer model loaded successfully.")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of documents."""
        if not texts:
            return []
        if os.getenv("APP_ENV", "").lower() == "testing" or settings.app_env == "testing":
            results = []
            for t in texts:
                vec = [0.0] * 384
                h = sum(ord(c) for c in t) % 384
                vec[h] = 1.0
                results.append(vec)
            return results

        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error computing document embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Compute embedding for a single query."""
        if os.getenv("APP_ENV", "").lower() == "testing" or settings.app_env == "testing":
            vec = [0.0] * 384
            h = sum(ord(c) for c in text) % 384
            vec[h] = 1.0
            return vec

        try:
            embedding = self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error computing query embedding: {e}")
            raise


def get_embedding_function() -> LocalSentenceTransformerEmbeddings:
    """Singleton getter for the local embedding model."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = LocalSentenceTransformerEmbeddings()
    return _EMBEDDER_INSTANCE
