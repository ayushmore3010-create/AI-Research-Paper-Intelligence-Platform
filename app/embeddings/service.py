"""Pluggable embedding service with a lightweight offline fallback."""

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._fallback = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
        self._fallback_fitted = False

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype="float32")
        except Exception:
            if not self._fallback_fitted:
                vectors = self._fallback.fit_transform(texts)
                self._fallback_fitted = True
            else:
                vectors = self._fallback.transform(texts)
            dense = vectors.toarray().astype("float32")
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            return dense / np.maximum(norms, 1e-12)
