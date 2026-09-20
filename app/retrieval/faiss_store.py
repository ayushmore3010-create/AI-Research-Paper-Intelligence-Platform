"""Persistent FAISS index with JSON metadata sidecar."""

import json
from pathlib import Path

import faiss
import numpy as np

from app.models.schemas import DocumentChunk, SearchResult


class FaissStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = directory / "papers.faiss"
        self.metadata_path = directory / "chunks.json"
        self.vectors_path = directory / "vectors.npy"
        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []
        self.vectors = np.empty((0, 0), dtype="float32")
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.chunks = [DocumentChunk(**item) for item in json.loads(self.metadata_path.read_text(encoding="utf-8"))]
            if self.vectors_path.exists():
                self.vectors = np.load(self.vectors_path)

    def add(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> None:
        if not len(chunks):
            return
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        if self.index.d != vectors.shape[1]:
            raise ValueError("Embedding dimension does not match the existing vector index")
        self.index.add(vectors.astype("float32"))
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype("float32")]) if self.vectors.size else vectors.astype("float32")
        self._save()

    def remove_document(self, document_id: str) -> None:
        kept = [chunk for chunk in self.chunks if chunk.document_id != document_id]
        if len(kept) == len(self.chunks):
            return
        kept_indices = [index for index, chunk in enumerate(self.chunks) if chunk.document_id != document_id]
        self.chunks = kept
        self.vectors = self.vectors[kept_indices] if len(kept_indices) else np.empty((0, 0), dtype="float32")
        self._rebuild()

    def search(self, vector: np.ndarray, top_k: int, threshold: float) -> list[SearchResult]:
        if self.index is None or not self.chunks:
            return []
        scores, indices = self.index.search(vector.astype("float32"), min(top_k, len(self.chunks)))
        return [SearchResult(chunk=self.chunks[i], score=float(score))
                for score, i in zip(scores[0], indices[0]) if i >= 0 and score >= threshold]

    def _rebuild(self) -> None:
        if not self.chunks:
            self.index = None
            self.index_path.unlink(missing_ok=True)
            self.metadata_path.unlink(missing_ok=True)
            self.vectors_path.unlink(missing_ok=True)
            return
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        self._save()

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps([chunk.model_dump() for chunk in self.chunks]), encoding="utf-8")
        np.save(self.vectors_path, self.vectors)
