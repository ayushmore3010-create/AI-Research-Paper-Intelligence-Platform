import numpy as np

from app.models.schemas import DocumentChunk
from app.retrieval.faiss_store import FaissStore


def test_faiss_search_and_delete(tmp_path):
    store = FaissStore(tmp_path)
    chunks = [DocumentChunk(chunk_id="a", document_id="doc-a", document_name="a.pdf", page_number=1, text="cats", source="a.pdf, page 1"),
              DocumentChunk(chunk_id="b", document_id="doc-b", document_name="b.pdf", page_number=2, text="dogs", source="b.pdf, page 2")]
    vectors = np.array([[1, 0], [0, 1]], dtype="float32")
    store.add(chunks, vectors)
    assert store.search(np.array([[1, 0]], dtype="float32"), 2, 0.5)[0].chunk.chunk_id == "a"
    store.remove_document("doc-a")
    assert store.search(np.array([[0, 1]], dtype="float32"), 2, 0.5)[0].chunk.chunk_id == "b"
