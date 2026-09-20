"""Grounded retrieval and citation construction."""

from app.config import Settings
from app.embeddings.service import EmbeddingService
from app.models.schemas import AnswerResponse, Citation, SearchResult
from app.rag.llm import LLMProvider
from app.retrieval.faiss_store import FaissStore
from app.storage import MetadataStore


class RAGService:
    def __init__(self, config: Settings, embeddings: EmbeddingService, vectorstore: FaissStore,
                 metadata: MetadataStore, llm: LLMProvider):
        self.config, self.embeddings, self.vectorstore = config, embeddings, vectorstore
        self.metadata, self.llm = metadata, llm

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        results = self.vectorstore.search(self.embeddings.encode([query]), top_k or self.config.retrieval_top_k,
                                          self.config.retrieval_threshold)
        self.metadata.log_query(query, len(results))
        return results

    def answer(self, question: str, top_k: int | None = None) -> AnswerResponse:
        results = self.search(question, top_k)
        if not results:
            return AnswerResponse(answer="Information not found in the uploaded papers.", grounded=False)
        context = "\n\n".join(f"[{index}] {result.chunk.source}\n{result.chunk.text}" for index, result in enumerate(results, 1))
        answer = self.llm.generate(f"Question: {question}\n\nContext:\n{context}\n\nCite sources as [number].")
        citations = [Citation(document_name=result.chunk.document_name, page_number=result.chunk.page_number,
                              chunk_id=result.chunk.chunk_id, excerpt=result.chunk.text[:300]) for result in results]
        return AnswerResponse(answer=answer, citations=citations)
