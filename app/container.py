"""Composition root for shared application services."""

from app.config import settings
from app.embeddings.service import EmbeddingService
from app.rag.llm import build_llm
from app.rag.service import RAGService
from app.retrieval.faiss_store import FaissStore
from app.services.ingestion import IngestionService
from app.storage import MetadataStore

metadata_store = MetadataStore(settings.database_path)
embedding_service = EmbeddingService(settings.embedding_model)
vector_store = FaissStore(settings.vectorstore_dir)
ingestion_service = IngestionService(settings, metadata_store, embedding_service, vector_store)
rag_service = RAGService(settings, embedding_service, vector_store, metadata_store, build_llm(settings))
