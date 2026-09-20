"""Document ingestion orchestration."""

from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.document_processing.pipeline import (chunk_pages, extract_pages, infer_metadata,
                                               new_document_id, sha256_bytes, validate_pdf)
from app.embeddings.service import EmbeddingService
from app.models.schemas import DocumentRecord
from app.retrieval.faiss_store import FaissStore
from app.storage import MetadataStore


class IngestionService:
    def __init__(self, config: Settings, metadata: MetadataStore, embeddings: EmbeddingService, vectorstore: FaissStore):
        self.config = config
        self.metadata = metadata
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def ingest(self, filename: str, content: bytes) -> DocumentRecord:
        validate_pdf(content, self.config.max_upload_mb)
        digest = sha256_bytes(content)
        existing = self.metadata.find_by_hash(digest)
        if existing:
            return existing
        document_id = new_document_id()
        safe_name = Path(filename).name
        pdf_path = self.config.uploads_dir / f"{document_id}.pdf"
        pdf_path.write_bytes(content)
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, document_id, safe_name)
        record = DocumentRecord(id=document_id, filename=safe_name, sha256=digest,
            page_count=len(pages), chunk_count=len(chunks), status="processing",
            metadata=infer_metadata(safe_name, pages), created_at=datetime.now(timezone.utc))
        self.metadata.save_document(record)
        try:
            self.vectorstore.add(chunks, self.embeddings.encode([chunk.text for chunk in chunks]))
            self.metadata.update_status(document_id, "ready", len(chunks))
            return record.model_copy(update={"status": "ready"})
        except Exception:
            self.metadata.update_status(document_id, "failed")
            raise

    def delete(self, document_id: str) -> None:
        self.vectorstore.remove_document(document_id)
        record = next((item for item in self.metadata.list_documents() if item.id == document_id), None)
        if record:
            (self.config.uploads_dir / f"{document_id}.pdf").unlink(missing_ok=True)
            self.metadata.delete_document(document_id)
