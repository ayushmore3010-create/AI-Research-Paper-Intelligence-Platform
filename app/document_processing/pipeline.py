"""Safe, page-aware PDF ingestion and deterministic chunking."""

import hashlib
import re
import uuid
from pathlib import Path

import fitz

from app.models.schemas import DocumentChunk, PaperMetadata


class InvalidDocumentError(ValueError):
    """Raised when a PDF cannot be processed safely."""


def validate_pdf(content: bytes, max_size_mb: int) -> None:
    if not content or content[:4] != b"%PDF":
        raise InvalidDocumentError("The uploaded file is not a valid PDF.")
    if len(content) > max_size_mb * 1024 * 1024:
        raise InvalidDocumentError(f"PDF exceeds the {max_size_mb} MB upload limit.")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extract_pages(pdf_path: Path) -> list[str]:
    try:
        with fitz.open(pdf_path) as document:
            pages = [clean_text(page.get_text("text")) for page in document]
    except (fitz.FileDataError, OSError) as exc:
        raise InvalidDocumentError("The PDF could not be opened or is corrupted.") from exc
    if not any(pages):
        raise InvalidDocumentError("This PDF contains no extractable text; OCR is required for scanned papers.")
    return pages


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_pages(pages: list[str], document_id: str, document_name: str,
                chunk_size: int = 1200, overlap: int = 180) -> list[DocumentChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: list[DocumentChunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        start = 0
        while start < len(page_text):
            end = min(len(page_text), start + chunk_size)
            text = page_text[start:end].strip()
            if text:
                chunk_id = f"{document_id}-p{page_number}-{len(chunks):04d}"
                chunks.append(DocumentChunk(chunk_id=chunk_id, document_id=document_id,
                    document_name=document_name, page_number=page_number, text=text,
                    source=f"{document_name}, page {page_number}"))
            if end == len(page_text):
                break
            start = end - overlap
    return chunks


def infer_metadata(filename: str, pages: list[str]) -> PaperMetadata:
    text = " ".join(pages)
    title = next((line.strip() for line in pages[0].splitlines() if len(line.strip()) > 10), Path(filename).stem)
    abstract_match = re.search(r"abstract\s+(.*?)(?:introduction|keywords|1\s+introduction)", text, re.I)
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    return PaperMetadata(title=title[:200], year=int(year_match.group()) if year_match else None,
                         abstract=abstract_match.group(1)[:2000].strip() if abstract_match else "")


def new_document_id() -> str:
    return uuid.uuid4().hex
