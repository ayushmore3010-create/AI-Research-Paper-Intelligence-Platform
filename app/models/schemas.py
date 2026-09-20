"""Validated domain schemas shared by services and API routes."""

from datetime import datetime

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    title: str = "Unknown title"
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    research_problem: str = ""
    methodology: str = ""
    dataset: str = ""
    algorithms: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    results: str = ""
    limitations: str = ""
    future_work: str = ""
    conclusion: str = ""


class DocumentRecord(BaseModel):
    id: str
    filename: str
    sha256: str
    page_count: int
    chunk_count: int = 0
    status: str = "uploaded"
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    created_at: datetime


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    source: str


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
