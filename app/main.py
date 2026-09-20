"""FastAPI backend for document and research assistant operations."""

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.container import ingestion_service, metadata_store, rag_service
from app.models.schemas import AnswerResponse, QuestionRequest, SearchResult

app = FastAPI(title=settings.app_name, version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/papers")
def list_papers():
    return metadata_store.list_documents()


@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    try:
        return ingestion_service.ingest(file.filename or "paper.pdf", await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Paper processing failed.") from exc


@app.delete("/api/papers/{document_id}", status_code=204)
def delete_paper(document_id: str) -> None:
    ingestion_service.delete(document_id)


@app.post("/api/search", response_model=list[SearchResult])
def search(question: QuestionRequest):
    return rag_service.search(question.question, question.top_k)


@app.post("/api/ask", response_model=AnswerResponse)
def ask(question: QuestionRequest):
    return rag_service.answer(question.question, question.top_k)
