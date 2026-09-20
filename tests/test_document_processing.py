from app.document_processing.pipeline import InvalidDocumentError, chunk_pages, validate_pdf


def test_chunking_preserves_page_metadata():
    chunks = chunk_pages(["research finding " * 150, "second page"], "doc-1", "paper.pdf", 100, 20)
    assert chunks
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert {chunk.page_number for chunk in chunks} == {1, 2}


def test_invalid_pdf_is_rejected():
    try:
        validate_pdf(b"not a pdf", 25)
    except InvalidDocumentError:
        return
    raise AssertionError("invalid PDF was accepted")
