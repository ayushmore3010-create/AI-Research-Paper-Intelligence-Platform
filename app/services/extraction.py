"""Structured research information extraction."""

from app.models.schemas import PaperMetadata


def summarize_paper(metadata: PaperMetadata) -> dict[str, str | list[str]]:
    """Return a stable structured summary; fields can later be LLM-enriched."""
    return metadata.model_dump()
