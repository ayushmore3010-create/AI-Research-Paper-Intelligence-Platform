"""Small, transparent RAG evaluation metrics for labeled examples."""

from collections.abc import Iterable


def retrieval_relevance(retrieved_ids: Iterable[str], relevant_ids: set[str], k: int | None = None) -> float:
    ids = list(retrieved_ids)[:k] if k else list(retrieved_ids)
    return sum(item in relevant_ids for item in ids) / len(ids) if ids else 0.0


def context_precision(retrieved_ids: Iterable[str], relevant_ids: set[str]) -> float:
    return retrieval_relevance(retrieved_ids, relevant_ids)


def context_recall(retrieved_ids: Iterable[str], relevant_ids: set[str]) -> float:
    retrieved = set(retrieved_ids)
    return len(retrieved & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0


def citation_correctness(cited_ids: Iterable[str], retrieved_ids: set[str]) -> float:
    cited = list(cited_ids)
    return sum(item in retrieved_ids for item in cited) / len(cited) if cited else 0.0
