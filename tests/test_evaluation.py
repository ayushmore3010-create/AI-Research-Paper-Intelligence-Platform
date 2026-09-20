from evaluation.metrics import citation_correctness, context_recall, retrieval_relevance


def test_rag_metrics_are_bounded_and_correct():
    assert retrieval_relevance(["a", "b"], {"a"}) == 0.5
    assert context_recall(["a"], {"a", "b"}) == 0.5
    assert citation_correctness(["a", "x"], {"a", "b"}) == 0.5
