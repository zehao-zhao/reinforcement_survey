from recsysrl.llm.reranker import LLMReranker


def test_reranker_passthrough():
    c = [1, 3, 2]
    assert LLMReranker(enabled=False).rerank(c) == c
    assert LLMReranker(enabled=True).rerank(c) == [3, 2, 1]
