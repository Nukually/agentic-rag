from dataclasses import replace

from src.agent.tools.rag_retrieve import _merge_candidates
from src.retrieval.vector_store import SearchHit


def hit(name, score, **kwargs):
    return SearchHit(name, f'{name}.txt', 1, score, **kwargs)


def test_rrf_prefers_cross_retriever_agreement_over_extreme_raw_scores():
    dense = [hit('dense-only', .99), hit('shared', .98)]
    sparse = [hit('sparse-only', 1000000), hit('shared', 1)]
    result = _merge_candidates(dense, sparse, .6, .4, 3)
    assert [item.text for item in result] == ['shared', 'dense-only', 'sparse-only']
    assert result[0].score > result[1].score


def test_rrf_depends_on_ranks_not_score_scale():
    dense = [hit('A', .9), hit('B', .8)]
    sparse = [hit('B', 20), hit('C', 10)]
    expected = _merge_candidates(dense, sparse, .6, .4, 3)
    scaled = _merge_candidates([replace(h, score=h.score*100) for h in dense], [replace(h, score=h.score*.001) for h in sparse], .6, .4, 3)
    assert scaled == expected


def test_zero_weight_branch_is_excluded_even_if_it_is_the_only_branch():
    assert _merge_candidates([], [hit('A', 100)], 1, 0, 10) == []


def test_rrf_does_not_double_count_duplicates_within_a_branch():
    a, b = hit('A', .9), hit('B', .8)
    assert _merge_candidates([a, a, b], [], 1, 0, 3) == _merge_candidates([a, b], [], 1, 0, 3)


def test_empty_and_zero_top_k():
    assert _merge_candidates([], [], .6, .4, 3) == []
    assert _merge_candidates([hit('A', 1)], [], 1, 0, 0) == []


def test_multi_query_batches_keep_independent_ranks_and_arrival_order():
    first = [hit('A', .99), hit('shared', .98)]
    second = [hit('B', .20), hit('shared', .19)]
    sparse = [[hit('shared', 9000), hit('C', 2)]]
    result = _merge_candidates([], [], .6, .4, 4, vector_hit_batches=[first, second], keyword_hit_batches=sparse)
    reversed_result = _merge_candidates([], [], .6, .4, 4, vector_hit_batches=[second, first], keyword_hit_batches=sparse)
    assert [h.text for h in result] == ['shared', 'A', 'B', 'C']
    assert reversed_result == result
    assert result[0].score == .6 / 62 + .4 / 61


def test_multi_query_duplicates_do_not_multiply_a_retriever_vote():
    batch = [hit('A', .9), hit('B', .8)]
    single = _merge_candidates([], [], 1, 0, 2, vector_hit_batches=[batch])
    repeated = _merge_candidates([], [], 1, 0, 2, vector_hit_batches=[batch, batch])
    assert repeated == single


def test_retrieve_hits_runs_fusion_without_model_or_reranker_calls():
    from src.agent.tools.rag_retrieve import retrieve_hits
    from src.retrieval.reranker import OpenAIStyleReranker

    class Clients:
        def embed_texts(self, texts):
            assert texts == ['synthetic question']
            return [[1., 0.]]

    class Dense:
        def search(self, **kwargs):
            return [hit('dense-only', .99), hit('shared', .98, chunk_id='chunk_shared')]

    class Sparse:
        def search(self, *args, **kwargs):
            return [hit('sparse-only', 1000000), hit('shared', 1, chunk_id='chunk_shared')]

    result = retrieve_hits('synthetic question', Clients(), Dense(), OpenAIStyleReranker('', '', '', 1), 2, 3,
                           keyword_index=Sparse(), query_rewrite_enabled=False, multi_query_enabled=False)
    assert [h.text for h in result.final_hits] == ['shared', 'dense-only']
    assert result.final_hits[0].chunk_id == 'chunk_shared'
    assert result.reranker_applied is False


def test_reference_prompts_keep_document_and_chunk_identity():
    from src.llm.prompts import build_agent_answer_prompt, build_user_prompt
    contexts = [{'text':'synthetic', 'source':'a.txt', 'page':'1', 'doc_id':'doc_a', 'chunk_id':'chunk_a'}]
    for prompt in (build_user_prompt('question', contexts), build_agent_answer_prompt('question', [], contexts)):
        assert '[ref:1]' in prompt
        assert 'doc_id=doc_a chunk_id=chunk_a' in prompt
