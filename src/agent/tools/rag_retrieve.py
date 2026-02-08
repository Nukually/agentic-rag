from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.llm.client import OpenAIClientBundle
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.vector_store import MilvusVectorStore, SearchHit


@dataclass(frozen=True)
class RetrievedHit:
    text: str
    source: str
    page: int
    vector_score: float
    rerank_score: float | None


@dataclass(frozen=True)
class RetrievalResult:
    vector_hits: list[RetrievedHit]
    final_hits: list[RetrievedHit]
    reranker_applied: bool
    reranker_message: str
    keyword_hits: list[RetrievedHit] | None = None


def _to_retrieved(hit: SearchHit, rerank_score: float | None = None) -> RetrievedHit:
    return RetrievedHit(
        text=hit.text,
        source=hit.source,
        page=hit.page,
        vector_score=hit.score,
        rerank_score=rerank_score,
    )


def retrieve_hits(
    query: str,
    llm_clients: OpenAIClientBundle,
    vector_store: MilvusVectorStore,
    reranker: OpenAIStyleReranker,
    top_k: int,
    candidate_k: int,
    keyword_index: KeywordIndex | None = None,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> RetrievalResult:
    candidate_k = max(top_k, candidate_k)
    fetch_k = candidate_k
    query_vector = llm_clients.embed_texts([query])[0]
    vector_hits_raw = vector_store.search(query_vector=query_vector, top_k=fetch_k)
    keyword_hits_raw = keyword_index.search(query, top_k=fetch_k) if keyword_index else []

    vector_hits = [_to_retrieved(hit) for hit in vector_hits_raw]
    keyword_hits = [_to_retrieved(hit) for hit in keyword_hits_raw] if keyword_index else None
    candidates = _merge_candidates(
        vector_hits=vector_hits_raw,
        keyword_hits=keyword_hits_raw,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        top_k=candidate_k,
    )
    if not candidates:
        return RetrievalResult(
            vector_hits=vector_hits,
            final_hits=[],
            reranker_applied=False,
            reranker_message="no candidate hits",
            keyword_hits=keyword_hits,
        )

    rerank_result = reranker.rerank(query=query, hits=candidates, top_k=top_k)
    if rerank_result.applied and rerank_result.items:
        final_hits = [_to_retrieved(item.hit, item.rerank_score) for item in rerank_result.items[:top_k]]
        return RetrievalResult(
            vector_hits=vector_hits,
            final_hits=final_hits,
            reranker_applied=True,
            reranker_message=rerank_result.message,
            keyword_hits=keyword_hits,
        )

    return RetrievalResult(
        vector_hits=vector_hits,
        final_hits=[_to_retrieved(hit) for hit in candidates[:top_k]],
        reranker_applied=False,
        reranker_message=rerank_result.message,
        keyword_hits=keyword_hits,
    )


def _merge_candidates(
    vector_hits: Iterable[SearchHit],
    keyword_hits: Iterable[SearchHit],
    vector_weight: float,
    keyword_weight: float,
    top_k: int,
) -> list[SearchHit]:
    vector_list = list(vector_hits)
    keyword_list = list(keyword_hits)
    if vector_list and not keyword_list:
        return vector_list[: max(1, top_k)]
    if keyword_list and not vector_list:
        return keyword_list[: max(1, top_k)]

    total = vector_weight + keyword_weight
    if total <= 0:
        vector_weight, keyword_weight = 1.0, 0.0
    else:
        vector_weight /= total
        keyword_weight /= total

    entries: dict[tuple[str, int, str], dict[str, object]] = {}
    for hit in vector_list:
        key = (hit.source, hit.page, hit.text)
        entries[key] = {"hit": hit, "v": hit.score, "k": 0.0}
    for hit in keyword_list:
        key = (hit.source, hit.page, hit.text)
        if key in entries:
            entries[key]["k"] = hit.score
        else:
            entries[key] = {"hit": hit, "v": 0.0, "k": hit.score}

    if not entries:
        return []

    v_norm = _normalize_scores((key, item["v"]) for key, item in entries.items())
    k_norm = _normalize_scores((key, item["k"]) for key, item in entries.items())

    scored: list[tuple[SearchHit, float]] = []
    for key, item in entries.items():
        combined = vector_weight * v_norm.get(key, 0.0) + keyword_weight * k_norm.get(key, 0.0)
        scored.append((item["hit"], combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    output: list[SearchHit] = []
    for hit, score in scored[: max(1, top_k)]:
        output.append(SearchHit(text=hit.text, source=hit.source, page=hit.page, score=float(score)))
    return output


def _normalize_scores(items: Iterable[tuple[tuple[str, int, str], float]]) -> dict[tuple[str, int, str], float]:
    values = list(items)
    if not values:
        return {}
    scores = [score for _, score in values]
    min_score = min(scores)
    max_score = max(scores)
    if max_score <= min_score:
        if max_score == 0.0:
            return {key: 0.0 for key, _ in values}
        return {key: 1.0 for key, _ in values}
    span = max_score - min_score
    return {key: (score - min_score) / span for key, score in values}
