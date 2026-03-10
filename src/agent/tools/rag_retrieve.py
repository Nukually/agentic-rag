"""Hybrid retrieval helper: dense vector + BM25 + optional reranking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from dataclasses import dataclass
from typing import Iterable

from src.llm.client import OpenAIClientBundle
from src.llm.prompts import (
    MULTI_QUERY_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    build_multi_query_prompt,
    build_query_rewrite_prompt,
)
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.vector_store import MilvusVectorStore, SearchHit


@dataclass(frozen=True)
class RetrievedHit:
    """Normalized retrieval hit used by agent tools and answer stage."""

    text: str
    source: str
    page: int
    vector_score: float
    rerank_score: float | None
    doc_id: str = ""
    file_name: str = ""
    source_type: str = ""
    company_code: str = ""
    company_name: str = ""
    report_year: int | None = None
    is_table: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    """Detailed retrieval output containing intermediate and final rankings."""

    vector_hits: list[RetrievedHit]
    final_hits: list[RetrievedHit]
    reranker_applied: bool
    reranker_message: str
    keyword_hits: list[RetrievedHit] | None = None
    retrieval_query: str = ""
    retrieval_queries: list[str] | None = None


def _to_retrieved(hit: SearchHit, rerank_score: float | None = None) -> RetrievedHit:
    """Convert a low-level search hit into project-level retrieval hit."""

    return RetrievedHit(
        text=hit.text,
        source=hit.source,
        page=hit.page,
        vector_score=hit.score,
        rerank_score=rerank_score,
        doc_id=hit.doc_id,
        file_name=hit.file_name,
        source_type=hit.source_type,
        company_code=hit.company_code,
        company_name=hit.company_name,
        report_year=hit.report_year,
        is_table=hit.is_table,
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
    query_rewrite_enabled: bool = True,
    multi_query_enabled: bool = True,
    multi_query_count: int = 3,
    diversify_by_company: bool = False,
    parallel_enabled: bool = True,
    parallel_max_workers: int = 8,
) -> RetrievalResult:
    """Retrieve and rank document chunks for a query.

    Args:
        query: User query text.
        llm_clients: Client bundle used to compute query embedding.
        vector_store: Vector index backend.
        reranker: Optional reranker client (can gracefully degrade).
        top_k: Final result count.
        candidate_k: Candidate pool size before rerank.
        keyword_index: Optional BM25 index.
        vector_weight: Weight for normalized vector score.
        keyword_weight: Weight for normalized keyword score.
        query_rewrite_enabled: Whether to run query rewriting before retrieval.
        multi_query_enabled: Whether to generate additional query variants.
        multi_query_count: Number of query variants to use (including primary).
        diversify_by_company: Whether to prioritize company coverage in final
            top-k selection.
        parallel_enabled: Whether to parallelize per-query retrieval branches.
        parallel_max_workers: Upper bound of thread workers for retrieval.

    Returns:
        RetrievalResult: Vector hits, optional keyword hits, and final hits.

    Example:
        >>> result = retrieve_hits(
        ...     query="net profit in 2025",
        ...     llm_clients=clients,
        ...     vector_store=store,
        ...     reranker=reranker,
        ...     top_k=8,
        ...     candidate_k=64,
        ... )
        >>> len(result.final_hits)
        8
    """

    raw_query = " ".join(query.strip().split())
    if not raw_query:
        return RetrievalResult(
            vector_hits=[],
            final_hits=[],
            reranker_applied=False,
            reranker_message="empty query",
            keyword_hits=[] if keyword_index else None,
            retrieval_query="",
            retrieval_queries=[],
        )

    retrieval_query = raw_query
    if query_rewrite_enabled:
        retrieval_query = _rewrite_query(raw_query, llm_clients=llm_clients)
    retrieval_queries = [retrieval_query]
    if multi_query_enabled and multi_query_count > 1:
        retrieval_queries = _build_multi_queries(
            base_query=retrieval_query,
            llm_clients=llm_clients,
            count=multi_query_count,
        )

    candidate_k = max(top_k, candidate_k)
    fetch_k = candidate_k
    vector_hit_batches, keyword_hit_batches = _collect_hit_batches(
        retrieval_queries=retrieval_queries,
        llm_clients=llm_clients,
        vector_store=vector_store,
        keyword_index=keyword_index,
        fetch_k=fetch_k,
        parallel_enabled=parallel_enabled,
        parallel_max_workers=parallel_max_workers,
    )

    vector_hits_raw = _merge_query_hits(vector_hit_batches, top_k=fetch_k)
    keyword_hits_raw = _merge_query_hits(keyword_hit_batches, top_k=fetch_k) if keyword_index else []

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
            retrieval_query=retrieval_query,
            retrieval_queries=retrieval_queries,
        )

    rerank_top_n = max(top_k, candidate_k) if diversify_by_company else top_k
    rerank_result = reranker.rerank(query=retrieval_query, hits=candidates, top_k=rerank_top_n)
    if rerank_result.applied and rerank_result.items:
        reranked_pool = [_to_retrieved(item.hit, item.rerank_score) for item in rerank_result.items]
        if diversify_by_company:
            final_hits = _diversify_retrieved_hits_by_company(reranked_pool, top_k=top_k)
        else:
            final_hits = reranked_pool[:top_k]
        return RetrievalResult(
            vector_hits=vector_hits,
            final_hits=final_hits,
            reranker_applied=True,
            reranker_message=rerank_result.message,
            keyword_hits=keyword_hits,
            retrieval_query=retrieval_query,
            retrieval_queries=retrieval_queries,
        )

    fallback_pool = [_to_retrieved(hit) for hit in candidates]
    fallback_hits = fallback_pool[:top_k]
    if diversify_by_company:
        fallback_hits = _diversify_retrieved_hits_by_company(fallback_pool, top_k=top_k)

    return RetrievalResult(
        vector_hits=vector_hits,
        final_hits=fallback_hits,
        reranker_applied=False,
        reranker_message=rerank_result.message,
        keyword_hits=keyword_hits,
        retrieval_query=retrieval_query,
        retrieval_queries=retrieval_queries,
    )


def _rewrite_query(query: str, llm_clients: OpenAIClientBundle) -> str:
    """Rewrite a user query into a retrieval-oriented single query."""

    prompt = build_query_rewrite_prompt(query)
    try:
        raw = llm_clients.chat(
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
    except Exception:
        return query

    parsed = _extract_json(raw)
    if parsed and isinstance(parsed.get("query"), str):
        candidate = " ".join(str(parsed["query"]).strip().split())
        if candidate:
            return candidate

    stripped = _strip_wrappers(raw)
    return stripped or query


def _build_multi_queries(base_query: str, llm_clients: OpenAIClientBundle, count: int) -> list[str]:
    """Generate multiple equivalent query variants for broader recall."""

    safe_count = max(1, min(int(count), 8))
    prompt = build_multi_query_prompt(base_query, count=safe_count)
    try:
        raw = llm_clients.chat(
            messages=[
                {"role": "system", "content": MULTI_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
    except Exception:
        return [base_query]

    candidates: list[str] = []
    parsed = _extract_json(raw)
    if parsed and isinstance(parsed.get("queries"), list):
        for item in parsed["queries"]:
            if not isinstance(item, str):
                continue
            normalized = " ".join(item.strip().split())
            if normalized:
                candidates.append(normalized)
    else:
        # Fallback for non-JSON responses.
        for line in raw.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            cleaned = _strip_wrappers(cleaned)
            if cleaned:
                candidates.append(cleaned)

    merged: list[str] = [base_query]
    seen = {base_query}
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
        if len(merged) >= safe_count:
            break
    return merged


def _collect_hit_batches(
    retrieval_queries: list[str],
    llm_clients: OpenAIClientBundle,
    vector_store: MilvusVectorStore,
    keyword_index: KeywordIndex | None,
    fetch_k: int,
    parallel_enabled: bool,
    parallel_max_workers: int,
) -> tuple[list[list[SearchHit]], list[list[SearchHit]]]:
    """Collect vector/keyword hit batches for all query variants.

    Implementation notes:
    - Query embeddings are computed in one batch request to reduce overhead.
    - Per-query retrieval can run in parallel.
    - If any parallel branch fails, that branch falls back to serial execution.
    """

    if not retrieval_queries:
        return [], []

    query_vectors = llm_clients.embed_texts(retrieval_queries)
    if len(query_vectors) != len(retrieval_queries):
        raise ValueError("embedding result count does not match retrieval query count")

    use_parallel = (
        parallel_enabled
        and len(retrieval_queries) > 1
        and int(parallel_max_workers) > 1
    )
    if not use_parallel:
        return _collect_hit_batches_serial(
            retrieval_queries=retrieval_queries,
            query_vectors=query_vectors,
            vector_store=vector_store,
            keyword_index=keyword_index,
            fetch_k=fetch_k,
        )

    max_workers = min(max(1, int(parallel_max_workers)), len(retrieval_queries))
    vector_hit_batches: list[list[SearchHit]] = [[] for _ in retrieval_queries]
    keyword_hit_batches: list[list[SearchHit]] = (
        [[] for _ in retrieval_queries] if keyword_index is not None else []
    )
    failed_indices: list[int] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _search_one_query,
                retrieval_query=retrieval_q,
                query_vector=query_vectors[idx],
                vector_store=vector_store,
                keyword_index=keyword_index,
                fetch_k=fetch_k,
            ): idx
            for idx, retrieval_q in enumerate(retrieval_queries)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                vector_hits, keyword_hits = future.result()
            except Exception:
                failed_indices.append(idx)
                continue
            vector_hit_batches[idx] = vector_hits
            if keyword_index is not None:
                keyword_hit_batches[idx] = keyword_hits

    # Branch-level fallback to keep behavior resilient under transient failures.
    for idx in failed_indices:
        vector_hits, keyword_hits = _search_one_query(
            retrieval_query=retrieval_queries[idx],
            query_vector=query_vectors[idx],
            vector_store=vector_store,
            keyword_index=keyword_index,
            fetch_k=fetch_k,
        )
        vector_hit_batches[idx] = vector_hits
        if keyword_index is not None:
            keyword_hit_batches[idx] = keyword_hits

    return vector_hit_batches, keyword_hit_batches


def _collect_hit_batches_serial(
    retrieval_queries: list[str],
    query_vectors: list[list[float]],
    vector_store: MilvusVectorStore,
    keyword_index: KeywordIndex | None,
    fetch_k: int,
) -> tuple[list[list[SearchHit]], list[list[SearchHit]]]:
    """Serial counterpart of retrieval batch collection."""

    vector_hit_batches: list[list[SearchHit]] = []
    keyword_hit_batches: list[list[SearchHit]] = []
    for idx, retrieval_q in enumerate(retrieval_queries):
        vector_hits, keyword_hits = _search_one_query(
            retrieval_query=retrieval_q,
            query_vector=query_vectors[idx],
            vector_store=vector_store,
            keyword_index=keyword_index,
            fetch_k=fetch_k,
        )
        vector_hit_batches.append(vector_hits)
        if keyword_index is not None:
            keyword_hit_batches.append(keyword_hits)
    return vector_hit_batches, keyword_hit_batches


def _search_one_query(
    retrieval_query: str,
    query_vector: list[float],
    vector_store: MilvusVectorStore,
    keyword_index: KeywordIndex | None,
    fetch_k: int,
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Run one retrieval branch (dense + optional keyword) for one query."""

    vector_hits = vector_store.search(query_vector=query_vector, top_k=fetch_k)
    if keyword_index is None:
        return vector_hits, []
    return vector_hits, keyword_index.search(retrieval_query, top_k=fetch_k)


def _merge_query_hits(hit_batches: Iterable[list[SearchHit]], top_k: int) -> list[SearchHit]:
    """Merge multi-query hit batches by key and keep max score."""

    merged: dict[tuple[str, int, str], SearchHit] = {}
    for batch in hit_batches:
        for hit in batch:
            key = (hit.source, hit.page, hit.text)
            current = merged.get(key)
            if current is None or hit.score > current.score:
                merged[key] = SearchHit(
                    text=hit.text,
                    source=hit.source,
                    page=hit.page,
                    score=float(hit.score),
                    doc_id=hit.doc_id,
                    file_name=hit.file_name,
                    source_type=hit.source_type,
                    company_code=hit.company_code,
                    company_name=hit.company_name,
                    report_year=hit.report_year,
                    is_table=hit.is_table,
                )

    ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return ranked[: max(1, top_k)]


def _diversify_retrieved_hits_by_company(hits: list[RetrievedHit], top_k: int) -> list[RetrievedHit]:
    """Promote company-level coverage while keeping relevance order signal.

    Strategy:
    - Keep per-company ranked lists in original order.
    - Round-robin pick one from each company bucket first.
    - Fill remaining slots by original ranking order.
    """

    if not hits:
        return []

    bucket_order: list[str] = []
    buckets: dict[str, list[RetrievedHit]] = {}
    for hit in hits:
        bucket = _company_bucket_key(hit)
        if bucket not in buckets:
            buckets[bucket] = []
            bucket_order.append(bucket)
        buckets[bucket].append(hit)

    selected: list[RetrievedHit] = []
    selected_ids: set[tuple[str, int, str]] = set()
    has_progress = True
    while has_progress and len(selected) < max(1, top_k):
        has_progress = False
        for bucket in bucket_order:
            items = buckets.get(bucket) or []
            if not items:
                continue
            candidate = items.pop(0)
            key = (candidate.source, candidate.page, candidate.text[:160])
            if key in selected_ids:
                continue
            selected_ids.add(key)
            selected.append(candidate)
            has_progress = True
            if len(selected) >= max(1, top_k):
                break

    if len(selected) < max(1, top_k):
        for hit in hits:
            key = (hit.source, hit.page, hit.text[:160])
            if key in selected_ids:
                continue
            selected_ids.add(key)
            selected.append(hit)
            if len(selected) >= max(1, top_k):
                break

    return selected


def _company_bucket_key(hit: RetrievedHit) -> str:
    """Build a stable company bucket key for diversity selection."""

    if hit.company_code:
        return f"code:{hit.company_code}"
    if hit.company_name:
        return f"name:{hit.company_name}"
    if hit.doc_id:
        return f"doc:{hit.doc_id}"
    if hit.file_name:
        return f"file:{hit.file_name}"
    return f"src:{hit.source}"


def _strip_wrappers(text: str) -> str:
    """Remove common markdown/quote wrappers from model output."""

    normalized = text.strip()
    if not normalized:
        return ""
    normalized = re.sub(r"^```(?:json|text)?", "", normalized, flags=re.IGNORECASE).strip()
    normalized = re.sub(r"```$", "", normalized).strip()
    normalized = normalized.strip("`'\"")
    return " ".join(normalized.split())


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from model output text."""

    cleaned = _strip_wrappers(text)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _merge_candidates(
    vector_hits: Iterable[SearchHit],
    keyword_hits: Iterable[SearchHit],
    vector_weight: float,
    keyword_weight: float,
    top_k: int,
) -> list[SearchHit]:
    """Merge dense and keyword candidates using normalized weighted scores."""

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
        output.append(
            SearchHit(
                text=hit.text,
                source=hit.source,
                page=hit.page,
                score=float(score),
                doc_id=hit.doc_id,
                file_name=hit.file_name,
                source_type=hit.source_type,
                company_code=hit.company_code,
                company_name=hit.company_name,
                report_year=hit.report_year,
                is_table=hit.is_table,
            )
        )
    return output


def _normalize_scores(items: Iterable[tuple[tuple[str, int, str], float]]) -> dict[tuple[str, int, str], float]:
    """Min-max normalize candidate scores into [0, 1] range."""

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
