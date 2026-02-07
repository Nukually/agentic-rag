from __future__ import annotations

from dataclasses import dataclass

from src.llm.client import OpenAIClientBundle
from src.retrieval.reranker import OpenAIStyleReranker
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
) -> RetrievalResult:
    fetch_k = max(top_k, candidate_k)
    query_vector = llm_clients.embed_texts([query])[0]
    vector_hits_raw = vector_store.search(query_vector=query_vector, top_k=fetch_k)

    vector_hits = [_to_retrieved(hit) for hit in vector_hits_raw]
    if not vector_hits_raw:
        return RetrievalResult(
            vector_hits=[],
            final_hits=[],
            reranker_applied=False,
            reranker_message="no vector hits",
        )

    rerank_result = reranker.rerank(query=query, hits=vector_hits_raw, top_k=top_k)
    if rerank_result.applied and rerank_result.items:
        final_hits = [_to_retrieved(item.hit, item.rerank_score) for item in rerank_result.items[:top_k]]
        return RetrievalResult(
            vector_hits=vector_hits,
            final_hits=final_hits,
            reranker_applied=True,
            reranker_message=rerank_result.message,
        )

    return RetrievalResult(
        vector_hits=vector_hits,
        final_hits=vector_hits[:top_k],
        reranker_applied=False,
        reranker_message=rerank_result.message,
    )
