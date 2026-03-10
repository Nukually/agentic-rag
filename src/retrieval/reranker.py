"""OpenAI-style reranker client with graceful fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from src.retrieval.vector_store import SearchHit


@dataclass(frozen=True)
class RerankItem:
    """One reranked hit with its relevance score."""

    hit: SearchHit
    rerank_score: float


@dataclass(frozen=True)
class RerankResult:
    """Rerank execution result including status and ranked items."""

    applied: bool
    message: str
    items: list[RerankItem]


class OpenAIStyleReranker:
    """Call a reranker API compatible with OpenAI-like request/response shape.

    Example:
        >>> result = reranker.rerank(query="profit", hits=candidates, top_k=8)
        >>> result.applied
        True
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        timeout: float,
    ) -> None:
        self.api_url = api_url.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = timeout

    def enabled(self) -> bool:
        """Return whether reranker configuration is complete."""

        return bool(self.api_url and self.api_key and self.model)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> RerankResult:
        """Rerank candidate hits and return sorted output.

        Args:
            query: User query.
            hits: Candidate documents to rerank.
            top_k: Max number of ranked items requested.

        Returns:
            RerankResult: Applied state, message, and ranked items.
        """

        if not hits:
            return RerankResult(applied=False, message="no candidate hits", items=[])
        if not self.enabled():
            return RerankResult(applied=False, message="reranker not configured", items=[])

        payload = {
            "model": self.model,
            "query": query,
            "documents": [h.text for h in hits],
            "top_n": min(max(top_k, 1), len(hits)),
            "return_documents": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoints = self._build_endpoints(self.api_url)
        last_error = "unknown rerank error"

        for endpoint in endpoints:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(endpoint, headers=headers, json=payload)
                if resp.status_code >= 400:
                    last_error = f"{endpoint} status={resp.status_code} body={resp.text[:180]}"
                    continue

                data = resp.json()
                pairs = self._parse_pairs(data, len(hits))
                if not pairs:
                    last_error = f"{endpoint} returned no rank pairs"
                    continue

                ranked_items: list[RerankItem] = []
                used: set[int] = set()
                for idx, score in sorted(pairs, key=lambda x: x[1], reverse=True):
                    if idx in used:
                        continue
                    if 0 <= idx < len(hits):
                        ranked_items.append(RerankItem(hit=hits[idx], rerank_score=score))
                        used.add(idx)

                if not ranked_items:
                    last_error = f"{endpoint} produced invalid indices"
                    continue

                return RerankResult(applied=True, message=f"reranked via {endpoint}", items=ranked_items)
            except Exception as exc:
                last_error = f"{endpoint} exception={exc}"

        return RerankResult(applied=False, message=last_error, items=[])

    @staticmethod
    def _build_endpoints(base_url: str) -> list[str]:
        """Generate candidate reranker endpoints from a base URL."""

        endpoints = [f"{base_url}/rerank"]
        if not base_url.endswith("/v1"):
            endpoints.insert(0, f"{base_url}/v1/rerank")

        # De-duplicate while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for item in endpoints:
            if item not in seen:
                uniq.append(item)
                seen.add(item)
        return uniq

    @staticmethod
    def _parse_pairs(payload: Any, total_docs: int) -> list[tuple[int, float]]:
        """Parse rank-index and score pairs from heterogeneous response shapes."""

        items = OpenAIStyleReranker._extract_list(payload)
        pairs: list[tuple[int, float]] = []

        for pos, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            raw_idx = item.get("index")
            if raw_idx is None:
                raw_idx = item.get("document_index")
            if raw_idx is None:
                raw_idx = item.get("id")
            if raw_idx is None:
                raw_idx = pos

            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= total_docs:
                continue

            score_raw = item.get("relevance_score")
            if score_raw is None:
                score_raw = item.get("score")
            if score_raw is None:
                score_raw = item.get("similarity")
            try:
                score = float(score_raw if score_raw is not None else 0.0)
            except (TypeError, ValueError):
                score = 0.0

            pairs.append((idx, score))

        return pairs

    @staticmethod
    def _extract_list(payload: Any) -> list[Any]:
        """Extract a result list from known response keys."""

        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []

        for key in ("results", "data", "output"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        return []
