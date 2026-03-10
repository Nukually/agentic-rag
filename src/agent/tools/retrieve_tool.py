"""Agent tool wrapper for retrieval and memory update behavior."""

from __future__ import annotations

import re
from typing import Callable

from src.agent.tools.rag_retrieve import RetrievalResult, retrieve_hits
from src.agent.tools.registry import ToolContext, ToolOutput


class RetrieveTool:
    """Retrieve tool that writes hits back to memory and references.

    Example:
        Tool input can be explicit query text or `"用户问题"` placeholder.
    """

    name = "retrieve"

    def __init__(self, retrieve_fn: Callable[[str], RetrievalResult] | None = None) -> None:
        """Initialize tool with optional custom retrieval implementation."""

        self.retrieve_fn = retrieve_fn

    def run(self, tool_input: str, context: ToolContext) -> ToolOutput:
        """Execute retrieval and return normalized tool output.

        Args:
            tool_input: Query text or placeholder.
            context: Tool execution context from agent runtime.

        Returns:
            ToolOutput: Observation text, references, and memory delta.
        """

        query = (tool_input or "").strip()
        if not query or query == "用户问题":
            query = context.question

        if self.retrieve_fn is not None:
            retrieval = self.retrieve_fn(query)
        else:
            effective_top_k, effective_candidate_k = self._effective_retrieval_scope(
                query=query,
                base_top_k=context.top_k,
                base_candidate_k=context.candidate_k,
            )
            coverage_mode = self._is_coverage_query(query)
            retrieval = retrieve_hits(
                query=query,
                llm_clients=context.llm_clients,
                vector_store=context.vector_store,
                reranker=context.reranker,
                top_k=effective_top_k,
                candidate_k=effective_candidate_k,
                keyword_index=context.keyword_index,
                vector_weight=context.hybrid_vector_weight,
                keyword_weight=context.hybrid_keyword_weight,
                query_rewrite_enabled=context.query_rewrite_enabled,
                multi_query_enabled=context.multi_query_enabled,
                multi_query_count=context.multi_query_count,
                diversify_by_company=coverage_mode,
            )

        observation = self._format_observation(retrieval)
        retrieval_text = "\n".join(hit.text for hit in retrieval.final_hits)

        return ToolOutput(
            observation=observation,
            references=list(retrieval.final_hits),
            memory_delta={
                "last_retrieval_query": query,
                "last_retrieval_text": retrieval_text,
                "last_references": list(retrieval.final_hits),
                "last_reranker_applied": retrieval.reranker_applied,
                "last_reranker_message": retrieval.reranker_message,
                "tool_observations": {"retrieve": observation},
            },
            metadata={
                "reranker_applied": retrieval.reranker_applied,
                "reranker_message": retrieval.reranker_message,
                "retrieval_text": retrieval_text,
            },
        )

    @staticmethod
    def _format_observation(retrieval: RetrievalResult) -> str:
        """Format retrieval result into planner-friendly observation text."""

        if not retrieval.final_hits:
            return "no hits"

        lines: list[str] = []
        if retrieval.retrieval_query:
            query_count = len(retrieval.retrieval_queries or [])
            lines.append(f"query={retrieval.retrieval_query} query_count={max(1, query_count)}")
        companies = sorted(
            {
                (hit.company_code or hit.company_name).strip()
                for hit in retrieval.final_hits
                if (hit.company_code or hit.company_name).strip()
            }
        )
        if companies:
            lines.append(f"companies_covered={len(companies)}")
        for i, hit in enumerate(retrieval.final_hits, start=1):
            score = hit.rerank_score if hit.rerank_score is not None else hit.vector_score
            if hit.rerank_score is not None:
                score_name = "r_score"
            else:
                score_name = "h_score" if retrieval.keyword_hits is not None else "v_score"
            snippet = " ".join(hit.text.split())[:120]
            company_marker = hit.company_code or hit.company_name
            if company_marker:
                lines.append(
                    f"[{i}] {hit.source} page={hit.page} company={company_marker} "
                    f"{score_name}={score:.4f} text={snippet}"
                )
            else:
                lines.append(f"[{i}] {hit.source} page={hit.page} {score_name}={score:.4f} text={snippet}")

        return "\n".join(lines)

    @staticmethod
    def _effective_retrieval_scope(query: str, base_top_k: int, base_candidate_k: int) -> tuple[int, int]:
        """Adjust retrieval scope for coverage-heavy queries.

        For prompts asking to cover all companies/entities, expand retrieval
        breadth to reduce omission risk in final references.
        """

        top_k = max(1, int(base_top_k))
        candidate_k = max(top_k, int(base_candidate_k))
        normalized = " ".join((query or "").strip().lower().split())
        if not normalized:
            return top_k, candidate_k

        coverage_patterns = (
            r"所有公司",
            r"全部公司",
            r"每家(公司|企业)",
            r"按公司逐一",
            r"不要遗漏",
            r"避免遗漏",
            r"补充遗漏",
            r"漏(掉|下).*(公司|企业)",
            r"all companies",
            r"all firms",
            r"complete coverage",
        )
        if any(re.search(pattern, normalized) for pattern in coverage_patterns):
            expanded_top_k = min(max(top_k, 16), 32)
            expanded_candidate_k = min(max(candidate_k, expanded_top_k * 8), 256)
            return expanded_top_k, expanded_candidate_k

        if RetrieveTool._is_coverage_query(normalized):
            expanded_top_k = min(max(top_k, 16), 32)
            expanded_candidate_k = min(max(candidate_k, expanded_top_k * 8), 256)
            return expanded_top_k, expanded_candidate_k

        return top_k, candidate_k

    @staticmethod
    def _is_coverage_query(query: str) -> bool:
        """Return whether query indicates global-coverage intent."""

        normalized = " ".join((query or "").strip().lower().split())
        if not normalized:
            return False
        coverage_patterns = (
            r"所有公司",
            r"全部公司",
            r"每家(公司|企业)",
            r"按公司逐一",
            r"不要遗漏",
            r"避免遗漏",
            r"补充遗漏",
            r"漏(掉|下).*(公司|企业)",
            r"all companies",
            r"all firms",
            r"complete coverage",
        )
        if any(re.search(pattern, normalized) for pattern in coverage_patterns):
            return True

        # Implicit broad asks: report-level reasoning without specific company target.
        has_broad_intent = bool(re.search(r"(根据报告|根据财报|根据年报).*(说明|分析|总结|原因|变化)", normalized))
        has_explicit_company = bool(
            re.search(r"(?<!\d)\d{6}(?!\d)", normalized)
            or re.search(r"(某某公司|xx公司|指定公司)", normalized)
        )
        return has_broad_intent and not has_explicit_company
