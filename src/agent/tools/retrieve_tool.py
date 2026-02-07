from __future__ import annotations

from typing import Callable

from src.agent.tools.rag_retrieve import RetrievalResult, retrieve_hits
from src.agent.tools.registry import ToolContext, ToolOutput


class RetrieveTool:
    name = "retrieve"

    def __init__(self, retrieve_fn: Callable[[str], RetrievalResult] | None = None) -> None:
        self.retrieve_fn = retrieve_fn

    def run(self, tool_input: str, context: ToolContext) -> ToolOutput:
        query = (tool_input or "").strip()
        if not query or query == "用户问题":
            query = context.question

        if self.retrieve_fn is not None:
            retrieval = self.retrieve_fn(query)
        else:
            retrieval = retrieve_hits(
                query=query,
                llm_clients=context.llm_clients,
                vector_store=context.vector_store,
                reranker=context.reranker,
                top_k=context.top_k,
                candidate_k=context.candidate_k,
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
        if not retrieval.final_hits:
            return "no hits"

        lines: list[str] = []
        for i, hit in enumerate(retrieval.final_hits, start=1):
            score = hit.rerank_score if hit.rerank_score is not None else hit.vector_score
            score_name = "r_score" if hit.rerank_score is not None else "v_score"
            snippet = " ".join(hit.text.split())[:120]
            lines.append(f"[{i}] {hit.source} page={hit.page} {score_name}={score:.4f} text={snippet}")

        return "\n".join(lines)
