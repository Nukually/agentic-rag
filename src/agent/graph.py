from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agent.planner import AgentPlanner, PlannedStep
from src.agent.tools.calculator import CalcResult, SafeCalculator
from src.agent.tools.rag_retrieve import RetrievalResult, RetrievedHit, retrieve_hits
from src.llm.client import OpenAIClientBundle
from src.llm.prompts import AGENT_FINAL_SYSTEM_PROMPT, build_agent_answer_prompt
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.vector_store import MilvusVectorStore


@dataclass(frozen=True)
class AgentTraceStep:
    step_no: int
    tool: str
    tool_input: str
    reason: str
    observation: str


@dataclass(frozen=True)
class AgentResult:
    answer: str
    references: list[RetrievedHit]
    traces: list[AgentTraceStep]
    reranker_applied: bool
    reranker_message: str


class AgentExecutor:
    def __init__(
        self,
        llm_clients: OpenAIClientBundle,
        vector_store: MilvusVectorStore,
        reranker: OpenAIStyleReranker,
        top_k: int,
        candidate_k: int,
        planner: AgentPlanner | None = None,
        answer_fn: Callable[[str, list[RetrievedHit], list[AgentTraceStep], list[dict[str, str]]], str] | None = None,
        retrieve_fn: Callable[[str], RetrievalResult] | None = None,
    ) -> None:
        self.llm_clients = llm_clients
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.planner = planner or AgentPlanner(llm_clients=llm_clients, max_steps=4)
        self.calculator = SafeCalculator()
        self.answer_fn = answer_fn
        self.retrieve_fn = retrieve_fn

    def run(self, question: str, history: list[dict[str, str]] | None = None) -> AgentResult:
        history = history or []

        planned_steps = self.planner.plan(question)
        traces: list[AgentTraceStep] = []
        references: list[RetrievedHit] = []

        reranker_applied = False
        reranker_message = "no retrieval"
        retrieval_context_text = ""

        for i, step in enumerate(planned_steps, start=1):
            if step.tool == "retrieve":
                retrieval = self._retrieve(step.input if step.input != "用户问题" else question)
                references = self._merge_references(references, retrieval.final_hits)
                reranker_applied = retrieval.reranker_applied
                reranker_message = retrieval.reranker_message
                retrieval_context_text = "\n".join(hit.text for hit in retrieval.final_hits)

                obs = self._format_retrieval_observation(retrieval)
                traces.append(
                    AgentTraceStep(
                        step_no=i,
                        tool=step.tool,
                        tool_input=step.input,
                        reason=step.reason,
                        observation=obs,
                    )
                )
                continue

            if step.tool == "calculate":
                try:
                    calc = self.calculator.evaluate(step.input, retrieval_context_text)
                    obs = self._format_calc_observation(calc)
                except Exception as exc:
                    obs = f"calc_failed: {exc}"

                traces.append(
                    AgentTraceStep(
                        step_no=i,
                        tool=step.tool,
                        tool_input=step.input,
                        reason=step.reason,
                        observation=obs,
                    )
                )
                continue

            if step.tool == "finish":
                traces.append(
                    AgentTraceStep(
                        step_no=i,
                        tool=step.tool,
                        tool_input=step.input,
                        reason=step.reason,
                        observation="finish requested by planner",
                    )
                )
                break

        answer = self._answer(question, references, traces, history)
        return AgentResult(
            answer=answer,
            references=references,
            traces=traces,
            reranker_applied=reranker_applied,
            reranker_message=reranker_message,
        )

    def _retrieve(self, query: str) -> RetrievalResult:
        if self.retrieve_fn is not None:
            return self.retrieve_fn(query)
        return retrieve_hits(
            query=query,
            llm_clients=self.llm_clients,
            vector_store=self.vector_store,
            reranker=self.reranker,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
        )

    def _answer(
        self,
        question: str,
        references: list[RetrievedHit],
        traces: list[AgentTraceStep],
        history: list[dict[str, str]],
    ) -> str:
        if self.answer_fn is not None:
            return self.answer_fn(question, references, traces, history)

        contexts = [
            {
                "text": hit.text,
                "source": hit.source,
                "page": str(hit.page),
            }
            for hit in references
        ]

        user_prompt = build_agent_answer_prompt(question=question, tool_traces=traces, contexts=contexts)
        messages = [{"role": "system", "content": AGENT_FINAL_SYSTEM_PROMPT}, *history]
        messages.append({"role": "user", "content": user_prompt})
        return self.llm_clients.chat(messages=messages)

    @staticmethod
    def _merge_references(current: list[RetrievedHit], incoming: list[RetrievedHit]) -> list[RetrievedHit]:
        merged = list(current)
        seen = {(item.source, item.page, item.text[:120]) for item in current}
        for hit in incoming:
            key = (hit.source, hit.page, hit.text[:120])
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
        return merged

    @staticmethod
    def _format_retrieval_observation(retrieval: RetrievalResult) -> str:
        if not retrieval.final_hits:
            return "no hits"
        parts: list[str] = []
        for i, hit in enumerate(retrieval.final_hits, start=1):
            score = hit.rerank_score if hit.rerank_score is not None else hit.vector_score
            score_name = "r_score" if hit.rerank_score is not None else "v_score"
            snippet = " ".join(hit.text.split())[:120]
            parts.append(
                f"[{i}] {hit.source} page={hit.page} {score_name}={score:.4f} text={snippet}"
            )
        return "\n".join(parts)

    @staticmethod
    def _format_calc_observation(calc: CalcResult) -> str:
        var_text = ", ".join(f"{k}={v}" for k, v in sorted(calc.variables.items()))
        if not var_text:
            var_text = "<no vars>"
        return f"expression={calc.expression}; value={calc.value}; vars={var_text}"
