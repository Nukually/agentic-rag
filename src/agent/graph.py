from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agent.memory import AgentMemory
from src.agent.planner import AgentPlanner
from src.agent.tools.rag_retrieve import RetrievalResult, RetrievedHit
from src.agent.tools.registry import ToolContext, ToolRegistry
from src.agent.tools.retrieve_tool import RetrieveTool
from src.agent.tools.calculate_tool import CalculateTool
from src.agent.tools.budget_analyst_tool import BudgetAnalystTool
from src.llm.client import OpenAIClientBundle
from src.llm.prompts import (
    AGENT_CHITCHAT_SYSTEM_PROMPT,
    AGENT_FINAL_SYSTEM_PROMPT,
    AGENT_GENERAL_SYSTEM_PROMPT,
    build_agent_answer_prompt,
)
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.vector_store import MilvusVectorStore
from src.retrieval.keyword_index import KeywordIndex


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
    memory_summary: str


class AgentExecutor:
    def __init__(
        self,
        llm_clients: OpenAIClientBundle,
        vector_store: MilvusVectorStore,
        reranker: OpenAIStyleReranker,
        top_k: int,
        candidate_k: int,
        keyword_index: KeywordIndex | None = None,
        hybrid_vector_weight: float = 0.6,
        hybrid_keyword_weight: float = 0.4,
        planner: AgentPlanner | None = None,
        answer_fn: Callable[[str, list[RetrievedHit], list[AgentTraceStep], list[dict[str, str]]], str] | None = None,
        retrieve_fn: Callable[[str], RetrievalResult] | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.llm_clients = llm_clients
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.keyword_index = keyword_index
        self.hybrid_vector_weight = hybrid_vector_weight
        self.hybrid_keyword_weight = hybrid_keyword_weight
        self.planner = planner or AgentPlanner(llm_clients=llm_clients, max_steps=4)
        self.answer_fn = answer_fn
        self.memory = AgentMemory()

        if registry is not None:
            self.registry = registry
        else:
            self.registry = ToolRegistry()
            self.registry.register(RetrieveTool(retrieve_fn=retrieve_fn))
            self.registry.register(CalculateTool())
            self.registry.register(BudgetAnalystTool())

    def run(self, question: str, history: list[dict[str, str]] | None = None) -> AgentResult:
        history = history or []
        route = self.planner.route_question(question)
        planned_steps = self.planner.plan(
            question=question,
            memory=self.memory,
            history=history,
            route=route,
        )
        traces: list[AgentTraceStep] = []
        references: list[RetrievedHit] = []

        reranker_applied = self.memory.last_reranker_applied
        reranker_message = self.memory.last_reranker_message or "no retrieval"

        run_state: dict[str, object] = {}
        for i, step in enumerate(planned_steps, start=1):
            if not self.registry.has(step.tool):
                traces.append(
                    AgentTraceStep(
                        step_no=i,
                        tool=step.tool,
                        tool_input=step.input,
                        reason=step.reason,
                        observation=f"tool_not_registered: {step.tool}",
                    )
                )
                continue

            tool_output = self.registry.run(
                name=step.tool,
                tool_input=step.input,
                context=ToolContext(
                    question=question,
                    history=history,
                    memory=self.memory,
                    run_state=run_state,
                    llm_clients=self.llm_clients,
                    vector_store=self.vector_store,
                    reranker=self.reranker,
                    keyword_index=self.keyword_index,
                    top_k=self.top_k,
                    candidate_k=self.candidate_k,
                    hybrid_vector_weight=self.hybrid_vector_weight,
                    hybrid_keyword_weight=self.hybrid_keyword_weight,
                ),
            )

            traces.append(
                AgentTraceStep(
                    step_no=i,
                    tool=step.tool,
                    tool_input=step.input,
                    reason=step.reason,
                    observation=tool_output.observation,
                )
            )

            references = self._merge_references(references, tool_output.references)
            self.memory.apply_delta(tool_output.memory_delta)

            if "retrieval_text" in tool_output.metadata:
                run_state["latest_retrieval_text"] = tool_output.metadata["retrieval_text"]
            if "reranker_applied" in tool_output.metadata:
                reranker_applied = bool(tool_output.metadata["reranker_applied"])
            if "reranker_message" in tool_output.metadata:
                reranker_message = str(tool_output.metadata["reranker_message"])

        if not references and self.memory.last_references:
            references = list(self.memory.last_references)

        if route == "闲聊":
            system_prompt = AGENT_CHITCHAT_SYSTEM_PROMPT
        elif route == "其他":
            system_prompt = AGENT_GENERAL_SYSTEM_PROMPT
        elif route is None and not planned_steps:
            system_prompt = AGENT_GENERAL_SYSTEM_PROMPT
        else:
            system_prompt = AGENT_FINAL_SYSTEM_PROMPT

        answer = self._answer(
            question=question,
            references=references,
            traces=traces,
            history=history,
            system_prompt=system_prompt,
        )

        self.memory.turn_count += 1
        self.memory.last_question = question
        self.memory.last_answer = answer
        if references:
            self.memory.last_references = list(references)

        return AgentResult(
            answer=answer,
            references=references,
            traces=traces,
            reranker_applied=reranker_applied,
            reranker_message=reranker_message,
            memory_summary=self.memory.summarize(),
        )

    def reset_memory(self) -> None:
        self.memory.reset()

    def available_tools(self) -> list[str]:
        return self.registry.names()

    def _answer(
        self,
        question: str,
        references: list[RetrievedHit],
        traces: list[AgentTraceStep],
        history: list[dict[str, str]],
        system_prompt: str = AGENT_FINAL_SYSTEM_PROMPT,
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
        messages = [{"role": "system", "content": system_prompt}, *history]
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
