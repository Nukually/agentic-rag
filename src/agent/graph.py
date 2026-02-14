from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from src.agent.memory import AgentMemory
from src.agent.planner import AgentPlanner, PlannedStep
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
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class AgentResult:
    answer: str
    references: list[RetrievedHit]
    traces: list[AgentTraceStep]
    reranker_applied: bool
    reranker_message: str
    memory_summary: str
    stage_timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ReflectionDecision:
    should_retry: bool
    reason: str
    replan_feedback: str


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
        planner_max_steps: int = 8,
        planner_history_window: int = 20,
        max_replan_retries: int = 1,
        max_answer_contexts: int = 16,
        max_answer_traces: int = 24,
        progress_callback: Callable[[str, float, str], None] | None = None,
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
        self.max_replan_retries = max(0, max_replan_retries)
        self.max_answer_contexts = max(1, max_answer_contexts)
        self.max_answer_traces = max(1, max_answer_traces)
        self.progress_callback = progress_callback
        self.planner = planner or AgentPlanner(
            llm_clients=llm_clients,
            max_steps=planner_max_steps,
            recent_history_window=planner_history_window,
        )
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
        stage_timings: dict[str, float] = {}
        run_started = perf_counter()

        route_started = perf_counter()
        route = self.planner.route_question(question)
        route_elapsed_ms = (perf_counter() - route_started) * 1000.0
        stage_timings["route"] = route_elapsed_ms
        self._emit_progress("route", route_elapsed_ms, f"route={route or 'none'}")

        traces: list[AgentTraceStep] = []
        references: list[RetrievedHit] = []

        reranker_applied = self.memory.last_reranker_applied
        reranker_message = self.memory.last_reranker_message or "no retrieval"

        run_state: dict[str, object] = {}
        last_steps: list[PlannedStep] = []
        last_observations: list[str] = []
        replan_feedback: str | None = None
        step_no = 1
        max_rounds = 1 + self.max_replan_retries

        for round_no in range(1, max_rounds + 1):
            plan_started = perf_counter()
            planned_steps = self.planner.plan(
                question=question,
                memory=self.memory,
                history=history,
                route=route,
                replan_feedback=replan_feedback,
                previous_steps=last_steps,
                previous_observations=last_observations,
            )
            plan_elapsed_ms = (perf_counter() - plan_started) * 1000.0
            plan_key = "plan" if round_no == 1 else f"replan_{round_no - 1}"
            plan_stage = "plan" if round_no == 1 else "replan"
            stage_timings[plan_key] = plan_elapsed_ms
            self._emit_progress(plan_stage, plan_elapsed_ms, f"round={round_no} steps={len(planned_steps)}")

            round_traces: list[AgentTraceStep] = []
            for i, step in enumerate(planned_steps, start=1):
                step_started = perf_counter()
                if step.tool == "finish":
                    step_elapsed_ms = (perf_counter() - step_started) * 1000.0
                    stage_timings[f"tool_r{round_no}_{i}_{step.tool}"] = step_elapsed_ms
                    trace = AgentTraceStep(
                        step_no=step_no,
                        tool=step.tool,
                        tool_input=step.input,
                        reason=step.reason,
                        observation="finish",
                        elapsed_ms=step_elapsed_ms,
                    )
                    step_no += 1
                    round_traces.append(trace)
                    traces.append(trace)
                    self._emit_progress("tool", step_elapsed_ms, f"round={round_no} step={i} tool={step.tool}")
                    break

                if not self.registry.has(step.tool):
                    step_elapsed_ms = (perf_counter() - step_started) * 1000.0
                    stage_timings[f"tool_r{round_no}_{i}_{step.tool}"] = step_elapsed_ms
                    trace = AgentTraceStep(
                        step_no=step_no,
                        tool=step.tool,
                        tool_input=step.input,
                        reason=step.reason,
                        observation=f"tool_not_registered: {step.tool}",
                        elapsed_ms=step_elapsed_ms,
                    )
                    step_no += 1
                    round_traces.append(trace)
                    traces.append(trace)
                    self._emit_progress("tool", step_elapsed_ms, f"round={round_no} step={i} tool={step.tool}")
                    continue

                try:
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
                except Exception as exc:
                    tool_output = None
                    tool_error = str(exc).strip() or exc.__class__.__name__
                    observation = f"tool_failed: {step.tool}: {tool_error[:220]}"

                    step_elapsed_ms = (perf_counter() - step_started) * 1000.0
                    stage_timings[f"tool_r{round_no}_{i}_{step.tool}"] = step_elapsed_ms
                    trace = AgentTraceStep(
                        step_no=step_no,
                        tool=step.tool,
                        tool_input=step.input,
                        reason=step.reason,
                        observation=observation,
                        elapsed_ms=step_elapsed_ms,
                    )
                    step_no += 1
                    round_traces.append(trace)
                    traces.append(trace)
                    self._emit_progress("tool", step_elapsed_ms, f"round={round_no} step={i} tool={step.tool} failed")
                    continue

                step_elapsed_ms = (perf_counter() - step_started) * 1000.0
                stage_timings[f"tool_r{round_no}_{i}_{step.tool}"] = step_elapsed_ms

                trace = AgentTraceStep(
                    step_no=step_no,
                    tool=step.tool,
                    tool_input=step.input,
                    reason=step.reason,
                    observation=tool_output.observation,
                    elapsed_ms=step_elapsed_ms,
                )
                step_no += 1
                round_traces.append(trace)
                traces.append(trace)
                self._emit_progress("tool", step_elapsed_ms, f"round={round_no} step={i} tool={step.tool}")

                references = self._merge_references(references, tool_output.references)
                self.memory.apply_delta(tool_output.memory_delta)

                if "retrieval_text" in tool_output.metadata:
                    run_state["latest_retrieval_text"] = tool_output.metadata["retrieval_text"]
                if "reranker_applied" in tool_output.metadata:
                    reranker_applied = bool(tool_output.metadata["reranker_applied"])
                if "reranker_message" in tool_output.metadata:
                    reranker_message = str(tool_output.metadata["reranker_message"])

            last_steps = planned_steps
            last_observations = [trace.observation for trace in round_traces]

            if round_no >= max_rounds:
                break

            reflect_started = perf_counter()
            reflection = self._reflect_round(
                route=route,
                planned_steps=planned_steps,
                round_traces=round_traces,
                references=references,
            )
            reflect_elapsed_ms = (perf_counter() - reflect_started) * 1000.0
            stage_timings[f"reflect_{round_no}"] = reflect_elapsed_ms
            retry_detail = "yes" if reflection.should_retry else "no"
            self._emit_progress(
                "reflect",
                reflect_elapsed_ms,
                f"round={round_no} retry={retry_detail} reason={reflection.reason}",
            )

            if not reflection.should_retry:
                break
            replan_feedback = reflection.replan_feedback

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

        answer_started = perf_counter()
        answer = self._answer(
            question=question,
            references=references,
            traces=traces,
            history=history,
            system_prompt=system_prompt,
        )
        answer_elapsed_ms = (perf_counter() - answer_started) * 1000.0
        stage_timings["answer"] = answer_elapsed_ms
        self._emit_progress("answer", answer_elapsed_ms, "final response")

        self.memory.turn_count += 1
        self.memory.last_question = question
        self.memory.last_answer = answer
        if references:
            self.memory.last_references = list(references)

        total_elapsed_ms = (perf_counter() - run_started) * 1000.0
        stage_timings["total"] = total_elapsed_ms
        self._emit_progress("total", total_elapsed_ms, "run complete")

        return AgentResult(
            answer=answer,
            references=references,
            traces=traces,
            reranker_applied=reranker_applied,
            reranker_message=reranker_message,
            memory_summary=self.memory.summarize(),
            stage_timings=stage_timings,
        )

    def _reflect_round(
        self,
        route: str | None,
        planned_steps: list[PlannedStep],
        round_traces: list[AgentTraceStep],
        references: list[RetrievedHit],
    ) -> ReflectionDecision:
        if not planned_steps:
            if route == "需要查询知识库":
                return ReflectionDecision(
                    should_retry=True,
                    reason="knowledge route but planner returned empty plan",
                    replan_feedback=(
                        "当前问题属于知识库查询，但计划为空。"
                        "请至少先规划一个 retrieve 步骤来获取依据。"
                    ),
                )
            return ReflectionDecision(
                should_retry=False,
                reason="no planned steps",
                replan_feedback="",
            )

        if any(step.tool == "finish" for step in planned_steps):
            return ReflectionDecision(
                should_retry=False,
                reason="planner requested finish",
                replan_feedback="",
            )

        tool_missing = any(
            "tool_not_registered" in step.observation or "tool_not_found" in step.observation
            for step in round_traces
        )
        if tool_missing:
            return ReflectionDecision(
                should_retry=False,
                reason="tool registry error",
                replan_feedback="",
            )

        tool_failures = [step.observation for step in round_traces if step.observation.startswith("tool_failed:")]
        if tool_failures:
            short_failure = " | ".join(tool_failures[:2])
            return ReflectionDecision(
                should_retry=True,
                reason="tool runtime error",
                replan_feedback=(
                    "上一次工具执行发生异常："
                    f"{short_failure}。"
                    "请调整步骤顺序，优先执行可获得依据的检索步骤，并避免重复失败工具调用。"
                ),
            )

        calc_failures = [
            step.observation
            for step in round_traces
            if step.tool == "calculate" and step.observation.startswith("calc_failed:")
        ]
        if calc_failures:
            unknown_variable = any("unknown variable" in item.lower() for item in calc_failures)
            had_retrieve = any(step.tool == "retrieve" for step in round_traces)
            if unknown_variable and not had_retrieve:
                return ReflectionDecision(
                    should_retry=True,
                    reason="calculate missing variables without retrieval",
                    replan_feedback=(
                        "上一次计算失败：unknown variable。"
                        "请先执行 retrieve 获取变量，再执行 calculate。"
                    ),
                )
            if unknown_variable and had_retrieve:
                return ReflectionDecision(
                    should_retry=True,
                    reason="retrieve results missing explicit variables for calculate",
                    replan_feedback=(
                        "上一次计算仍然缺少变量。"
                        "请重新检索并优先返回包含显式变量赋值（如 A=123）的片段，再执行 calculate。"
                    ),
                )

        retrieve_no_hits = any(
            step.tool == "retrieve" and step.observation.strip() == "no hits"
            for step in round_traces
        )
        if retrieve_no_hits:
            return ReflectionDecision(
                should_retry=True,
                reason="retrieve returned no hits",
                replan_feedback=(
                    "上一次 retrieve 无命中。"
                    "请重规划并优先用更贴近用户问题的检索步骤。"
                ),
            )

        if route == "需要查询知识库" and not references:
            return ReflectionDecision(
                should_retry=True,
                reason="knowledge route without references",
                replan_feedback=(
                    "当前属于知识库查询，但没有得到引用。"
                    "请优先规划 retrieve，必要时再调用其他工具。"
                ),
            )

        return ReflectionDecision(
            should_retry=False,
            reason="execution quality acceptable",
            replan_feedback="",
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
            for hit in references[: self.max_answer_contexts]
        ]

        user_prompt = build_agent_answer_prompt(
            question=question,
            tool_traces=traces[: self.max_answer_traces],
            contexts=contexts,
        )
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

    def _emit_progress(self, stage: str, elapsed_ms: float, detail: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(stage, elapsed_ms, detail)
        except Exception:
            return
