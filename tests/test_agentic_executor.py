from __future__ import annotations

import unittest

from src.agent.memory import AgentMemory
from src.agent.graph import AgentExecutor
from src.agent.planner import AgentPlanner, PlannedStep
from src.agent.tools.rag_retrieve import RetrievedHit, RetrievalResult
from src.agent.tools.registry import ToolContext, ToolOutput, ToolRegistry


class DummyPlanner:
    def plan(
        self,
        question: str,
        memory: AgentMemory | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> list[PlannedStep]:
        return [
            PlannedStep(tool="retrieve", input="AGENTIC-CASE-ALPHA-OPS-2049", reason="need vars"),
            PlannedStep(tool="calculate", input="Q1_PROFIT + Q2_PROFIT - RD_COST", reason="compute requested expression"),
        ]


class AgentExecutorTest(unittest.TestCase):
    def test_planner_extracts_symbolic_expression(self) -> None:
        q = "请根据文档计算 Q1_PROFIT + Q2_PROFIT - RD_COST 的值"
        expr = AgentPlanner._extract_symbolic_expression(q)  # type: ignore[attr-defined]
        self.assertEqual(expr, "Q1_PROFIT + Q2_PROFIT - RD_COST")

    def test_agentic_flow_with_calculation(self) -> None:
        hit = RetrievedHit(
            text="Q1_PROFIT = 137.5\nQ2_PROFIT = 262.5\nRD_COST = 80",
            source="knowledge/agentic_test_case.md",
            page=0,
            vector_score=0.99,
            rerank_score=0.98,
        )

        def fake_retrieve(query: str) -> RetrievalResult:
            return RetrievalResult(
                vector_hits=[hit],
                final_hits=[hit],
                reranker_applied=True,
                reranker_message="ok",
            )

        def fake_answer(
            question: str,
            references: list[RetrievedHit],
            traces: list[object],
            history: list[dict[str, str]],
        ) -> str:
            self.assertEqual(len(references), 1)
            self.assertEqual(len(traces), 2)
            self.assertEqual(getattr(traces[0], "tool", ""), "retrieve")
            self.assertEqual(getattr(traces[1], "tool", ""), "calculate")
            calc_obs = getattr(traces[1], "observation", "")
            self.assertIn("value=320.0", calc_obs)
            return "计算结果为 320.0"

        executor = AgentExecutor(
            llm_clients=object(),  # type: ignore[arg-type]
            vector_store=object(),  # type: ignore[arg-type]
            reranker=object(),  # type: ignore[arg-type]
            top_k=4,
            candidate_k=8,
            planner=DummyPlanner(),  # type: ignore[arg-type]
            answer_fn=fake_answer,
            retrieve_fn=fake_retrieve,
        )

        result = executor.run("请计算 Q1_PROFIT + Q2_PROFIT - RD_COST")
        self.assertEqual(result.answer, "计算结果为 320.0")
        self.assertEqual(len(result.traces), 2)

    def test_followup_reuses_memory_without_second_retrieve(self) -> None:
        hit = RetrievedHit(
            text="Q1_PROFIT = 137.5\nQ2_PROFIT = 262.5\nRD_COST = 80",
            source="knowledge/agentic_test_case.md",
            page=0,
            vector_score=0.99,
            rerank_score=0.98,
        )

        retrieve_calls: list[str] = []

        def fake_retrieve(query: str) -> RetrievalResult:
            retrieve_calls.append(query)
            return RetrievalResult(
                vector_hits=[hit],
                final_hits=[hit],
                reranker_applied=True,
                reranker_message="ok",
            )

        def fake_answer(
            question: str,
            references: list[RetrievedHit],
            traces: list[object],
            history: list[dict[str, str]],
        ) -> str:
            if traces:
                return str(getattr(traces[-1], "observation", ""))
            return "no_trace"

        executor = AgentExecutor(
            llm_clients=object(),  # type: ignore[arg-type]
            vector_store=object(),  # type: ignore[arg-type]
            reranker=object(),  # type: ignore[arg-type]
            top_k=4,
            candidate_k=8,
            answer_fn=fake_answer,
            retrieve_fn=fake_retrieve,
        )

        first = executor.run("请计算 Q1_PROFIT + Q2_PROFIT - RD_COST")
        self.assertIn("value=320.0", first.answer)

        second = executor.run("把刚才结果再加 10")
        self.assertIn("value=330.0", second.answer)
        self.assertEqual(len(retrieve_calls), 1)
        self.assertEqual(len(second.references), 1)
        self.assertTrue(all(getattr(t, "tool", "") != "retrieve" for t in second.traces))

    def test_tool_registry_custom_tool(self) -> None:
        class EchoTool:
            name = "echo"

            def run(self, tool_input: str, context: ToolContext) -> ToolOutput:
                return ToolOutput(observation=f"echo:{tool_input}")

        registry = ToolRegistry()
        registry.register(EchoTool())

        ctx = ToolContext(
            question="q",
            history=[],
            memory=AgentMemory(),
            run_state={},
            llm_clients=object(),  # type: ignore[arg-type]
            vector_store=object(),  # type: ignore[arg-type]
            reranker=object(),  # type: ignore[arg-type]
            top_k=4,
            candidate_k=8,
        )
        result = registry.run("echo", "hello", ctx)
        self.assertEqual(result.observation, "echo:hello")
        self.assertIn("echo", registry.names())


if __name__ == "__main__":
    unittest.main()
