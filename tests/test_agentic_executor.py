from __future__ import annotations

import unittest

from src.agent.graph import AgentExecutor
from src.agent.planner import AgentPlanner, PlannedStep
from src.agent.tools.rag_retrieve import RetrievedHit, RetrievalResult


class DummyPlanner:
    def plan(self, question: str) -> list[PlannedStep]:
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


if __name__ == "__main__":
    unittest.main()
