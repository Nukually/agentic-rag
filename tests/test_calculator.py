"""Regression tests for arithmetic variables extracted from retrieval text."""

import pytest

from src.agent.tools.calculator import SafeCalculator


def test_current_context_overrides_remembered_variables():
    result = SafeCalculator().evaluate(
        "REVENUE + TAX",
        context_text="REVENUE=200",
        additional_variables={"REVENUE": 100, "TAX": 5, "INVALID": "invalid"},
    )

    assert result.value == 205
    assert result.variables == {"REVENUE": 200, "TAX": 5}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("REVENUE=1,000.5", {"REVENUE": 1000.5}),
        ("REVENUE：-12,345,678.90 TAX:3", {"REVENUE": -12345678.9, "TAX": 3}),
        ("REVENUE=1000.5", {"REVENUE": 1000.5}),
        ("REVENUE=100, TAX=3.", {"REVENUE": 100, "TAX": 3}),
        ("REVENUE=1,00.5", {}),
    ],
)
def test_extract_numeric_assignments(text, expected):
    assert SafeCalculator.extract_variables(text) == expected


def test_arithmetic_and_ast_allow_list_unchanged():
    calculator = SafeCalculator()

    assert calculator.evaluate("-(2 + 3) * 4 / 2").value == -10
    with pytest.raises(ValueError, match="unsupported operation"):
        calculator.evaluate("__import__('os')")


def test_empty_current_retrieval_does_not_fall_back_to_historical_text():
    from src.agent.memory import AgentMemory
    from src.agent.tools.calculate_tool import CalculateTool
    from src.agent.tools.registry import ToolContext

    context = ToolContext(
        question="new question",
        history=[],
        memory=AgentMemory(last_retrieval_text="REVENUE=100"),
        run_state={"latest_retrieval_text": ""},
        llm_clients=None,
        vector_store=None,
        reranker=None,
        keyword_index=None,
        top_k=3,
        candidate_k=5,
        hybrid_vector_weight=0.6,
        hybrid_keyword_weight=0.4,
        query_rewrite_enabled=False,
        multi_query_enabled=False,
        multi_query_count=1,
    )

    result = CalculateTool().run("REVENUE", context)

    assert result.observation == "calc_failed: unknown variable: REVENUE"
