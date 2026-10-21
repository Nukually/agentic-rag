"""Agent regressions using real tools with offline planner/retrieval fakes."""

import pytest

from src.agent.graph import AgentExecutor
from src.agent.planner import PlannedStep
from src.agent.tools.rag_retrieve import RetrievalResult, RetrievedHit


class OfflinePlanner:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.feedback = []

    def route_question(self, question):
        return "需要查询知识库"

    def plan(self, **kwargs):
        self.feedback.append(kwargs["replan_feedback"])
        return next(self.plans)


class NoExternalCalls:
    def __getattr__(self, name):
        raise AssertionError(f"Unexpected external dependency call: {name}")


def step(tool, tool_input=""):
    return PlannedStep(tool=tool, input=tool_input, reason="offline regression")


def retrieval(text="REVENUE=100", source="company-a.txt"):
    hits = [RetrievedHit(text, source, 1, 1.0, None)] if text else []
    return RetrievalResult(hits, hits, False, "disabled")


def executor(planner, retrieve_fn, retries=0):
    dependency = NoExternalCalls()
    return AgentExecutor(
        llm_clients=dependency,
        vector_store=dependency,
        reranker=dependency,
        top_k=3,
        candidate_k=5,
        planner=planner,
        retrieve_fn=retrieve_fn,
        answer_fn=lambda question, references, traces, history: "offline answer",
        max_replan_retries=retries,
    )


@pytest.mark.parametrize("failure", ["exception", "no_hits"])
def test_new_retrieval_does_not_reuse_previous_references_or_variables(failure):
    planner = OfflinePlanner([
        [step("retrieve", "company A"), step("calculate", "REVENUE")],
        [step("retrieve", "company B"), step("calculate", "REVENUE"), step("finish")],
    ])

    def retrieve(query):
        if query == "company A":
            return retrieval()
        if failure == "exception":
            raise RuntimeError("retrieval unavailable")
        return retrieval(text="")

    agent = executor(planner, retrieve)
    first = agent.run("company A revenue")
    assert first.references[0].source == "company-a.txt"
    assert agent.memory.variables["REVENUE"] == 100

    second = agent.run("company B revenue")

    assert second.references == []
    assert second.traces[1].observation.startswith("calc_failed: unknown variable: REVENUE")
    assert agent.memory.last_references == []
    assert agent.memory.last_retrieval_text == ""


def test_successful_retrieval_uses_current_values():
    planner = OfflinePlanner([
        [step("retrieve", "company A"), step("calculate", "REVENUE")],
        [step("retrieve", "company B"), step("calculate", "REVENUE")],
    ])
    agent = executor(
        planner,
        lambda query: retrieval() if query == "company A" else retrieval("REVENUE=1,000.5", "company-b.txt"),
    )
    agent.run("company A revenue")

    result = agent.run("company B revenue")

    assert [hit.source for hit in result.references] == ["company-b.txt"]
    assert agent.memory.last_calc_value == 1000.5
    assert agent.memory.variables["REVENUE"] == 1000.5


def test_calculation_followup_without_retrieval_keeps_prior_context():
    planner = OfflinePlanner([
        [step("retrieve", "company A"), step("calculate", "REVENUE")],
        [step("calculate", "LAST_RESULT * 0.1")],
        [step("calculate", "LAST_RESULT * 2")],
    ])
    agent = executor(planner, lambda query: retrieval())
    agent.run("company A revenue")

    result = agent.run("take ten percent of that result")

    assert [hit.source for hit in result.references] == ["company-a.txt"]
    assert agent.memory.last_calc_value == 10

    agent.run("double that new result")
    assert agent.memory.last_calc_value == 20


@pytest.mark.parametrize("failure", ["exception", "no_hits", "unknown_variable"])
def test_finish_does_not_hide_tool_failure_from_replanning(failure):
    first_tool = step("calculate", "MISSING") if failure == "unknown_variable" else step("retrieve", "first")
    planner = OfflinePlanner([
        [first_tool, step("finish")],
        [step("retrieve", "retry"), step("finish")],
    ])

    def retrieve(query):
        if query == "first":
            if failure == "exception":
                raise RuntimeError("retrieval unavailable")
            return retrieval(text="")
        return retrieval("REVENUE=200", "retry.txt")

    agent = executor(planner, retrieve, retries=1)

    result = agent.run("company B revenue")

    assert planner.feedback[0] is None
    assert len(planner.feedback) == 2
    assert planner.feedback[1]
    assert [hit.source for hit in result.references] == ["retry.txt"]
    assert [trace.tool for trace in result.traces][-2:] == ["retrieve", "finish"]


def test_fresh_retrieval_can_compare_against_explicit_last_result():
    planner = OfflinePlanner([
        [step("retrieve", "company A"), step("calculate", "REVENUE")],
        [step("retrieve", "company B"), step("calculate", "REVENUE - LAST_RESULT")],
    ])
    agent = executor(
        planner,
        lambda query: retrieval() if query == "company A" else retrieval("REVENUE=250", "company-b.txt"),
    )
    agent.run("company A revenue")

    result = agent.run("compare company B revenue with the prior result")

    assert [hit.source for hit in result.references] == ["company-b.txt"]
    assert agent.memory.last_calc_value == 150


def test_failed_retrieval_still_allows_explicit_last_result():
    planner = OfflinePlanner([
        [step("retrieve", "company A"), step("calculate", "REVENUE")],
        [step("retrieve", "company B"), step("calculate", "LAST_RESULT * 0.1")],
    ])

    def retrieve(query):
        if query == "company B":
            raise RuntimeError("retrieval unavailable")
        return retrieval()

    agent = executor(planner, retrieve)
    agent.run("company A revenue")

    result = agent.run("look up company B, then take ten percent of the prior result")

    assert result.references == []
    assert agent.memory.last_calc_value == 10
    assert "REVENUE" not in agent.memory.variables


def test_finish_does_not_hide_other_calculation_errors():
    planner = OfflinePlanner([
        [step("calculate", "1 / 0"), step("finish")],
        [step("calculate", "1 + 1"), step("finish")],
    ])
    agent = executor(planner, lambda query: retrieval(), retries=1)

    agent.run("calculate")

    assert len(planner.feedback) == 2
    assert planner.feedback[1]
    assert agent.memory.last_calc_value == 2


@pytest.mark.parametrize("followup_plan", [[], [step("finish")], [step("calculate", "MISSING")], [step("calculate", "2 + 2")]])
def test_independent_or_failed_plan_does_not_attach_old_references(followup_plan):
    planner = OfflinePlanner([
        [step("retrieve", "company A")],
        followup_plan,
    ])
    agent = executor(planner, lambda query: retrieval())
    agent.run("company A revenue")

    result = agent.run("an independent question")

    assert result.references == []


def test_successful_budget_followup_can_reuse_historical_evidence():
    planner = OfflinePlanner([
        [step("retrieve", "budget")],
        [step("budget_analyst", "用户问题")],
    ])
    agent = executor(planner, lambda query: retrieval("2025年预算100亿元；2026年预算120亿元"))
    agent.run("look up budgets")

    result = agent.run("analyze that budget growth")

    assert [hit.source for hit in result.references] == ["company-a.txt"]
    assert "无法评级" not in result.traces[0].observation


def test_reference_merge_keeps_chunks_with_matching_text_prefixes():
    prefix = "same text " * 20
    a = RetrievedHit(prefix + "first ending", "report.txt", 1, 1.0, None, chunk_id="first")
    b = RetrievedHit(prefix + "second ending", "report.txt", 1, 1.0, None, chunk_id="second")
    updated_a = RetrievedHit("changed display", "report.txt", 1, 1.0, None, chunk_id="first")

    assert AgentExecutor._merge_references([a], [b, updated_a]) == [a, b]


def test_reference_merge_without_ids_uses_full_text():
    prefix = "same text " * 20
    a = RetrievedHit(prefix + "first ending", "report.txt", 1, 1.0, None)
    b = RetrievedHit(prefix + "second ending", "report.txt", 1, 1.0, None)

    assert AgentExecutor._merge_references([a], [b, a]) == [a, b]


def test_chitchat_never_attaches_historical_references():
    planner = OfflinePlanner([
        [step("retrieve", "company A"), step("calculate", "REVENUE")],
        [step("calculate", "LAST_RESULT")],
    ])
    agent = executor(planner, lambda query: retrieval())
    agent.run("company A revenue")
    planner.route_question = lambda question: "闲聊"

    result = agent.run("hello")

    assert result.references == []


def test_unsuccessful_budget_analysis_does_not_attach_old_references():
    planner = OfflinePlanner([
        [step("retrieve", "company A")],
        [step("budget_analyst", "用户问题")],
    ])
    agent = executor(planner, lambda query: retrieval())
    agent.run("company A revenue")

    result = agent.run("analyze budgets without any budget data")

    assert "无法评级" in result.traces[0].observation
    assert result.references == []


def test_answer_context_includes_persistent_document_and_chunk_ids(monkeypatch):
    observed = []
    agent = executor(OfflinePlanner([]), lambda query: retrieval())
    agent.answer_fn = None

    class OfflineChat:
        def chat(self, messages):
            return "offline answer"

    agent.llm_clients = OfflineChat()
    monkeypatch.setattr(
        "src.agent.graph.build_agent_answer_prompt",
        lambda **kwargs: observed.append(kwargs["contexts"]) or "offline prompt",
    )
    hit = RetrievedHit("evidence", "report.txt", 2, 1.0, None, doc_id="doc_a", chunk_id="chunk_a")

    agent._answer("question", [hit], [], [])

    assert observed == [[{"text": "evidence", "source": "report.txt", "page": "2", "doc_id": "doc_a", "chunk_id": "chunk_a"}]]
