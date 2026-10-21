"""Offline ranking metrics and input validation, independent of external services."""

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation.retrieval import evaluate_retrieval, read_jsonl


def test_document_duplicates_are_removed_before_top_k():
    report = evaluate_retrieval(
        [{"id": "q", "relevant_doc_ids": ["a", "a", "b"]}],
        [{"id": "q", "retrieved_doc_ids": ["a", "a", "b", "c"]}],
        k=2,
    )

    assert report["metrics"] == {"recall_at_k": 1.0, "mrr_at_k": 1.0, "ndcg_at_k": 1.0}
    assert report["per_case"][0]["retrieved_doc_ids"] == ["a", "b"]
    assert report["per_case"][0]["relevant_doc_count"] == 2


def test_macro_metrics_include_empty_results_and_ignore_result_row_order():
    report = evaluate_retrieval(
        [
            {"id": "partial", "relevant_doc_ids": ["a", "b"]},
            {"id": "empty", "relevant_doc_ids": ["c"]},
        ],
        [
            {"id": "empty", "retrieved_doc_ids": []},
            {"id": "partial", "retrieved_doc_ids": ["x", "a", "b"]},
        ],
        k=2,
    )

    discount = 1 / math.log2(3)
    assert report["metrics"]["recall_at_k"] == 0.25
    assert report["metrics"]["mrr_at_k"] == 0.25
    assert report["metrics"]["ndcg_at_k"] == pytest.approx(discount / (1 + discount) / 2)
    assert report["case_count"] == report["positive_case_count"] == 2
    assert report["no_answer_case_count"] == 0


def test_no_answer_cases_are_separate_from_positive_means():
    report = evaluate_retrieval(
        [
            {"id": "positive", "relevant_doc_ids": ["a"]},
            {"id": "no-answer-empty", "relevant_doc_ids": []},
            {"id": "no-answer-nonempty", "relevant_doc_ids": []},
        ],
        [
            {"id": "positive", "retrieved_doc_ids": ["a"]},
            {"id": "no-answer-empty", "retrieved_doc_ids": []},
            {"id": "no-answer-nonempty", "retrieved_doc_ids": ["x"]},
        ],
        k=1,
    )

    assert report["positive_case_count"] == 1
    assert report["no_answer_case_count"] == 2
    assert report["no_answer_empty_result_count"] == 1
    assert report["metrics"] == {"recall_at_k": 1.0, "mrr_at_k": 1.0, "ndcg_at_k": 1.0}
    assert report["per_case"][1]["recall_at_k"] is None


def test_all_no_answer_cases_have_null_positive_metrics():
    report = evaluate_retrieval(
        [{"id": "q", "relevant_doc_ids": []}],
        [{"id": "q", "retrieved_doc_ids": []}],
        k=5,
    )

    assert report["positive_case_count"] == 0
    assert report["metrics"] == {"recall_at_k": None, "mrr_at_k": None, "ndcg_at_k": None}


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "2", None])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluate_retrieval([], [], k=k)


@pytest.mark.parametrize(
    ("cases", "results", "message"),
    [
        ([], [], "at least one case"),
        ([{"id": "q", "relevant_doc_ids": []}] * 2, [], "cases: duplicate id: q"),
        (
            [{"id": "q", "relevant_doc_ids": []}],
            [{"id": "q", "retrieved_doc_ids": []}] * 2,
            "results: duplicate id: q",
        ),
        ([{"id": "q", "relevant_doc_ids": []}], [], "missing result ids: q"),
        (
            [{"id": "q", "relevant_doc_ids": []}],
            [{"id": "q", "retrieved_doc_ids": []}, {"id": "unknown", "retrieved_doc_ids": []}],
            "unknown result ids: unknown",
        ),
    ],
)
def test_case_result_alignment_is_strict(cases, results, message):
    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(cases, results, k=2)


@pytest.mark.parametrize("invalid_id", [None, "", "  ", 1, True])
def test_case_and_result_ids_must_be_nonempty_strings(invalid_id):
    with pytest.raises(ValueError, match="id must be a nonempty string"):
        evaluate_retrieval([{"id": invalid_id, "relevant_doc_ids": []}], [], k=1)
    with pytest.raises(ValueError, match="id must be a nonempty string"):
        evaluate_retrieval(
            [{"id": "q", "relevant_doc_ids": []}],
            [{"id": invalid_id, "retrieved_doc_ids": []}],
            k=1,
        )


@pytest.mark.parametrize("invalid_docs", [None, "doc", [1], [None], [""], ["  "]])
def test_document_labels_and_rankings_are_validated(invalid_docs):
    with pytest.raises(ValueError, match="relevant_doc_ids must be a list of nonempty strings"):
        evaluate_retrieval([{"id": "q", "relevant_doc_ids": invalid_docs}], [], k=1)
    with pytest.raises(ValueError, match="retrieved_doc_ids must be a list of nonempty strings"):
        evaluate_retrieval(
            [{"id": "q", "relevant_doc_ids": []}],
            [{"id": "q", "retrieved_doc_ids": invalid_docs}],
            k=1,
        )


@pytest.mark.parametrize("field", ["relevant_doc_ids", "retrieved_doc_ids"])
def test_missing_document_list_is_rejected(field):
    cases = [{"id": "q", "relevant_doc_ids": []}]
    results = [{"id": "q", "retrieved_doc_ids": []}]
    del (cases if field == "relevant_doc_ids" else results)[0][field]
    with pytest.raises(ValueError, match=f"{field} must be a list"):
        evaluate_retrieval(cases, results, k=1)


def test_read_jsonl_reports_line_number_and_rejects_nonobjects(tmp_path):
    source = tmp_path / "cases.jsonl"
    source.write_text('\n{"id":"ok"}\n{broken}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"cases.jsonl:3: invalid JSON"):
        read_jsonl(source)

    source.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_jsonl(source)


def test_cli_runs_synthetic_fixture_without_service_configuration(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate_retrieval.py"),
            "--cases", str(root / "eval/examples/cases.jsonl"),
            "--results", str(root / "eval/examples/results.jsonl"),
            "--k", "2",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(result.stdout)
    assert report["case_count"] == 3
    assert report["positive_case_count"] == 2
    assert report["no_answer_case_count"] == report["no_answer_empty_result_count"] == 1
    assert report["metrics"]["recall_at_k"] == 0.75
    assert report["metrics"]["mrr_at_k"] == 0.75


def test_cli_validation_failure_is_nonzero_and_does_not_print_metrics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate_retrieval.py"),
            "--cases", str(root / "eval/examples/cases.jsonl"),
            "--results", str(root / "eval/examples/results.jsonl"),
            "--k", "0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "k must be a positive integer" in result.stderr
