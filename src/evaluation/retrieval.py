"""Document-level retrieval metrics for fixed cases and recorded rankings."""

from __future__ import annotations

import json
import math
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    """Read one JSON object per nonblank line, reporting malformed line numbers."""

    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _index_rows(rows: list[dict], field: str, label: str) -> dict[str, list[str]]:
    indexed = {}
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {position}: expected an object")
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{label} row {position}: id must be a nonempty string")
        if case_id in indexed:
            raise ValueError(f"{label}: duplicate id: {case_id}")
        doc_ids = row.get(field)
        if not isinstance(doc_ids, list) or any(
            not isinstance(doc_id, str) or not doc_id.strip() for doc_id in doc_ids
        ):
            raise ValueError(f"{label} id {case_id}: {field} must be a list of nonempty strings")
        indexed[case_id] = doc_ids
    return indexed


def evaluate_retrieval(cases: list[dict], results: list[dict], k: int) -> dict:
    """Compute macro Recall/MRR/nDCG@K over cases with positive document labels.

    Rankings are deduplicated by document ID before taking K documents. Cases
    with no relevant documents are reported separately and excluded from the
    positive-case means. Every case must have exactly one recorded result;
    a failed retrieval should be recorded explicitly as an empty ranking.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    labels = _index_rows(cases, "relevant_doc_ids", "cases")
    rankings = _index_rows(results, "retrieved_doc_ids", "results")
    if not labels:
        raise ValueError("cases must contain at least one case")
    unknown = rankings.keys() - labels.keys()
    missing = labels.keys() - rankings.keys()
    if unknown or missing:
        errors = []
        if unknown:
            errors.append(f"unknown result ids: {', '.join(sorted(unknown))}")
        if missing:
            errors.append(f"missing result ids: {', '.join(sorted(missing))}")
        raise ValueError("; ".join(errors))

    per_case = []
    positive_metrics = []
    no_answer_count = 0
    no_answer_empty_count = 0
    for case_id, relevant_ids in labels.items():
        relevant = set(relevant_ids)
        retrieved = list(dict.fromkeys(rankings[case_id]))[:k]
        hits = [rank for rank, doc_id in enumerate(retrieved, start=1) if doc_id in relevant]
        metrics = {"recall_at_k": None, "mrr_at_k": None, "ndcg_at_k": None}
        if relevant:
            ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
            metrics = {
                "recall_at_k": len(hits) / len(relevant),
                "mrr_at_k": 1 / hits[0] if hits else 0.0,
                "ndcg_at_k": sum(1 / math.log2(rank + 1) for rank in hits) / ideal_dcg,
            }
            positive_metrics.append(metrics)
        else:
            no_answer_count += 1
            no_answer_empty_count += int(not retrieved)
        per_case.append(
            {
                "id": case_id,
                "relevant_doc_count": len(relevant),
                "retrieved_doc_ids": retrieved,
                "hit_count": len(hits),
                **metrics,
            }
        )

    return {
        "k": k,
        "case_count": len(labels),
        "positive_case_count": len(positive_metrics),
        "no_answer_case_count": no_answer_count,
        "no_answer_empty_result_count": no_answer_empty_count,
        "metrics": {
            name: sum(row[name] for row in positive_metrics) / len(positive_metrics)
            if positive_metrics else None
            for name in ("recall_at_k", "mrr_at_k", "ndcg_at_k")
        },
        "per_case": per_case,
    }
