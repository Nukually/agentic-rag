"""Score recorded document rankings offline; no model or database is called."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval import evaluate_retrieval, read_jsonl


def main() -> None:
    """Print document-level metrics; bundled examples are synthetic fixtures."""

    parser = argparse.ArgumentParser(description="Evaluate recorded document rankings offline")
    parser.add_argument("--cases", required=True, type=Path, help="JSONL: id, relevant_doc_ids")
    parser.add_argument("--results", required=True, type=Path, help="JSONL: id, retrieved_doc_ids")
    parser.add_argument("--k", type=int, default=5, help="Unique documents to score (default: 5)")
    args = parser.parse_args()
    try:
        report = evaluate_retrieval(read_jsonl(args.cases), read_jsonl(args.results), args.k)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
