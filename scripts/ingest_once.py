"""One-shot script: ingest source docs and rebuild retrieval artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.pipeline import IngestPipeline
from src.llm.client import OpenAIClientBundle
from src.retrieval.index_builder import RAGIndexer
from src.retrieval.vector_store import MilvusVectorStore
from src.utils.config import load_config


def main() -> None:
    """Run full rebuild or incremental upsert and print summary stats.

    Example:
        $ python3 scripts/ingest_once.py
        $ python3 scripts/ingest_once.py --file knowledge/a.pdf
    """

    parser = argparse.ArgumentParser(description="Build or upsert retrieval index")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Incrementally upsert one file (repeatable).",
    )
    args = parser.parse_args()

    config = load_config()
    ingest = IngestPipeline(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    clients = OpenAIClientBundle(config)
    store = MilvusVectorStore(uri=config.milvus_uri, collection_name=config.milvus_collection)
    indexer = RAGIndexer(ingest_pipeline=ingest, llm_clients=clients, vector_store=store)

    if args.file:
        stats = indexer.upsert_files(
            file_paths=args.file,
            processed_data_dir=config.processed_data_dir,
            raw_data_dir=config.raw_data_dir,
        )
        print(
            f"[INFO] Incremental upsert done. files={stats.file_count} chunks={stats.chunk_count} "
            f"dim={stats.embedding_dim} processed={stats.processed_file}"
        )
    else:
        stats = indexer.rebuild(config.raw_data_dir, config.processed_data_dir)
        print(
            f"[INFO] Full rebuild done. files={stats.file_count} chunks={stats.chunk_count} "
            f"dim={stats.embedding_dim} processed={stats.processed_file}"
        )


if __name__ == "__main__":
    main()
