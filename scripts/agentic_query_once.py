from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.graph import AgentExecutor
from src.ingest.pipeline import IngestPipeline
from src.llm.client import OpenAIClientBundle
from src.retrieval.index_builder import RAGIndexer
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.vector_store import MilvusVectorStore
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot agentic RAG query")
    parser.add_argument(
        "--question",
        default="请根据 AGENTIC-CASE-ALPHA-OPS-2049 文档，计算 Q1_PROFIT + Q2_PROFIT - RD_COST 的值，并说明依据。",
        help="Question to ask",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild index before querying")
    args = parser.parse_args()

    config = load_config()
    clients = OpenAIClientBundle(config)
    reranker = OpenAIStyleReranker(
        api_url=config.reranker_api_url,
        api_key=config.reranker_api_key,
        model=config.reranker_model,
        timeout=config.reranker_timeout,
    )
    ingest = IngestPipeline(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    store = MilvusVectorStore(uri=config.milvus_uri, collection_name=config.milvus_collection)
    indexer = RAGIndexer(ingest_pipeline=ingest, llm_clients=clients, vector_store=store)

    if args.rebuild_index:
        print("[INFO] Rebuilding index...")
        stats = indexer.rebuild(config.raw_data_dir, config.processed_data_dir)
        print(f"[INFO] done files={stats.file_count} chunks={stats.chunk_count} dim={stats.embedding_dim}")

    keyword_index = KeywordIndex.from_processed_dir(config.processed_data_dir)
    if store.row_count() == 0 and keyword_index is None:
        raise RuntimeError("Index is empty; run scripts/ingest_once.py or use --rebuild-index")
    agent = AgentExecutor(
        llm_clients=clients,
        vector_store=store,
        reranker=reranker,
        top_k=config.retrieval_top_k,
        candidate_k=config.retrieval_candidate_k,
        keyword_index=keyword_index,
        hybrid_vector_weight=config.hybrid_vector_weight,
        hybrid_keyword_weight=config.hybrid_keyword_weight,
    )

    print(f"\nQuestion: {args.question}\n")
    print(f"[INFO] Registered tools: {', '.join(agent.available_tools())}\n")
    result = agent.run(args.question)

    print("=== Agent Trace ===")
    for step in result.traces:
        print(f"[{step.step_no}] tool={step.tool}")
        print(f"    input: {step.tool_input}")
        if step.reason:
            print(f"    reason: {step.reason}")
        print(f"    observation: {step.observation}\n")

    print("=== Answer ===")
    print(result.answer)
    print("\n=== References ===")
    if not result.references:
        print("(none)")
    for i, hit in enumerate(result.references, start=1):
        score = hit.rerank_score if hit.rerank_score is not None else hit.vector_score
        if hit.rerank_score is not None:
            score_name = "r_score"
        else:
            score_name = "h_score" if keyword_index is not None else "v_score"
        print(f"[ref:{i}] {hit.source} page={hit.page} {score_name}={score:.4f}")

    print("\n=== Memory Summary ===")
    print(result.memory_summary)


if __name__ == "__main__":
    main()
