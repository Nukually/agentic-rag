from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.rag_retrieve import RetrievalResult, retrieve_hits
from src.ingest.pipeline import IngestPipeline
from src.llm.client import OpenAIClientBundle
from src.llm.prompts import RAG_SYSTEM_PROMPT, build_user_prompt
from src.retrieval.index_builder import RAGIndexer
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.vector_store import MilvusVectorStore
from src.utils.config import load_config


def _snippet(text: str, limit: int = 110) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "..."


def _show_retrieval(result: RetrievalResult) -> None:
    print("\n=== 1) 向量召回结果 ===")
    if not result.vector_hits:
        print("(空)")
    for i, hit in enumerate(result.vector_hits[:8], start=1):
        print(
            f"[{i}] v_score={hit.vector_score:.4f} page={hit.page} source={hit.source}\n"
            f"    {_snippet(hit.text)}"
        )

    if result.keyword_hits:
        print("\n=== 2) 关键词召回结果 ===")
        for i, hit in enumerate(result.keyword_hits[:8], start=1):
            print(
                f"[{i}] k_score={hit.vector_score:.4f} page={hit.page} source={hit.source}\n"
                f"    {_snippet(hit.text)}"
            )

    print("\n=== 3) 重排结果 ===")
    if result.reranker_applied:
        print(f"状态: 已启用 ({result.reranker_message})")
    else:
        print(f"状态: 未启用/已降级 ({result.reranker_message})")

    if not result.final_hits:
        print("(空)")
    for i, hit in enumerate(result.final_hits, start=1):
        score_label = "h_score" if result.keyword_hits is not None else "v_score"
        rr = "-" if hit.rerank_score is None else f"{hit.rerank_score:.4f}"
        print(
            f"[{i}] r_score={rr} {score_label}={hit.vector_score:.4f} page={hit.page} source={hit.source}\n"
            f"    {_snippet(hit.text)}"
        )


def _show_answer(answer: str, result: RetrievalResult) -> None:
    print("\n=== 4) 最终回答 ===")
    print(answer)

    print("\n=== 5) 引用 ===")
    if not result.final_hits:
        print("(空)")
        return

    score_label = "h_score" if result.keyword_hits is not None else "v_score"
    for i, hit in enumerate(result.final_hits, start=1):
        rr = "-" if hit.rerank_score is None else f"{hit.rerank_score:.4f}"
        print(
            f"[ref:{i}] source={hit.source} page={hit.page} r_score={rr} {score_label}={hit.vector_score:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot RAG query with visual retrieval output")
    parser.add_argument("--question", required=True, help="Question to ask")
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
        print("[INFO] 正在重建索引...")
        stats = indexer.rebuild(config.raw_data_dir, config.processed_data_dir)
        print(
            f"[INFO] 索引完成 files={stats.file_count} chunks={stats.chunk_count} "
            f"dim={stats.embedding_dim}"
        )

    print(f"\nQuestion: {args.question}")
    keyword_index = KeywordIndex.from_processed_dir(config.processed_data_dir)
    if store.row_count() == 0 and keyword_index is None:
        raise RuntimeError("索引为空，请先执行 scripts/ingest_once.py 或使用 --rebuild-index")
    retrieval = retrieve_hits(
        query=args.question,
        llm_clients=clients,
        vector_store=store,
        reranker=reranker,
        top_k=config.retrieval_top_k,
        candidate_k=config.retrieval_candidate_k,
        keyword_index=keyword_index,
        vector_weight=config.hybrid_vector_weight,
        keyword_weight=config.hybrid_keyword_weight,
    )
    _show_retrieval(retrieval)

    contexts = [
        {"text": hit.text, "source": hit.source, "page": str(hit.page)} for hit in retrieval.final_hits
    ]
    prompt = build_user_prompt(args.question, contexts)
    answer = clients.chat(
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    _show_answer(answer, retrieval)


if __name__ == "__main__":
    main()
