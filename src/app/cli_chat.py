from __future__ import annotations

import argparse

from src.agent.graph import AgentExecutor
from src.ingest.pipeline import IngestPipeline
from src.llm.client import OpenAIClientBundle
from src.retrieval.index_builder import RAGIndexer
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.vector_store import MilvusVectorStore
from src.utils.config import load_config


def _build_index(indexer: RAGIndexer, raw_data_dir: str, processed_data_dir: str) -> None:
    stats = indexer.rebuild(raw_data_dir=raw_data_dir, processed_data_dir=processed_data_dir)
    print(
        f"[INFO] Index rebuilt. files={stats.file_count} chunks={stats.chunk_count} "
        f"dim={stats.embedding_dim} processed={stats.processed_file}"
    )


def _snippet(text: str, limit: int = 90) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "..."


def _print_agent_trace(result: object) -> None:
    traces = getattr(result, "traces", [])
    if not traces:
        print("[Agent] No tool step executed.\n")
        return

    print("[Agent] Tool Trace:")
    for step in traces:
        step_no = getattr(step, "step_no", 0)
        tool = getattr(step, "tool", "")
        tool_input = getattr(step, "tool_input", "")
        reason = getattr(step, "reason", "")
        observation = getattr(step, "observation", "")
        print(f"  [{step_no}] tool={tool} input={tool_input}")
        if reason:
            print(f"      reason: {reason}")
        print(f"      observation: {_snippet(observation, limit=140)}")
    print("")


def chat_loop() -> None:
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
    agent = AgentExecutor(
        llm_clients=clients,
        vector_store=store,
        reranker=reranker,
        top_k=config.retrieval_top_k,
        candidate_k=config.retrieval_candidate_k,
    )

    parser = argparse.ArgumentParser(description="Minimal Agentic RAG CLI")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild index from RAW_DATA_DIR before starting chat",
    )
    args = parser.parse_args()

    if args.rebuild_index:
        _build_index(indexer, config.raw_data_dir, config.processed_data_dir)

    if store.row_count() == 0:
        print("[WARN] Vector store is empty. Run /rebuild to create index.")

    print("\nRAG chat started. Commands: /rebuild /reset /tools /memory /exit")
    print(f"[Agent] Registered tools: {', '.join(agent.available_tools())}")
    history: list[dict[str, str]] = []

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exit")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("[INFO] Exit")
            break
        if user_text.lower() == "/reset":
            history = []
            agent.reset_memory()
            print("[INFO] Conversation reset")
            continue
        if user_text.lower() == "/tools":
            print(f"[INFO] Tools: {', '.join(agent.available_tools())}")
            continue
        if user_text.lower() == "/memory":
            print(f"[INFO] Memory: {agent.memory.summarize()}")
            continue
        if user_text.lower() == "/rebuild":
            _build_index(indexer, config.raw_data_dir, config.processed_data_dir)
            continue

        if store.row_count() == 0:
            print("[WARN] No index data. Run /rebuild first.")
            continue

        try:
            result = agent.run(
                question=user_text,
                history=history,
            )
        except Exception as exc:
            print(f"[ERROR] Agent execution failed: {exc}")
            continue

        _print_agent_trace(result)

        print(f"AI> {result.answer}\n")
        if result.references:
            print("References:")
            for i, hit in enumerate(result.references, start=1):
                ref_score = hit.rerank_score if hit.rerank_score is not None else hit.vector_score
                score_name = "r_score" if hit.rerank_score is not None else "v_score"
                print(f"  [ref:{i}] {hit.source} page={hit.page} {score_name}={ref_score:.4f}")
            print("")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": result.answer})
        history = history[-12:]


def main() -> None:
    chat_loop()


if __name__ == "__main__":
    main()
