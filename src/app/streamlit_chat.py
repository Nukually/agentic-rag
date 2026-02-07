from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

from src.agent.graph import AgentExecutor
from src.ingest.pipeline import IngestPipeline
from src.llm.client import OpenAIClientBundle
from src.retrieval.index_builder import RAGIndexer
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.vector_store import MilvusVectorStore
from src.utils.config import load_config


@dataclass
class AppRuntime:
    agent: AgentExecutor
    indexer: RAGIndexer
    store: MilvusVectorStore
    config: Any


def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

:root {
  --brand-ink: #12343b;
  --brand-accent: #1f7a8c;
  --brand-soft: #e8f3f5;
  --brand-card: #f9fcfd;
  --brand-border: #d7e7eb;
}

html, body, [class*="css"]  {
  font-family: 'Manrope', sans-serif;
}

.main .block-container {
  max-width: 920px;
  padding-top: 1.6rem;
  padding-bottom: 2rem;
}

.agent-header {
  background: linear-gradient(120deg, #eff7f9 0%, #f9fcfd 100%);
  border: 1px solid var(--brand-border);
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 14px;
}

.agent-title {
  color: var(--brand-ink);
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 3px;
}

.agent-subtitle {
  color: #365860;
  font-size: 0.93rem;
}

.meta-chip {
  display: inline-block;
  border: 1px solid var(--brand-border);
  background: var(--brand-soft);
  color: var(--brand-ink);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.78rem;
  margin-right: 6px;
}

.trace-box {
  border: 1px solid var(--brand-border);
  border-radius: 10px;
  background: var(--brand-card);
  padding: 8px 10px;
  margin: 7px 0;
}

.trace-tool {
  color: var(--brand-ink);
  font-weight: 700;
}

.trace-reason {
  color: #4a6670;
  font-size: 0.86rem;
}

.ref-item {
  color: #20464f;
  font-size: 0.88rem;
  margin: 3px 0;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _init_runtime() -> AppRuntime:
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
    return AppRuntime(agent=agent, indexer=indexer, store=store, config=config)


def _runtime() -> AppRuntime:
    if "runtime" not in st.session_state:
        st.session_state.runtime = _init_runtime()
    return st.session_state.runtime


def _messages() -> list[dict[str, Any]]:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages


def _chat_history_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        history.append({"role": role, "content": str(msg.get("content", ""))})
    return history


def _render_assistant_extras(msg: dict[str, Any], idx: int) -> None:
    traces = msg.get("traces") or []
    refs = msg.get("references") or []
    memory_summary = msg.get("memory_summary", "")

    with st.expander(f"工具轨迹与引用 #{idx + 1}", expanded=False):
        if traces:
            st.markdown("**Tool Trace**")
            for step in traces:
                st.markdown(
                    (
                        "<div class='trace-box'>"
                        f"<div><span class='trace-tool'>[{step.get('step_no', '?')}] {step.get('tool', '')}</span>"
                        f"  input: <code>{step.get('tool_input', '')}</code></div>"
                        f"<div class='trace-reason'>reason: {step.get('reason', '')}</div>"
                        f"<div class='trace-reason'>obs: {step.get('observation', '')}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.caption("本轮没有工具调用。")

        st.markdown("**References**")
        if refs:
            for ref in refs:
                st.markdown(
                    f"<div class='ref-item'>- {ref}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("无引用")

        if memory_summary:
            st.markdown("**Memory**")
            st.code(memory_summary)


def _render_messages(messages: list[dict[str, Any]]) -> None:
    for idx, msg in enumerate(messages):
        role = msg.get("role", "assistant")
        content = str(msg.get("content", ""))
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(content)
            if role == "assistant":
                _render_assistant_extras(msg, idx)


def _run_turn(question: str) -> None:
    runtime = _runtime()
    messages = _messages()

    messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在规划并执行..."):
            history = _chat_history_from_messages(messages[:-1])
            result = runtime.agent.run(question=question, history=history)

        st.markdown(result.answer)

        refs = []
        for hit in result.references:
            score = hit.rerank_score if hit.rerank_score is not None else hit.vector_score
            score_name = "r_score" if hit.rerank_score is not None else "v_score"
            refs.append(f"{hit.source} page={hit.page} {score_name}={score:.4f}")

        traces = [
            {
                "step_no": t.step_no,
                "tool": t.tool,
                "tool_input": t.tool_input,
                "reason": t.reason,
                "observation": t.observation,
            }
            for t in result.traces
        ]

        msg = {
            "role": "assistant",
            "content": result.answer,
            "traces": traces,
            "references": refs,
            "memory_summary": result.memory_summary,
        }
        messages.append(msg)

        _render_assistant_extras(msg, len(messages) - 1)


def main() -> None:
    st.set_page_config(page_title="Agentic RAG Chat", page_icon="🧭", layout="centered")
    _inject_css()

    runtime = _runtime()
    messages = _messages()

    st.markdown(
        """
<div class="agent-header">
  <div class="agent-title">Agentic RAG Web Chat</div>
  <div class="agent-subtitle">多轮记忆 + 工具轨迹 + 引用可追溯</div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f"<span class='meta-chip'>tools: {', '.join(runtime.agent.available_tools())}</span>"
            f"<span class='meta-chip'>collection: {runtime.config.milvus_collection}</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<span class='meta-chip'>rows: {runtime.store.row_count()}</span>",
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.subheader("操作")

        if st.button("重建索引", use_container_width=True):
            with st.spinner("正在重建索引..."):
                stats = runtime.indexer.rebuild(runtime.config.raw_data_dir, runtime.config.processed_data_dir)
            st.success(
                f"完成：files={stats.file_count}, chunks={stats.chunk_count}, dim={stats.embedding_dim}"
            )

        if st.button("清空会话", use_container_width=True):
            st.session_state.messages = []
            runtime.agent.reset_memory()
            st.success("会话与 memory 已清空")

        if st.button("仅重置 Memory", use_container_width=True):
            runtime.agent.reset_memory()
            st.success("memory 已重置")

        st.divider()
        st.caption("Memory Snapshot")
        st.code(runtime.agent.memory.summarize())

    _render_messages(messages)

    question = st.chat_input("输入问题，例如：请根据 AGENTIC-CASE-ALPHA-OPS-2049 文档计算 Q1_PROFIT + Q2_PROFIT - RD_COST")
    if question:
        _run_turn(question)
        st.rerun()


if __name__ == "__main__":
    main()
