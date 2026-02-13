from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    llm_api_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: float
    llm_temperature: float

    embedding_api_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_timeout: float
    embedding_batch_size: int

    reranker_api_url: str
    reranker_api_key: str
    reranker_model: str
    reranker_timeout: float

    milvus_uri: str
    milvus_collection: str

    raw_data_dir: str
    processed_data_dir: str

    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    retrieval_candidate_k: int
    hybrid_vector_weight: float
    hybrid_keyword_weight: float
    chat_history_max_messages: int
    planner_max_steps: int
    planner_recent_history_messages: int
    answer_max_contexts: int
    answer_max_traces: int


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def load_config() -> AppConfig:
    load_dotenv(override=True)

    return AppConfig(
        llm_api_url=_required("LLM_API_URL"),
        llm_api_key=_required("LLM_API_KEY"),
        llm_model=_required("LLM_MODEL"),
        llm_timeout=_get_float("LLM_TIMEOUT", 30.0),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.2),
        embedding_api_url=_required("EMBEDDING_API_URL"),
        embedding_api_key=_required("EMBEDDING_API_KEY"),
        embedding_model=_required("EMBEDDING_MODEL"),
        embedding_timeout=_get_float("EMBEDDING_TIMEOUT", 30.0),
        embedding_batch_size=_get_int("EMBEDDING_BATCH_SIZE", 64),
        reranker_api_url=os.getenv("RERANKER_API_URL", "").strip(),
        reranker_api_key=os.getenv("RERANKER_API_KEY", "").strip(),
        reranker_model=os.getenv("RERANKER_MODEL", "").strip(),
        reranker_timeout=_get_float("RERANKER_TIMEOUT", 30.0),
        milvus_uri=os.getenv("MILVUS_URI", "./data/index/milvus.db").strip(),
        milvus_collection=os.getenv("MILVUS_COLLECTION", "rag_chunks").strip(),
        raw_data_dir=os.getenv("RAW_DATA_DIR", "./knowledge").strip(),
        processed_data_dir=os.getenv("PROCESSED_DATA_DIR", "./data/processed").strip(),
        chunk_size=_get_int("CHUNK_SIZE", 1200),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 180),
        retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 8),
        retrieval_candidate_k=_get_int("RETRIEVAL_CANDIDATE_K", 64),
        hybrid_vector_weight=_get_float("HYBRID_VECTOR_WEIGHT", 0.6),
        hybrid_keyword_weight=_get_float("HYBRID_KEYWORD_WEIGHT", 0.4),
        chat_history_max_messages=_get_int("CHAT_HISTORY_MAX_MESSAGES", 80),
        planner_max_steps=_get_int("PLANNER_MAX_STEPS", 8),
        planner_recent_history_messages=_get_int("PLANNER_RECENT_HISTORY_MESSAGES", 20),
        answer_max_contexts=_get_int("ANSWER_MAX_CONTEXTS", 16),
        answer_max_traces=_get_int("ANSWER_MAX_TRACES", 24),
    )
