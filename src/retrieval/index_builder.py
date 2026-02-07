from __future__ import annotations

from dataclasses import dataclass

from src.ingest.pipeline import IngestPipeline
from src.llm.client import OpenAIClientBundle
from src.retrieval.vector_store import MilvusVectorStore


@dataclass(frozen=True)
class IndexStats:
    file_count: int
    chunk_count: int
    embedding_dim: int
    processed_file: str


class RAGIndexer:
    def __init__(
        self,
        ingest_pipeline: IngestPipeline,
        llm_clients: OpenAIClientBundle,
        vector_store: MilvusVectorStore,
    ) -> None:
        self.ingest_pipeline = ingest_pipeline
        self.llm_clients = llm_clients
        self.vector_store = vector_store

    def rebuild(self, raw_data_dir: str, processed_data_dir: str) -> IndexStats:
        files = self.ingest_pipeline.discover_files(raw_data_dir)
        chunks = self.ingest_pipeline.build_chunks(raw_data_dir)
        if not chunks:
            raise ValueError(f"No chunks generated from directory: {raw_data_dir}")

        embeddings = self.llm_clients.embed_texts([chunk.text for chunk in chunks])
        if not embeddings:
            raise ValueError("Embedding API returned empty vectors")

        dim = len(embeddings[0])
        self.vector_store.recreate(dimension=dim)
        self.vector_store.insert_chunks(chunks, embeddings)

        output_path = self.ingest_pipeline.dump_processed(chunks, processed_data_dir)
        return IndexStats(
            file_count=len(files),
            chunk_count=len(chunks),
            embedding_dim=dim,
            processed_file=str(output_path),
        )
