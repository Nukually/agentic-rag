"""indexing操作在这里"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.ingest.pipeline import ChunkRecord, IngestPipeline
from src.llm.client import OpenAIClientBundle
from src.retrieval.vector_store import MilvusVectorStore


@dataclass(frozen=True)
class IndexStats:
    """Summary stats produced by one index rebuild run."""

    file_count: int
    chunk_count: int
    embedding_dim: int
    processed_file: str


class RAGIndexer:
    """Coordinate ingestion, embedding, and vector index rebuild.

    Example:
        >>> stats = indexer.rebuild("./knowledge", "./data/processed")
        >>> stats.chunk_count
        1024
    """

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
        """Rebuild all retrieval artifacts from source files.

        Args:
            raw_data_dir: Directory containing original documents.
            processed_data_dir: Directory used to output `chunks.jsonl`.

        Returns:
            IndexStats: File/chunk/dimension summary for this rebuild.
        """

        files = self.ingest_pipeline.discover_files(raw_data_dir)
        chunks = self.ingest_pipeline.build_chunks_for_files(files, root_dir=raw_data_dir)
        if not chunks:
            raise ValueError(f"No chunks generated from directory: {raw_data_dir}")
        # 调用embedding
        embeddings = self._embed_chunks(chunks)

        dim = len(embeddings[0])
        self.vector_store.recreate(dimension=dim)# 建立索引
        self.vector_store.insert_chunks(chunks, embeddings)# 插入metadata

        output_path = self.ingest_pipeline.dump_processed(chunks, processed_data_dir)#作为关键词检索索引的数据源
        return IndexStats(
            file_count=len(files),
            chunk_count=len(chunks),
            embedding_dim=dim,
            processed_file=str(output_path),
        )

    def upsert_files(
        self,
        file_paths: list[str],
        processed_data_dir: str,
        raw_data_dir: str | None = None,
    ) -> IndexStats:
        """Incrementally upsert one or more files without full rebuild.

        Args:
            file_paths: Files to parse and upsert into indexes.
            processed_data_dir: Directory used to output `chunks.jsonl`.
            raw_data_dir: Required source root, shared with full rebuilds.

        Returns:
            IndexStats: File/chunk/dimension summary for this upsert run.
        """

        if raw_data_dir is None:
            raise ValueError("raw_data_dir is required for stable document identities")
        files = self._resolve_input_files(file_paths=file_paths, raw_data_dir=raw_data_dir)
        if not files:
            raise ValueError("No valid files to upsert")

        existing_chunks = self._load_processed(processed_data_dir)
        chunks = self.ingest_pipeline.build_chunks_for_files(files, root_dir=raw_data_dir)
        if not chunks:
            raise ValueError("No chunks generated from selected files")

        embeddings = self._embed_chunks(chunks)

        dim = len(embeddings[0])
        self.vector_store.ensure_collection(dimension=dim)

        doc_ids = sorted({chunk.doc_id for chunk in chunks if chunk.doc_id})
        self.vector_store.delete_by_doc_ids(doc_ids)
        self.vector_store.insert_chunks(chunks, embeddings)

        kept_chunks = [chunk for chunk in existing_chunks if chunk.doc_id not in doc_ids]
        output_path = self.ingest_pipeline.dump_processed(kept_chunks + chunks, processed_data_dir)
        return IndexStats(
            file_count=len(files),
            chunk_count=len(chunks),
            embedding_dim=dim,
            processed_file=str(output_path),
        )

    def _resolve_input_files(self, file_paths: list[str], raw_data_dir: str | None = None) -> list[Path]:
        """Resolve user-provided file specs into existing filesystem paths."""

        raw_root = Path(raw_data_dir).resolve() if raw_data_dir else None
        resolved: list[Path] = []
        seen: set[Path] = set()

        for raw_path in file_paths:
            if not str(raw_path).strip():
                continue
            candidate = Path(str(raw_path).strip()).expanduser()

            choices: list[Path] = []
            if candidate.is_absolute():
                choices.append(candidate)
            else:
                if raw_root is not None:
                    choices.append(raw_root / candidate)
                choices.append(candidate)

            picked: Path | None = None
            for path in choices:
                expanded = path.expanduser()
                if expanded.exists() and expanded.is_file():
                    picked = expanded
                    break
            if picked is None:
                continue

            normalized = picked.resolve()
            if normalized in seen:
                continue
            seen.add(normalized)
            if raw_root is not None and not normalized.is_relative_to(raw_root):
                raise ValueError(f"File is outside raw_data_dir: {raw_path}")
            resolved.append(normalized)

        return resolved

    def _embed_chunks(self, chunks: list[ChunkRecord]) -> list[list[float]]:
        """Validate the complete batch before replacing any indexed rows."""

        embeddings = self.llm_clients.embed_texts([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks) or not embeddings or not embeddings[0]:
            raise ValueError("Invalid embedding count or empty vectors")
        dimension = len(embeddings[0])
        if any(len(vector) != dimension or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector) for vector in embeddings):
            raise ValueError("Invalid embedding dimensions or non-finite values")
        return embeddings

    @staticmethod
    def _load_processed(output_dir: str) -> list[ChunkRecord]:
        path = Path(output_dir) / "chunks.jsonl"
        if not path.exists():
            return []
        chunks: list[ChunkRecord] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunk = ChunkRecord(**json.loads(line))
                if not chunk.doc_id.startswith("doc_") or not chunk.chunk_id or not chunk.content_hash:
                    raise ValueError("legacy chunk metadata")
                chunks.append(chunk)
        except (TypeError, ValueError) as exc:
            raise ValueError("Legacy or invalid chunk index; run a full rebuild before incremental updates") from exc
        return chunks
