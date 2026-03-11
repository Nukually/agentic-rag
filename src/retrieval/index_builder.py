"""indexing操作在这里"""

from __future__ import annotations

import json
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
        chunks = self.ingest_pipeline.build_chunks_for_files(files)
        if not chunks:
            raise ValueError(f"No chunks generated from directory: {raw_data_dir}")
        # 调用embedding
        embeddings = self.llm_clients.embed_texts([chunk.text for chunk in chunks])
        if not embeddings:
            raise ValueError("Embedding API returned empty vectors")

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
            raw_data_dir: Optional raw root used to resolve relative paths.

        Returns:
            IndexStats: File/chunk/dimension summary for this upsert run.
        """

        files = self._resolve_input_files(file_paths=file_paths, raw_data_dir=raw_data_dir)
        if not files:
            raise ValueError("No valid files to upsert")

        chunks = self.ingest_pipeline.build_chunks_for_files(files)
        if not chunks:
            raise ValueError("No chunks generated from selected files")

        embeddings = self.llm_clients.embed_texts([chunk.text for chunk in chunks])
        if not embeddings:
            raise ValueError("Embedding API returned empty vectors")

        dim = len(embeddings[0])
        self.vector_store.ensure_collection(dimension=dim)

        doc_ids = sorted({chunk.doc_id for chunk in chunks if chunk.doc_id})
        self.vector_store.delete_by_doc_ids(doc_ids)
        self.vector_store.insert_chunks(chunks, embeddings)

        output_path = self._upsert_processed(chunks=chunks, output_dir=processed_data_dir, remove_doc_ids=doc_ids)
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
            candidate = Path(str(raw_path).strip())
            if not candidate:
                continue

            choices: list[Path] = []
            if candidate.is_absolute():
                choices.append(candidate)
            else:
                choices.append(candidate)
                if raw_root is not None:
                    choices.append(raw_root / candidate)

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
            resolved.append(picked)

        return resolved

    def _upsert_processed(self, chunks: list[ChunkRecord], output_dir: str, remove_doc_ids: list[str]) -> Path:
        """Upsert chunks into `chunks.jsonl` by replacing rows with same `doc_id`."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_path / "chunks.jsonl"

        remove_ids = {doc_id for doc_id in remove_doc_ids if doc_id}
        existing_rows: list[dict] = []
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    doc_id = str(row.get("doc_id", ""))
                    if doc_id and doc_id in remove_ids:
                        continue
                    existing_rows.append(row)

        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in existing_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for chunk in chunks:
                payload = {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "doc_id": chunk.doc_id,
                    "file_name": chunk.file_name,
                    "source_type": chunk.source_type,
                    "company_code": chunk.company_code,
                    "company_name": chunk.company_name,
                    "report_year": chunk.report_year,
                    "is_table": chunk.is_table,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return jsonl_path
