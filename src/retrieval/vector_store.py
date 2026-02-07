from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymilvus import MilvusClient

from src.ingest.pipeline import ChunkRecord


@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    page: int
    score: float


class MilvusVectorStore:
    def __init__(self, uri: str, collection_name: str) -> None:
        self.collection_name = collection_name
        self.client = MilvusClient(uri=uri)

    def has_collection(self) -> bool:
        return self.client.has_collection(collection_name=self.collection_name)

    def row_count(self) -> int:
        if not self.has_collection():
            return 0
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        raw = stats.get("row_count", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def recreate(self, dimension: int) -> None:
        if self.has_collection():
            self.client.drop_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            metric_type="COSINE",
            auto_id=True,
            consistency_level="Strong",
        )

    def insert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        data: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, embeddings):
            data.append(
                {
                    "vector": vector,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                }
            )

        if data:
            self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        if not self.has_collection():
            return []

        raw_result = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "source", "page", "chunk_index"],
        )
        if not raw_result:
            return []

        first_query_hits = raw_result[0]
        hits: list[SearchHit] = []
        for item in first_query_hits:
            entity = item.get("entity", item)
            score = float(item.get("distance", item.get("score", 0.0)))
            hits.append(
                SearchHit(
                    text=str(entity.get("text", "")),
                    source=str(entity.get("source", "")),
                    page=int(entity.get("page", 0) or 0),
                    score=score,
                )
            )

        return hits
