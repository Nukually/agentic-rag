"""封装所有向量数据库的操作"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymilvus import MilvusClient

from src.ingest.pipeline import ChunkRecord


@dataclass(frozen=True)
class SearchHit:
    """One retrieval hit returned from vector/keyword search."""

    text: str
    source: str
    page: int
    score: float
    doc_id: str = ""
    file_name: str = ""
    source_type: str = ""
    company_code: str = ""
    company_name: str = ""
    report_year: int | None = None
    is_table: bool = False


class MilvusVectorStore:
    """Thin wrapper around `MilvusClient` for project-specific operations.

    Args:
        uri: Milvus connection URI. Defaults to local sqlite-backed URI in
            config.
        collection_name: Logical collection used for chunk vectors.
    """

    def __init__(
        self,
        uri: str,
        collection_name: str,
        ivf_nlist: int = 1024,
        ivf_nprobe: int = 32,
    ) -> None:
        self.collection_name = collection_name
        self.client = MilvusClient(uri=uri)
        self.ivf_nlist = max(1, int(ivf_nlist))
        self.ivf_nprobe = max(1, int(ivf_nprobe))
        self.index_type_in_use = "AUTOINDEX"

    def has_collection(self) -> bool:
        """Check whether the target collection exists."""

        return self.client.has_collection(collection_name=self.collection_name)

    def row_count(self) -> int:
        """Return number of rows in collection, or 0 if missing."""

        if not self.has_collection():
            return 0
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        raw = stats.get("row_count", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def recreate(self, dimension: int) -> None:
        """用新的向量维度重建索引"""

        if self.has_collection():
            self.client.drop_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            metric_type="COSINE",
            auto_id=True,
            consistency_level="Strong",
        )
        self._ensure_supported_index()

    def insert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """插入chunk的metadata到数据库里"""

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
                    "doc_id": chunk.doc_id,
                    "file_name": chunk.file_name,
                    "source_type": chunk.source_type,
                    "company_code": chunk.company_code,
                    "company_name": chunk.company_name,
                    "report_year": chunk.report_year,
                    "is_table": chunk.is_table,
                }
            )

        if data:
            self.client.insert(collection_name=self.collection_name, data=data)

    def ensure_collection(self, dimension: int) -> None:
        """Create collection if absent; keep existing collection untouched."""

        if self.has_collection():
            self._ensure_supported_index()
            return
        self.recreate(dimension=dimension)

    def delete_by_doc_ids(self, doc_ids: list[str]) -> int:
        """Delete rows by `doc_id` metadata, returning deleted row count."""

        unique_ids = sorted({doc_id.strip() for doc_id in doc_ids if doc_id and doc_id.strip()})
        if not unique_ids or not self.has_collection():
            return 0

        deleted = 0
        for doc_id in unique_ids:
            filter_value = _escape_filter_value(doc_id)
            try:
                result = self.client.delete(
                    collection_name=self.collection_name,
                    filter=f'doc_id == "{filter_value}"',
                )
            except Exception:
                continue
            deleted += _extract_delete_count(result)
        return deleted

    def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        """Search nearest chunks by vector similarity.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of hits.

        Returns:
            list[SearchHit]: Ranked search hits with text/source/page metadata.
        """

        if not self.has_collection():
            return []

        output_fields = [
            "text",
            "source",
            "page",
            "chunk_index",
            "doc_id",
            "file_name",
            "source_type",
            "company_code",
            "company_name",
            "report_year",
            "is_table",
        ]
        search_params = self._build_search_params()
        try:
            if search_params is not None:
                raw_result = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=top_k,
                    search_params=search_params,
                    output_fields=output_fields,
                )
            else:
                raw_result = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=output_fields,
                )
        except TypeError:
            # Backward compatibility for older pymilvus versions without `search_params`.
            try:
                raw_result = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=output_fields,
                )
            except Exception:
                # Backward compatibility for older collections without new metadata fields.
                raw_result = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "source", "page", "chunk_index"],
                )
        except Exception:
            # Backward compatibility for older collections without new metadata fields.
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
                    doc_id=str(entity.get("doc_id", "")),
                    file_name=str(entity.get("file_name", "")),
                    source_type=str(entity.get("source_type", "")),
                    company_code=str(entity.get("company_code", "")),
                    company_name=str(entity.get("company_name", "")),
                    report_year=_to_int_or_none(entity.get("report_year")),
                    is_table=bool(entity.get("is_table", False)),
                )
            )

        return hits

    def _ensure_supported_index(self) -> None:
        """建立索引：

        Priority:
        1) IVF_FLAT (tunable by nlist/nprobe)
        2) AUTOINDEX
        3) FLAT
        """

        candidates: list[tuple[str, dict[str, int]]] = [
            ("IVF_FLAT", {"nlist": self.ivf_nlist}),
            ("AUTOINDEX", {}),
            ("FLAT", {}),
        ]
        for index_type, params in candidates:
            if self._try_create_index(index_type=index_type, params=params):
                self.index_type_in_use = index_type
                return

    def _try_create_index(self, index_type: str, params: dict[str, int]) -> bool:
        """Try to create one index type with cross-version compatibility."""

        index_params = {
            "metric_type": "COSINE",
            "index_type": index_type,
            "params": params,
        }

        # Preferred style in newer client versions.
        try:
            self.client.create_index(
                collection_name=self.collection_name,
                field_name="vector",
                index_params=index_params,
            )
            return True
        except Exception:
            pass

        # Fallback style with explicit kwargs.
        try:
            self.client.create_index(
                collection_name=self.collection_name,
                field_name="vector",
                metric_type="COSINE",
                index_type=index_type,
                params=params,
            )
            return True
        except Exception:
            pass

        # Fallback for clients exposing prepare_index_params().
        prepare = getattr(self.client, "prepare_index_params", None)
        if not callable(prepare):
            return False
        try:
            prepared = prepare()
            add_index = getattr(prepared, "add_index", None)
            if callable(add_index):
                add_index(
                    field_name="vector",
                    metric_type="COSINE",
                    index_type=index_type,
                    params=params,
                )
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=prepared,
            )
            return True
        except Exception:
            return False

    def _build_search_params(self) -> dict[str, object] | None:
        """Build search params aligned with chosen index type."""

        if self.index_type_in_use == "IVF_FLAT":
            return {
                "metric_type": "COSINE",
                "params": {"nprobe": self.ivf_nprobe},
            }
        return None


def _to_int_or_none(value: Any) -> int | None:
    """Best-effort conversion to int for optional numeric metadata fields."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_delete_count(result: Any) -> int:
    """Best-effort parse delete count from Milvus client response."""

    if isinstance(result, dict):
        raw = result.get("delete_count", result.get("count", 0))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0
    return 0


def _escape_filter_value(value: str) -> str:
    """Escape string value for Milvus filter expression literal."""

    return value.replace("\\", "\\\\").replace('"', '\\"')
