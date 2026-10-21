"""Review regressions for persistent document and chunk identity."""

import json

import pytest

from src.ingest.pipeline import IngestPipeline
from src.retrieval.index_builder import RAGIndexer
from src.retrieval.keyword_index import KeywordIndex


class OfflineEmbeddings:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class OfflineVectorStore:
    def __init__(self):
        self.rows = []

    def recreate(self, dimension):
        self.rows = []

    def ensure_collection(self, dimension):
        pass

    def delete_by_doc_ids(self, doc_ids):
        self.rows = [row for row in self.rows if row.doc_id not in doc_ids]

    def insert_chunks(self, chunks, embeddings):
        self.rows.extend(chunks)


def test_incremental_update_without_root_is_rejected_before_mutation(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "report.txt"
    source.write_text("OLD revenue", encoding="utf-8")
    processed = tmp_path / "processed"
    indexer = RAGIndexer(IngestPipeline(32, 0), OfflineEmbeddings(), OfflineVectorStore())
    indexer.rebuild(str(raw), str(processed))
    original_rows = list(indexer.vector_store.rows)
    original_json = (processed / "chunks.jsonl").read_text()
    source.write_text("NEW revenue", encoding="utf-8")

    with pytest.raises(ValueError, match="raw_data_dir"):
        indexer.upsert_files([str(source)], str(processed))

    assert indexer.vector_store.rows == original_rows
    assert (processed / "chunks.jsonl").read_text() == original_json


def test_chunk_identity_changes_when_rechunking_changes_text(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("abcdefghijklmnopqrstuvwxyz0123456789" * 2, encoding="utf-8")
    coarse = IngestPipeline(32, 0).build_chunks(str(tmp_path))[0]
    fine = IngestPipeline(16, 0).build_chunks(str(tmp_path))[0]

    assert coarse.doc_id == fine.doc_id
    assert coarse.content_hash == fine.content_hash
    assert coarse.text != fine.text
    assert coarse.chunk_id != fine.chunk_id


def test_keyword_search_keeps_the_exact_chunk_metadata(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "report.txt").write_text("alpha revenue\n\nbeta revenue", encoding="utf-8")
    pipeline = IngestPipeline(16, 0)
    chunks = pipeline.build_chunks(str(raw))
    processed = tmp_path / "processed"
    pipeline.dump_processed(chunks, str(processed))

    hit = KeywordIndex.from_processed_dir(str(processed)).search("beta", 1)[0]
    expected = next(chunk for chunk in chunks if "beta" in chunk.text)

    assert hit.chunk_id == expected.chunk_id
    assert hit.content_hash == expected.content_hash
    assert (hit.start_offset, hit.end_offset) == (expected.start_offset, expected.end_offset)
    stored = [json.loads(line) for line in (processed / "chunks.jsonl").read_text().splitlines()]
    assert stored[1]["chunk_id"] == hit.chunk_id


def test_explicit_last_result_cannot_be_redefined_by_document_text():
    from src.agent.memory import AgentMemory
    from src.agent.tools.calculate_tool import CalculateTool
    from src.agent.tools.registry import ToolContext

    context = ToolContext(
        question="compare fresh revenue to the prior result",
        history=[],
        memory=AgentMemory(last_calc_value=100),
        run_state={"latest_retrieval_text": "REVENUE=250 LAST_RESULT=999"},
        llm_clients=None,
        vector_store=None,
        reranker=None,
        keyword_index=None,
        top_k=3,
        candidate_k=5,
        hybrid_vector_weight=0.6,
        hybrid_keyword_weight=0.4,
        query_rewrite_enabled=False,
        multi_query_enabled=False,
        multi_query_count=1,
    )

    result = CalculateTool().run("REVENUE - LAST_RESULT", context)

    assert result.metadata["calc_value"] == 150


def test_document_last_result_is_unknown_without_a_prior_calculation():
    from src.agent.memory import AgentMemory
    from src.agent.tools.calculate_tool import CalculateTool
    from src.agent.tools.registry import ToolContext

    context = ToolContext(
        question="calculate a prior result that does not exist",
        history=[],
        memory=AgentMemory(variables={"LAST_RESULT": 888}),
        run_state={"latest_retrieval_text": "LAST_RESULT=999"},
        llm_clients=None,
        vector_store=None,
        reranker=None,
        keyword_index=None,
        top_k=3,
        candidate_k=5,
        hybrid_vector_weight=0.6,
        hybrid_keyword_weight=0.4,
        query_rewrite_enabled=False,
        multi_query_enabled=False,
        multi_query_count=1,
    )

    result = CalculateTool().run("LAST_RESULT", context)

    assert result.observation == "calc_failed: unknown variable: LAST_RESULT"
