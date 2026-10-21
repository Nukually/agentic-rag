import json
from dataclasses import asdict

import pytest

from src.ingest.pipeline import IngestPipeline
from src.retrieval.index_builder import RAGIndexer
from src.retrieval.keyword_index import KeywordIndex


class Embeddings:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class VectorStore:
    def __init__(self):
        self.rows = []

    def recreate(self, dimension):
        self.rows = []

    def ensure_collection(self, dimension):
        pass

    def delete_by_doc_ids(self, doc_ids):
        self.rows = [row for row in self.rows if row.doc_id not in doc_ids]

    def insert_chunks(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError('embedding count mismatch')
        self.rows.extend(chunks)


def document(root, relative_path, text):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def indexer():
    return RAGIndexer(IngestPipeline(32, 0), Embeddings(), VectorStore())


def test_same_filename_in_different_directories_does_not_replace_other_document(tmp_path):
    raw = tmp_path / 'knowledge'
    a = document(raw, 'a/report.txt', 'COMPANY_A revenue 100')
    b = document(raw, 'b/report.txt', 'COMPANY_B revenue 200')
    processed = tmp_path / 'processed'
    subject = indexer()
    for path in (a, b):
        subject.upsert_files([str(path)], str(processed), str(raw))
    assert len(subject.vector_store.rows) == 2
    rows = [json.loads(line) for line in (processed / 'chunks.jsonl').read_text().splitlines()]
    assert len({row['doc_id'] for row in rows}) == 2
    keyword = KeywordIndex.from_processed_dir(str(processed))
    assert keyword.search('COMPANY_A', 1)[0].text == 'COMPANY_A revenue 100'
    assert keyword.search('COMPANY_B', 1)[0].text == 'COMPANY_B revenue 200'


def test_document_identity_survives_content_update_and_root_move(tmp_path):
    raw = tmp_path / 'knowledge'
    path = document(raw, 'a/report.txt', 'first version')
    pipeline = IngestPipeline(32, 0)
    old = pipeline.build_chunks(str(raw))[0]
    path.write_text('second version', encoding='utf-8')
    new = pipeline.build_chunks(str(raw))[0]
    assert old.doc_id == new.doc_id
    assert old.content_hash != new.content_hash
    assert old.chunk_id != new.chunk_id
    moved_raw = tmp_path / 'moved'
    document(moved_raw, 'a/report.txt', 'second version')
    moved = pipeline.build_chunks(str(moved_raw))[0]
    assert asdict(moved) | {'source': new.source} == asdict(new)


def test_rebuild_and_incremental_upsert_share_ids_and_replace_only_target(tmp_path):
    raw = tmp_path / 'knowledge'
    a = document(raw, 'a.txt', 'old A')
    document(raw, 'b.txt', 'keep B')
    subject = indexer()
    processed = tmp_path / 'processed'
    subject.rebuild(str(raw), str(processed))
    old_id = next(row.doc_id for row in subject.vector_store.rows if row.text == 'old A')
    a.write_text('new A', encoding='utf-8')
    subject.upsert_files(['a.txt'], str(processed), str(raw))
    assert sorted(row.text for row in subject.vector_store.rows) == ['keep B', 'new A']
    assert next(row.doc_id for row in subject.vector_store.rows if row.text == 'new A') == old_id
    subject.upsert_files(['a.txt'], str(processed), str(raw))
    assert len(subject.vector_store.rows) == 2


def test_chunk_ids_and_indices_are_unique_across_pdf_units(tmp_path, monkeypatch):
    from src.ingest.parsers.text_pdf import ParsedUnit
    path = document(tmp_path, 'report.pdf', 'synthetic bytes')
    units = [ParsedUnit(str(path), 'page one', 1), ParsedUnit(str(path), '[TABLE page=1 index=1]\na | b', 1), ParsedUnit(str(path), 'page two', 2)]
    monkeypatch.setattr('src.ingest.pipeline.parse_document', lambda _: units)
    chunks = IngestPipeline(100, 0).build_chunks(str(tmp_path))
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert len({chunk.chunk_id for chunk in chunks}) == 3
    for chunk, unit in zip(chunks, units):
        assert chunk.text == unit.text[chunk.start_offset:chunk.end_offset]


def test_legacy_index_requires_full_rebuild_before_incremental_update(tmp_path):
    path = document(tmp_path, 'report.txt', 'new content')
    processed = tmp_path / 'processed'
    processed.mkdir()
    old_data = json.dumps({'doc_id':'report', 'text':'old content'}) + '\n'
    (processed / 'chunks.jsonl').write_text(old_data)
    subject = indexer()
    with pytest.raises(ValueError, match='rebuild'):
        subject.upsert_files([str(path)], str(processed), str(tmp_path))
    assert (processed / 'chunks.jsonl').read_text() == old_data
    assert subject.vector_store.rows == []


def test_invalid_embedding_batch_does_not_delete_existing_document(tmp_path):
    path = document(tmp_path, 'report.txt', 'first version')
    subject = indexer()
    processed = tmp_path / 'processed'
    subject.upsert_files([str(path)], str(processed), str(tmp_path))
    old_rows = list(subject.vector_store.rows)
    old_json = (processed / 'chunks.jsonl').read_text()
    path.write_text('new content ' * 20)
    subject.llm_clients.embed_texts = lambda texts: [[1.0, 0.0]]
    with pytest.raises(ValueError, match='embedding'):
        subject.upsert_files([str(path)], str(processed), str(tmp_path))
    assert subject.vector_store.rows == old_rows
    assert (processed / 'chunks.jsonl').read_text() == old_json
