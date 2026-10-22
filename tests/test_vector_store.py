from dataclasses import asdict
from types import SimpleNamespace

import pytest

from src.ingest.pipeline import ChunkRecord
from src.retrieval.vector_store import MilvusVectorStore
from src.agent.tools.rag_retrieve import _merge_candidates, _merge_query_hits, _to_retrieved


def store(client):
    subject = MilvusVectorStore.__new__(MilvusVectorStore)
    subject.collection_name = 'test-chunks'
    subject.client = client
    subject.index_type_in_use = 'FLAT'
    subject._ensure_supported_index = lambda: None
    return subject


def test_dimension_mismatch_is_rejected_before_updates():
    subject = store(SimpleNamespace(
        has_collection=lambda **_: True,
        describe_collection=lambda **_: {'fields':[{'name':'vector','params':{'dim':'2'}}]},
    ))
    subject.ensure_collection(2)
    with pytest.raises(ValueError, match='rebuild'):
        subject.ensure_collection(3)


def test_delete_failure_is_not_silently_followed_by_duplicate_inserts():
    def fail(**kwargs):
        raise RuntimeError('synthetic delete failure')
    subject = store(SimpleNamespace(has_collection=lambda **_: True, delete=fail))
    with pytest.raises(RuntimeError, match='synthetic delete'):
        subject.delete_by_doc_ids(['doc_test'])


def test_chunk_provenance_survives_vector_search_fusion_and_rerank_conversion():
    chunk = ChunkRecord('synthetic text', 'a.txt', 1, 3, doc_id='doc_test', chunk_id='chunk_test', content_hash='hash', start_offset=5, end_offset=19)
    inserted = []
    subject = store(SimpleNamespace(
        has_collection=lambda **_: True,
        insert=lambda **kwargs: inserted.extend(kwargs['data']),
        search=lambda **_: [[{'entity':asdict(chunk), 'distance':.9}]],
    ))
    subject.insert_chunks([chunk], [[1., 0.]])
    assert inserted[0] == asdict(chunk) | {'vector':[1., 0.]}
    hit = subject.search([1., 0.], 1)[0]
    merged = _merge_candidates(_merge_query_hits([[hit]], 1), [], 1, 0, 1)[0]
    final = _to_retrieved(merged, .95)
    assert (final.doc_id, final.chunk_id, final.content_hash, final.start_offset, final.end_offset) == ('doc_test', 'chunk_test', 'hash', 5, 19)


def test_existing_vector_index_is_reused_without_recreation():
    attempts = []

    def create_index(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError('index already exists')

    subject = MilvusVectorStore.__new__(MilvusVectorStore)
    subject.collection_name = 'test-chunks'
    subject.ivf_nlist = 1024
    subject.ivf_nprobe = 32
    subject.index_type_in_use = 'AUTOINDEX'
    subject.client = SimpleNamespace(
        list_indexes=lambda **_: ['vector_index'],
        describe_index=lambda **_: {'field_name':'vector', 'index_type':'IVF_FLAT'},
        create_index=create_index,
    )

    subject._ensure_supported_index()

    assert attempts == []
    assert subject.index_type_in_use == 'IVF_FLAT'
    assert subject._build_search_params()['params']['nprobe'] == 32
