import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('milvus_lite')

from src.ingest.pipeline import ChunkRecord
from src.retrieval.vector_store import MilvusVectorStore


def test_persisted_collection_can_be_searched_after_reopening(tmp_path):
    uri = str(tmp_path / 'restart.db')
    first = MilvusVectorStore(uri, 'restart_check')
    try:
        first.recreate(2)
        chunk = ChunkRecord(
            'synthetic restart fact', 'demo.txt', 0, 0,
            doc_id='doc_restart', chunk_id='chunk_restart',
        )
        first.insert_chunks([chunk], [[1.0, 0.0]])
    finally:
        first.client.close()

    reopened = MilvusVectorStore(uri, 'restart_check')
    try:
        result = reopened.search([1.0, 0.0], 1)
        assert reopened.row_count() == 1
        assert result[0].text == 'synthetic restart fact'
        assert result[0].chunk_id == 'chunk_restart'
    finally:
        reopened.client.close()


def test_persisted_collection_can_be_searched_from_a_new_process(tmp_path):
    uri = str(tmp_path / 'new-process.db')
    root = Path(__file__).resolve().parents[1]
    create = '''
import sys
from src.ingest.pipeline import ChunkRecord
from src.retrieval.vector_store import MilvusVectorStore
store = MilvusVectorStore(sys.argv[1], 'process_check')
try:
    store.recreate(2)
    store.insert_chunks([ChunkRecord('persisted process fact', 'demo.txt', 0, 0)], [[1.0, 0.0]])
finally:
    store.client.close()
'''
    search = '''
import sys
from src.retrieval.vector_store import MilvusVectorStore
store = MilvusVectorStore(sys.argv[1], 'process_check')
try:
    assert store.search([1.0, 0.0], 1)[0].text == 'persisted process fact'
finally:
    store.client.close()
'''
    env = {name: os.environ[name] for name in ('PATH', 'HOME', 'TMPDIR') if name in os.environ}
    env['PYTHONPATH'] = str(root)
    for code in (create, search):
        result = subprocess.run(
            [sys.executable, '-c', code, uri],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
