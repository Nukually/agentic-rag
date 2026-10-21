import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('entrypoint', [
    ['-m', 'src.app.cli_chat'],
    ['scripts/ingest_once.py'],
    ['scripts/query_once.py'],
    ['scripts/agentic_query_once.py'],
    ['scripts/evaluate_retrieval.py'],
])
def test_help_does_not_require_credentials_or_connect_to_services(entrypoint):
    result = subprocess.run(
        [sys.executable, *entrypoint, '--help'],
        cwd=Path(__file__).resolve().parents[1],
        env={}, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
