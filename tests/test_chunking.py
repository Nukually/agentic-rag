from src.ingest.chunking import split_text


def test_split_text_basic() -> None:
    text = "a" * 300
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    assert chunks[0].text
