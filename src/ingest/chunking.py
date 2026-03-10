"""Text chunking utilities for retrieval-oriented document segmentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One chunk of text with original character offsets."""

    text: str
    start: int
    end: int


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split text into overlapping chunks by character window.

    Args:
        text: Raw input text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between neighboring
            chunks.

    Returns:
        list[Chunk]: Ordered chunk sequence for downstream embedding/indexing.

    Example:
        >>> split_text("abcdef", chunk_size=4, chunk_overlap=1)
        [Chunk(text='abcd', start=0, end=4), Chunk(text='def', start=3, end=6)]
    """

    content = text.strip()
    if not content:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    overlap = min(chunk_overlap, max(chunk_size - 1, 0))
    step = max(chunk_size - overlap, 1)

    chunks: list[Chunk] = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        piece = content[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, start=start, end=end))
        if end == len(content):
            break
        start += step

    return chunks
