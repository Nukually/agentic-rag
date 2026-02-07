from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
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
