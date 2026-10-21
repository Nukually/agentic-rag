"""Structure-aware text chunks with offsets into the original document."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One chunk of text with original character offsets."""

    text: str
    start: int
    end: int


def _blocks(text: str) -> list[tuple[int, int, bool]]:
    blocks: list[tuple[int, int, bool]] = []
    paragraph_start: int | None = None
    paragraph_end = offset = 0
    fence = ""
    code_start = code_end = 0
    for line in text.splitlines(keepends=True):
        start = offset + len(line) - len(line.lstrip())
        end = offset + len(line.rstrip())
        offset += len(line)
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)", line)
        if fence:
            code_end = end
            if (fence_match and fence_match[1][0] == fence[0]
                    and len(fence_match[1]) >= len(fence)
                    and not fence_match[2].strip()):
                blocks.append((code_start, code_end, False))
                fence = ""
            continue
        heading = bool(re.match(r"^ {0,3}#{1,6}(?:\s|$)", line))
        table_row = "|" in line
        if start >= end or heading or table_row or fence_match:
            if paragraph_start is not None:
                blocks.append((paragraph_start, paragraph_end, False))
                paragraph_start = None
            if fence_match:
                fence = fence_match[1]
                code_start, code_end = start, end
            elif heading or table_row:
                blocks.append((start, end, heading))
        else:
            if paragraph_start is None:
                paragraph_start = start
            paragraph_end = end
    if paragraph_start is not None:
        blocks.append((paragraph_start, paragraph_end, False))
    if fence:
        blocks.append((code_start, code_end, False))
    return blocks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Pack whole paragraphs and pipe-table rows; headings start new chunks.

    Args:
        text: Raw input text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Maximum overlap, clamped below chunk_size. Whole trailing
            blocks are reused only when they fit this budget and the next block.
            Oversized blocks use character windows with this overlap.

    Blank lines delimit paragraphs; lines containing a pipe are kept as rows.
    Backtick and tilde fenced code stays whole unless it exceeds chunk_size;
    headings, pipes, and blank lines inside a fence do not split the code block.
    Boundary whitespace may be omitted. Every returned text is an exact slice
    of the input, and section boundaries do not carry overlap from prior text.

    Returns:
        list[Chunk]: Ordered chunk sequence for downstream embedding/indexing.

    Example:
        >>> split_text("abcdef", chunk_size=4, chunk_overlap=1)
        [Chunk(text='abcd', start=0, end=4), Chunk(text='def', start=3, end=6)]
    """

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    overlap = min(chunk_overlap, max(chunk_size - 1, 0))
    blocks: list[tuple[int, int, bool]] = []
    for start, end, heading in _blocks(text):
        while end - start > chunk_size:
            blocks.append((start, start + chunk_size, heading))
            start += chunk_size - overlap
            heading = False
        blocks.append((start, end, heading))

    active: list[tuple[int, int]] = []
    chunks: list[Chunk] = []
    for start, end, heading in blocks:
        if active and (heading or end - active[0][0] > chunk_size):
            previous_start, previous_end = active[0][0], active[-1][1]
            chunks.append(Chunk(text[previous_start:previous_end], previous_start, previous_end))
            active = [
                block for block in active
                if not heading and previous_end - block[0] <= overlap
                and end - block[0] <= chunk_size
            ]
        active.append((start, end))
    if active:
        start, end = active[0][0], active[-1][1]
        chunks.append(Chunk(text[start:end], start, end))

    return [chunk for chunk in chunks if chunk.text.strip()]
