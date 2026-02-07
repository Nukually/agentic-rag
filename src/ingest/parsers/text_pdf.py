from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class ParsedUnit:
    source: str
    text: str
    page: int


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _parse_pdf(path: Path) -> list[ParsedUnit]:
    units: list[ParsedUnit] = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                units.append(ParsedUnit(source=str(path), text=text, page=page_no))
    return units


def parse_document(path: str) -> list[ParsedUnit]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {file_path}")

    if suffix == ".pdf":
        return _parse_pdf(file_path)

    text = _read_text_file(file_path)
    if not text:
        return []
    return [ParsedUnit(source=str(file_path), text=text, page=0)]
