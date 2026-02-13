from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class ParsedUnit:
    source: str
    text: str
    page: int


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _normalize_cell(value: Any) -> str:
    text = str(value or "").replace("\n", " ")
    return " ".join(text.split())


def _trim_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    keep_indices = [
        index for index in range(width) if any(r[index].strip() for r in padded)
    ]
    if not keep_indices:
        return []

    return [[row[index] for index in keep_indices] for row in padded]


def _table_to_text(rows: list[list[Any]], page_no: int, table_index: int) -> str:
    normalized_rows: list[list[str]] = []
    for row in rows:
        cells = [_normalize_cell(cell) for cell in row]
        if any(cells):
            normalized_rows.append(cells)

    normalized_rows = _trim_empty_columns(normalized_rows)
    if not normalized_rows:
        return ""

    width = max(len(row) for row in normalized_rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in normalized_rows]

    lines: list[str] = [f"[TABLE page={page_no} index={table_index}]"]
    header = normalized_rows[0]
    lines.append(" | ".join(header))
    if len(normalized_rows) > 1:
        lines.append(" | ".join(["---"] * width))
        for row in normalized_rows[1:]:
            lines.append(" | ".join(row))

    return "\n".join(lines).strip()


def _extract_tables(page: fitz.Page, source: str, page_no: int) -> list[ParsedUnit]:
    if not hasattr(page, "find_tables"):
        return []

    try:
        table_finder = page.find_tables()
    except Exception:
        return []

    tables = getattr(table_finder, "tables", []) or []
    units: list[ParsedUnit] = []
    for table_index, table in enumerate(tables, start=1):
        try:
            rows = table.extract() or []
        except Exception:
            continue

        text = _table_to_text(rows, page_no=page_no, table_index=table_index)
        if text:
            units.append(ParsedUnit(source=source, text=text, page=page_no))

    return units


def _parse_pdf(path: Path) -> list[ParsedUnit]:
    units: list[ParsedUnit] = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                units.append(ParsedUnit(source=str(path), text=text, page=page_no))
            units.extend(_extract_tables(page=page, source=str(path), page_no=page_no))
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
