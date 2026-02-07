from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ingest.chunking import split_text
from src.ingest.parsers.text_pdf import SUPPORTED_SUFFIXES, parse_document


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    source: str
    page: int
    chunk_index: int


class IngestPipeline:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def discover_files(self, root_dir: str) -> list[Path]:
        root = Path(root_dir)
        if not root.exists():
            return []

        files: list[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)
        return sorted(files)

    def build_chunks(self, root_dir: str) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []

        for file_path in self.discover_files(root_dir):
            parsed_units = parse_document(str(file_path))
            for unit in parsed_units:
                piece_list = split_text(unit.text, self.chunk_size, self.chunk_overlap)
                for idx, piece in enumerate(piece_list):
                    chunks.append(
                        ChunkRecord(
                            text=piece.text,
                            source=unit.source,
                            page=unit.page,
                            chunk_index=idx,
                        )
                    )

        return chunks

    def dump_processed(self, chunks: list[ChunkRecord], output_dir: str) -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        out_file = path / "chunks.jsonl"

        with out_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

        return out_file
