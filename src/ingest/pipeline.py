"""入库流程"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ingest.chunking import split_text
from src.ingest.parsers.text_pdf import SUPPORTED_SUFFIXES, parse_document


@dataclass(frozen=True)
class ChunkRecord:
    """Persistent chunk payload written to vector and keyword indexes."""

    text: str
    source: str
    page: int
    chunk_index: int
    doc_id: str = ""
    file_name: str = ""
    source_type: str = ""
    company_code: str = ""
    company_name: str = ""
    report_year: int | None = None
    is_table: bool = False


class IngestPipeline:
    """Discover files, parse content, split into chunks, and dump JSONL.

    Args:
        chunk_size: Maximum chunk character length.
        chunk_overlap: Character overlap between neighboring chunks.

    Example:
        >>> ingest = IngestPipeline(chunk_size=1200, chunk_overlap=180)
        >>> chunks = ingest.build_chunks("./knowledge")
        >>> ingest.dump_processed(chunks, "./data/processed")
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def discover_files(self, root_dir: str) -> list[Path]:
        """Find supported source files recursively under `root_dir`."""

        root = Path(root_dir)
        if not root.exists():
            return []

        files: list[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)
        return sorted(files)

    def build_chunks(self, root_dir: str) -> list[ChunkRecord]:
        """接口"""

        return self.build_chunks_for_files(self.discover_files(root_dir))

    def build_chunks_for_files(self, file_paths: list[Path]) -> list[ChunkRecord]:
        """建立chunk"""

        chunks: list[ChunkRecord] = []

        for file_path in file_paths:
            if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            parsed_units = parse_document(str(file_path))
            for unit in parsed_units:
                unit_metadata = self._extract_unit_metadata(source=unit.source, text=unit.text)
                piece_list = split_text(unit.text, self.chunk_size, self.chunk_overlap)
                for idx, piece in enumerate(piece_list):
                    chunks.append(
                        ChunkRecord(
                            text=piece.text,
                            source=unit.source,
                            page=unit.page,
                            chunk_index=idx,
                            doc_id=unit_metadata["doc_id"],
                            file_name=unit_metadata["file_name"],
                            source_type=unit_metadata["source_type"],
                            company_code=unit_metadata["company_code"],
                            company_name=unit_metadata["company_name"],
                            report_year=unit_metadata["report_year"],
                            is_table=unit_metadata["is_table"],
                        )
                    )

        return chunks

    def dump_processed(self, chunks: list[ChunkRecord], output_dir: str) -> Path:
        """Write chunk records to `chunks.jsonl` for keyword indexing.

        Args:
            chunks: Chunk records to persist.
            output_dir: Directory where `chunks.jsonl` will be created.

        Returns:
            Path: Output JSONL path.
        """

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        out_file = path / "chunks.jsonl"

        with out_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

        return out_file

    @staticmethod
    def _extract_unit_metadata(source: str, text: str) -> dict[str, object]:
        """Extract lightweight retrieval metadata from source path and unit text."""

        path = Path(source)
        stem = path.stem
        file_name = path.name
        source_type = path.suffix.lower().lstrip(".")

        company_code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", stem)
        company_code = company_code_match.group(1) if company_code_match else ""

        # Keep only short Chinese name-like fragments; avoid long noisy suffixes.
        chinese_names = re.findall(r"[\u4e00-\u9fff]{2,12}", stem)
        company_name = chinese_names[0] if chinese_names else ""

        year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", stem)
        if year_match is None:
            # Fallback for date-like tokens such as 20260203 -> 2026.
            year_match = re.search(r"(20\d{2})\d{4}", stem)
        report_year: int | None = int(year_match.group(1)) if year_match else None

        normalized_text = (text or "").lstrip()
        is_table = normalized_text.startswith("[TABLE ")

        return {
            "doc_id": stem,
            "file_name": file_name,
            "source_type": source_type,
            "company_code": company_code,
            "company_name": company_name,
            "report_year": report_year,
            "is_table": is_table,
        }
