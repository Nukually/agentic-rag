from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.retrieval.vector_store import SearchHit

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class KeywordDoc:
    text: str
    source: str
    page: int
    chunk_index: int
    tokens: list[str]


class KeywordIndex:
    def __init__(self, docs: list[KeywordDoc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b

        self._doc_len: list[int] = []
        self._avg_len: float = 0.0
        self._inv_index: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}

        self._build()

    @classmethod
    def from_jsonl(cls, jsonl_path: str) -> KeywordIndex | None:
        path = Path(jsonl_path)
        if not path.exists():
            return None

        docs: list[KeywordDoc] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = str(raw.get("text", ""))
                tokens = _tokenize(text)
                if not tokens:
                    continue

                docs.append(
                    KeywordDoc(
                        text=text,
                        source=str(raw.get("source", "")),
                        page=int(raw.get("page", 0) or 0),
                        chunk_index=int(raw.get("chunk_index", 0) or 0),
                        tokens=tokens,
                    )
                )

        if not docs:
            return None
        return cls(docs)

    @classmethod
    def from_processed_dir(cls, processed_dir: str) -> KeywordIndex | None:
        path = Path(processed_dir) / "chunks.jsonl"
        return cls.from_jsonl(str(path))

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if not query or not self.docs:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        query_tf = Counter(tokens)
        scores: dict[int, float] = defaultdict(float)

        for term, qf in query_tf.items():
            postings = self._inv_index.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            q_boost = 1.0 + math.log(1.0 + qf)
            for doc_id, tf in postings:
                dl = self._doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self._avg_len, 1.0))
                score = idf * (tf * (self.k1 + 1.0) / denom)
                scores[doc_id] += score * q_boost

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        hits: list[SearchHit] = []
        for doc_id, score in ranked[: max(1, top_k)]:
            doc = self.docs[doc_id]
            hits.append(
                SearchHit(
                    text=doc.text,
                    source=doc.source,
                    page=doc.page,
                    score=float(score),
                )
            )
        return hits

    def _build(self) -> None:
        inv_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_len: list[int] = []
        df: dict[str, int] = defaultdict(int)

        for doc_id, doc in enumerate(self.docs):
            counts = Counter(doc.tokens)
            doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                inv_index[term].append((doc_id, tf))
                df[term] += 1

        self._doc_len = doc_len
        self._avg_len = sum(doc_len) / max(len(doc_len), 1)
        self._inv_index = dict(inv_index)
        self._idf = {
            term: math.log(1.0 + (len(self.docs) - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
