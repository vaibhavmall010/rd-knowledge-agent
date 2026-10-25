"""Okapi BM25 over document chunks, with tokenisation that keeps engineering identifiers intact."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from rdagent.corpus import Chunk

TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or "
    "the this to was we were what when which who why will with".split()
)


def _stem(tok: str) -> str:
    """Light plural stemming for words only: 'decisions' -> 'decision', 'cycles' -> 'cycle'."""
    if tok.isalpha() and len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; 'OV-600P' yields 'ov-600p' plus its parts 'ov' and '600p'."""
    tokens = []
    for tok in TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        tok = _stem(tok)
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(p for p in tok.split("-") if p and p not in STOPWORDS)
    return tokens


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Title and section are repeated so a match there counts more than one in the body.
        self._docs = [
            tokenize(f"{c.doc_id} {c.title} {c.title} {c.section} {c.section} {c.text}") for c in chunks
        ]
        self._tf = [Counter(d) for d in self._docs]
        self._len = [len(d) for d in self._docs]
        self._avg_len = sum(self._len) / max(len(self._len), 1)
        df = Counter(t for d in self._docs for t in set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, i: int, query: list[str]) -> float:
        tf, length = self._tf[i], self._len[i]
        score = 0.0
        for t in query:
            f = tf.get(t)
            if not f:
                continue
            norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self._avg_len))
            score += self._idf.get(t, 0.0) * norm
        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str | None = None,
        product: str | None = None,
    ) -> list[SearchHit]:
        q = tokenize(query)
        hits = []
        for i, chunk in enumerate(self.chunks):
            if doc_type and chunk.doc_type != doc_type:
                continue
            if product and product.upper() not in {p.upper() for p in chunk.products}:
                continue
            score = self._score(i, q)
            if score > 0:
                hits.append(SearchHit(chunk, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
