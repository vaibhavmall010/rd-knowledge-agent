"""Load engineering documents with YAML front matter and split them into section chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
SECTION_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    title: str
    products: tuple[str, ...]
    date: str
    body: str
    metadata: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    title: str
    section: str
    products: tuple[str, ...]
    text: str


def parse_front_matter(raw: str) -> tuple[dict, str]:
    match = FRONT_MATTER.match(raw)
    if not match:
        raise ValueError("document has no YAML front matter")
    meta = yaml.safe_load(match.group(1)) or {}
    return meta, match.group(2).strip()


def load_document(path: Path) -> Document:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    for key in ("id", "type", "title", "date"):
        if key not in meta:
            raise ValueError(f"{path.name}: front matter is missing '{key}'")
    return Document(
        doc_id=str(meta["id"]),
        doc_type=str(meta["type"]),
        title=str(meta["title"]),
        products=tuple(str(p) for p in meta.get("products", [])),
        date=str(meta["date"]),
        body=body,
        metadata={k: v for k, v in meta.items() if k not in {"id", "type", "title", "products", "date"}},
    )


def load_documents(directory: Path) -> list[Document]:
    docs = [load_document(p) for p in sorted(Path(directory).glob("*.md"))]
    ids = [d.doc_id for d in docs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate document ids: {sorted(duplicates)}")
    return docs


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries so no piece exceeds max_chars (unless one paragraph does)."""
    pieces, buffer = [], ""
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if buffer and len(buffer) + len(para) + 2 > max_chars:
            pieces.append(buffer)
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer:
        pieces.append(buffer)
    return pieces


def chunk_document(doc: Document, max_chars: int = 900) -> list[Chunk]:
    """One chunk per '## ' section; long sections are split by paragraph.

    Every chunk keeps the document id, type, title, section name and products, so a retrieved
    passage can always be cited and filtered.
    """
    headings = list(SECTION_HEADING.finditer(doc.body))
    sections: list[tuple[str, str]] = []
    intro_end = headings[0].start() if headings else len(doc.body)
    intro = re.sub(r"^#\s+.*$", "", doc.body[:intro_end], flags=re.MULTILINE).strip()
    if intro:
        sections.append(("Overview", intro))
    for i, h in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(doc.body)
        sections.append((h.group(1), doc.body[h.end() : end].strip()))

    chunks = []
    for name, text in sections:
        for j, piece in enumerate(_split_long(text, max_chars)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks)}",
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    title=doc.title,
                    section=name if j == 0 else f"{name} (cont.)",
                    products=doc.products,
                    text=piece,
                )
            )
    return chunks
