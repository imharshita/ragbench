"""Chunking strategies for the retrieval corpus.

A *document* is one source file. A *chunk* is an indexed passage. Chunk ids are stable
so retrieval results and citations stay comparable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


def load_corpus(directory: str | Path) -> list[Document]:
    """Load `*.md` files; the file stem becomes the document id."""
    path = Path(directory)
    files = sorted(path.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md files found in {path}")
    return [Document(doc_id=f.stem, text=f.read_text(encoding="utf-8")) for f in files]


def _chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}:{index:03d}"


def chunk_fixed(doc: Document, size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Split a document into fixed-size character windows with overlap.

    A window boundary is nudged backwards to the nearest whitespace so chunks do not
    start mid-word. Overlap must be smaller than size.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text = doc.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(id=_chunk_id(doc.doc_id, index), doc_id=doc.doc_id, text=piece))
            index += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_paragraphs(doc: Document, max_chars: int = 600) -> list[Chunk]:
    """Split on blank lines, then merge adjacent paragraphs up to `max_chars`."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    chunks: list[Chunk] = []
    buffer = ""
    index = 0
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if buffer and len(candidate) > max_chars:
            chunks.append(Chunk(id=_chunk_id(doc.doc_id, index), doc_id=doc.doc_id, text=buffer))
            index += 1
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        chunks.append(Chunk(id=_chunk_id(doc.doc_id, index), doc_id=doc.doc_id, text=buffer))
    return chunks


CHUNKERS = {
    "fixed": chunk_fixed,
    "paragraph": chunk_paragraphs,
}


def chunk_corpus(
    documents: list[Document], strategy: str = "fixed", **kwargs: int
) -> list[Chunk]:
    """Apply one named chunking strategy to every document."""
    try:
        chunker = CHUNKERS[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(CHUNKERS)}") from exc

    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker(doc, **kwargs))
    return chunks
