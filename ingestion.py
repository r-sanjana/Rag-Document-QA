"""
ingestion.py — multi-PDF ingestion (no Streamlit, no Gemini, no ChromaDB).

    uploaded files ─► validate (max 10) ─► hash (duplicates) ─► parse each PDF
                  ─► chunk ─► chunks with full metadata ─► UploadResult per file

Every file is processed independently: a corrupt PDF becomes an "error" result
and the other PDFs are still processed.

Metadata on EVERY chunk (this is what ChromaDB stores and what the knowledge
graph's provenance points back to):

    source        PDF name shown in the UI (made unique if two different PDFs share a filename)
    page          0-based page index (kept because existing code uses it)
    page_number   1-based page number = page + 1   <- shown to humans
    chunk_id      deterministic id from graph_rag.make_chunk_id(): source + page + text.
                  The SAME id is the ChromaDB id and the knowledge-graph chunk key.
    content_hash  SHA-256 of the whole PDF file (identifies the document/version)
"""

import hashlib
import io
import os
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from graph_rag import make_chunk_id

MAX_PDFS = 10


@dataclass
class UploadResult:
    filename: str
    status: str                       # "indexed" | "duplicate" | "empty" | "error" | "rejected"
    message: str
    source: Optional[str] = None      # the name the document is stored under
    content_hash: Optional[str] = None
    chunks: list = field(default_factory=list)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selection_error(n_selected: int) -> Optional[str]:
    """None if the selection is OK, otherwise a message. We never silently
    process 'the first 10'."""
    if n_selected > MAX_PDFS:
        return (f"You selected {n_selected} PDFs, but at most {MAX_PDFS} are supported. "
                f"Please select {MAX_PDFS} or fewer (nothing was indexed).")
    return None


def unique_source_name(filename: str, content_hash: str, taken: set) -> str:
    """Same filename + different content must not overwrite: the newcomer is
    stored as 'name [ab12cd].pdf'."""
    if filename not in taken:
        return filename
    stem, ext = os.path.splitext(filename)
    n = 6
    while True:
        candidate = f"{stem} [{content_hash[:n]}]{ext}"
        if candidate not in taken:
            return candidate
        n += 2


def extract_chunks(data: bytes, source: str, content_hash: str, chunk_size: int, chunk_overlap: int) -> list:
    """PDF bytes -> chunks with full metadata. Raises on unreadable PDFs.
    Returns [] if the PDF has no extractable text."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("the PDF is password-protected")
    pages = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""          # one unreadable page must not lose the whole document
        if text.strip():
            pages.append(Document(
                page_content=text,
                metadata={"source": source, "page": index, "page_number": index + 1, "content_hash": content_hash},
            ))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, seen = [], set()
    for chunk in splitter.split_documents(pages):
        chunk.metadata["chunk_id"] = make_chunk_id(chunk)
        if chunk.metadata["chunk_id"] in seen:      # identical text repeated on the same page
            continue
        seen.add(chunk.metadata["chunk_id"])
        chunks.append(chunk)
    return chunks


def prepare_uploads(files: list, known_hashes: dict, taken_sources: set, n_existing_docs: int,
                    chunk_size: int, chunk_overlap: int) -> list:
    """files = [(filename, bytes)]. known_hashes = {content_hash: source} of the
    documents already indexed. Returns one UploadResult per file, in order.

      same filename  + same content      -> duplicate
      other filename + same content      -> duplicate
      same filename  + different content -> indexed under a distinct name
    """
    known = dict(known_hashes)
    taken = set(taken_sources)
    capacity = MAX_PDFS - n_existing_docs
    results = []
    for filename, data in files:
        h = file_hash(data)
        if h in known:
            results.append(UploadResult(filename, "duplicate",
                                        f"identical content to '{known[h]}', which is already indexed — skipped",
                                        source=known[h], content_hash=h))
            continue
        if capacity <= 0:
            results.append(UploadResult(filename, "rejected",
                                        f"the limit of {MAX_PDFS} indexed PDFs is reached — remove one first", content_hash=h))
            continue
        source = unique_source_name(filename, h, taken)
        try:
            chunks = extract_chunks(data, source, h, chunk_size, chunk_overlap)
        except Exception as e:
            results.append(UploadResult(filename, "error", f"could not be read as a PDF ({type(e).__name__}: {e})", content_hash=h))
            continue
        if not chunks:
            results.append(UploadResult(filename, "empty",
                                        "no extractable text (empty, or scanned/image-only) — not indexed", content_hash=h))
            continue
        msg = f"{len(chunks)} chunks"
        if source != filename:
            msg += f" (same filename as an indexed PDF but different content, stored as '{source}')"
        results.append(UploadResult(filename, "indexed", msg, source=source, content_hash=h, chunks=chunks))
        known[h] = source
        taken.add(source)
        capacity -= 1
    return results
