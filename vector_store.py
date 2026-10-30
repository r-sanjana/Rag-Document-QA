"""
vector_store.py — everything that touches ChromaDB (no Streamlit).

ChromaDB is THE database of this project. It stores, persistently on disk:
    id         chunk_id            (deterministic; also the knowledge-graph key)
    document   chunk text
    embedding  local sentence-transformers embedding (no API calls, no quota)
    metadata   source, page, page_number, chunk_id, content_hash

Two collections live in the same persistent folder:
    rag_documents_local   the PDF chunks (Traditional RAG, Hybrid, Local Search text units)
    graph_entities_local  one embedding per knowledge-graph entity (used by Local Search to
                    find entities whose NAME does not literally appear in the question)

NetworkX is not a database; it only holds the graph in memory.
"""

import os
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from evidence import evidence_from_document
from graph_rag import KnowledgeGraph, MatchedEntity

# Local embedding model (runs on this computer, so indexing and searching use NO Gemini quota).
# The default below can be overridden with LOCAL_EMBEDDING_MODEL (a Hugging Face model id or a FOLDER
# containing a downloaded model). It is read when the model object is CREATED, not at import time:
# app.py imports this module before load_dotenv() has read the .env file.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = ".chroma_store"
# "_local" names keep these vectors apart from any index built earlier with Gemini embeddings
# (different models make different-sized vectors; they must never share a collection).
COLLECTION_NAME = "rag_documents_local"
ENTITY_COLLECTION = "graph_entities_local"

EMBED_BATCH = 100        # texts per ChromaDB add call

# Semantic entity matching (Local Search fallback). Distance is Chroma's COSINE
# distance (0 = same direction, lower = closer). 0.35 is an UNCALIBRATED starting
# value — tune it on your own documents.
ENTITY_MATCH_MAX_DISTANCE = 0.35
ENTITY_MATCH_TOP_K = 3


class LocalEmbeddings(Embeddings):
    """Sentence-transformers embeddings computed on this computer.

    Two things matter here:
    * LAZY: the model is loaded the first time a text is embedded, not when the
      object is created. Reading stored chunks back from ChromaDB at start-up
      therefore does not load the model at all.
    * OFFLINE-FIRST: if the model is already in the local Hugging Face cache it is
      loaded with `local_files_only=True`, i.e. with NO network access. A normal
      load sends HEAD requests to huggingface.co on every start; on a network that
      blocks them each request waits and retries (WinError 10060, "Retrying in 1s
      [Retry 1/5]") which makes start-up very slow. Only when the model is not
      cached yet is it downloaded, once.

    all-MiniLM-L6-v2 reads at most ~256 tokens (~1000 characters) per text; longer
    chunks are silently truncated for embedding purposes, so keep the chunk size
    around 1000 characters or less."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get("LOCAL_EMBEDDING_MODEL") or EMBEDDING_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("Local embeddings need sentence-transformers: pip install sentence-transformers") from e
            try:
                self._model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
            except Exception:                                   # not cached yet -> download once
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_documents(self, texts: list) -> list:
        if not texts:
            return []
        vectors = self._load().encode(list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def _shared_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()


def make_embeddings(api_key: Optional[str] = None) -> LocalEmbeddings:
    """One embedding object (and so one loaded model) per Python process, shared by
    every browser session. `api_key` is accepted only so existing callers keep
    working; local embeddings do not use it."""
    return _shared_embeddings()


def open_store(embeddings, persist_dir: Optional[str] = None, name: Optional[str] = None) -> Chroma:
    """Opens (or creates) a persistent collection. Cosine distance, so the
    'distance' shown in the UI is 1 - cosine similarity. The folder defaults to
    CHROMA_PERSIST_DIR and the name to COLLECTION_NAME (both read at call time)."""
    return Chroma(
        collection_name=name or COLLECTION_NAME, embedding_function=embeddings,
        persist_directory=persist_dir or CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def count(store: Chroma) -> int:
    return store._collection.count()


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------

def add_chunks(store: Chroma, chunks: list) -> tuple[int, int]:
    """Adds ONLY chunks whose chunk_id is not in Chroma yet (so nothing already
    indexed is embedded again). Returns (added, already_present)."""
    ids = [c.metadata["chunk_id"] for c in chunks]
    existing: set = set()
    for i in range(0, len(ids), 500):
        existing |= set(store.get(ids=ids[i:i + 500])["ids"])
    new = [(cid, c) for cid, c in zip(ids, chunks) if cid not in existing]
    for i in range(0, len(new), EMBED_BATCH):
        batch = new[i:i + EMBED_BATCH]
        store.add_documents([c for _, c in batch], ids=[cid for cid, _ in batch])
    return len(new), len(chunks) - len(new)


def delete_source(store: Chroma, source: str) -> int:
    """Removes only the chunks of one PDF. Other PDFs are untouched."""
    ids = store.get(where={"source": source})["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)


def load_documents(store: Chroma) -> "OrderedDict[str, list]":
    """Reads the chunks back out of Chroma, grouped per PDF, so the app can
    restore its document list after a restart (persistence)."""
    got = store.get(include=["documents", "metadatas"])
    by_source: "OrderedDict[str, list]" = OrderedDict()
    for text, md in zip(got["documents"], got["metadatas"]):
        by_source.setdefault(md.get("source", "document"), []).append(Document(page_content=text, metadata=dict(md)))
    for chunks in by_source.values():
        chunks.sort(key=lambda d: d.metadata.get("page", 0))       # stable: keeps order within a page
    return by_source


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------

def retrieve_vector_evidence(store: Chroma, question: str, k: int = 6, max_per_pdf: int = 2,
                             sources: Optional[list] = None) -> list:
    """Global semantic search over ALL indexed PDFs (optionally restricted to
    `sources`). Keeps Chroma's distance for every hit. No PDF is forced into
    the result: chunks are taken best-first, but one PDF may contribute at most
    `max_per_pdf`, so a single verbose document cannot crowd out the rest."""
    n = count(store)
    if n == 0:
        return []
    flt = {"source": {"$in": list(sources)}} if sources else None

    def _search(fetch: int) -> list:
        return store.similarity_search_with_score(question, k=min(fetch, n), filter=flt)

    def _select(hits: list) -> list:
        per_pdf: dict = {}
        picked = []
        for doc, dist in hits:
            src = doc.metadata.get("source")
            if per_pdf.get(src, 0) >= max_per_pdf:
                continue
            per_pdf[src] = per_pdf.get(src, 0) + 1
            picked.append(evidence_from_document(doc, distance=float(dist)))
            if len(picked) >= k:
                break
        return picked

    fetch = max(k * 5, 30)
    picked = _select(_search(fetch))
    if len(picked) < k and fetch < n:          # a cap-heavy top list: look deeper once
        picked = _select(_search(n))
    return picked

def open_entity_store(embeddings):
    """
    Open the existing persistent entity embedding collection
    without deleting or rebuilding it.
    """
    return Chroma(
        collection_name=ENTITY_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Entity index (Local Search)
# ---------------------------------------------------------------------------

def rebuild_entity_index(embeddings, kg: KnowledgeGraph, persist_dir: Optional[str] = None,
                         name: Optional[str] = None) -> Optional[Chroma]:
    """One embedding per entity: 'name (type): first description'. Rebuilt
    whenever the graph is rebuilt (cheap: one batched call per 100 entities)."""
    name = name or ENTITY_COLLECTION
    store = open_store(embeddings, persist_dir, name)
    try:
        store.delete_collection()
    except Exception:
        pass
    store = open_store(embeddings, persist_dir, name)
    docs, ids = [], []
    for key, data in kg.graph.nodes(data=True):
        desc = (data.get("descriptions") or [{"text": ""}])[0]["text"]
        docs.append(Document(page_content=f"{data['name']} ({data.get('type', 'other')}): {desc}".strip(),
                             metadata={"key": key}))
        ids.append(key)
    if not docs:
        return None
    for i in range(0, len(docs), EMBED_BATCH):
        store.add_documents(docs[i:i + EMBED_BATCH], ids=ids[i:i + EMBED_BATCH])
    return store


def semantic_entity_matches(entity_store: Optional[Chroma], question: str, kg: KnowledgeGraph,
                            top_k: int = ENTITY_MATCH_TOP_K,
                            max_distance: float = ENTITY_MATCH_MAX_DISTANCE) -> list:
    """Entities whose embedding is close to the question. match_type='semantic'."""
    if entity_store is None or count(entity_store) == 0:
        return []
    out = []
    for doc, dist in entity_store.similarity_search_with_score(question, k=top_k):
        key = doc.metadata.get("key")
        if key in kg.graph.nodes and dist <= max_distance:
            out.append(MatchedEntity(node_key=key, display_name=kg.graph.nodes[key]["name"],
                                     match_type="semantic", matched_text=f"distance {dist:.2f}"))
    return out