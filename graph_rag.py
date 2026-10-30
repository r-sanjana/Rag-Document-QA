"""
graph_rag.py — GraphRAG-style extension for the RAG Document Q&A app.

This is a "GraphRAG-style" approach (knowledge graph + local entity-neighborhood
retrieval). It is NOT Microsoft's full GraphRAG implementation.

STEP 1 (this file, so far): ENTITY & RELATIONSHIP EXTRACTION
-------------------------------------------------------------
Turns one existing text chunk into structured, provenance-tagged graph data:

    chunk text ──► Gemini (structured output) ──► validation ──► ChunkGraphData
                                                                  ├─ entities
                                                                  └─ triples (subject, predicate, object)

Nothing here touches the vector pipeline (Chroma, embeddings, retrieval) and
nothing here imports Streamlit, so it can be tested on its own:

    python graph_rag.py                                  # built-in 2-chunk Motor A example
    python graph_rag.py --build-graph                    # ...and merge it into a graph
    python graph_rag.py --pdf your.pdf --start 10 --max-chunks 3 --build-graph

STEP 2 (added below): GRAPH CONSTRUCTION
-----------------------------------------
Merges the per-chunk results into ONE NetworkX graph:

    [ChunkGraphData, ChunkGraphData, ...] ──► build_knowledge_graph() ──► KnowledgeGraph
       (Step 1 output, one per chunk)              nodes = entities (merged by normalized name)
                                                   edges = facts (merged by subject/predicate/object)
                                                   every node & edge keeps its provenance

STEP 3A: GRAPH RETRIEVAL (question entity matching + neighborhood traversal)
STEP 3B: GRAPH CONTEXT FORMATTING + GRAPH-RAG ANSWER GENERATION (used by app.py)

The Streamlit UI lives in app.py; this module never imports Streamlit.
"""

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import networkx as nx
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Entity types we accept. The LLM is asked to pick one; anything else becomes
# "other". Keeping this small makes the graph easier to read and to filter.
ALLOWED_ENTITY_TYPES = {
    "component", "system", "person", "organization",
    "document", "process", "concept", "parameter", "other",
}

# Chunks shorter than this (after stripping) carry no real information, so we
# skip them without spending an LLM call.
MIN_CHUNK_CHARS = 40

# A predicate should be a short verb phrase (controlled_by), not a sentence.
MAX_PREDICATE_CHARS = 40

# Words that make a predicate NEGATIVE ("unrelated_to", "is_not_part_of").
# A graph edge means "these two are connected", so a negative statement must
# not become an edge: neighborhood traversal would follow it and wrongly pull
# in exactly the thing the text says is NOT related. Matched per word, so
# "notified_by" is fine but "not_connected_to" is dropped.
NEGATIVE_PREDICATE_WORDS = {"not", "no", "non", "never", "cannot", "unrelated"}

# Words that are never real entities. "It is powered by PSU-1" must not create
# a node called "It".
GENERIC_ENTITY_NAMES = {
    "it", "its", "this", "that", "these", "those", "they", "them",
    "he", "she", "we", "you", "i", "which", "who",
    "the system", "the device", "the unit", "the component",
    "the document", "this section",
}

# Cache versions. The extraction / report caches are keyed by chunk_id (or by a
# content signature). If the prompt or schema changes in a way that makes old
# cached results wrong, bump the version and the old cache file is ignored.
EXTRACTION_VERSION = "2"   # v2: entities/relationships carry a `description`
REPORT_VERSION = "1"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
#
# Two groups of models, on purpose:
#
# 1. What the LLM RETURNS (ExtractedEntity / ExtractedRelationship /
#    ChunkExtraction). These contain NO provenance. The LLM is never asked
#    for source/page/chunk id, because it could invent them.
#
# 2. What OUR CODE produces after validation (GraphEntity / GraphTriple /
#    ChunkGraphData). Provenance is attached here, from the chunk's own
#    metadata, so it is always correct.
#
# Note: Optional fields have no default; the code always passes them
# explicitly (page=None when there is no page).

class ExtractedEntity(BaseModel):
    name: str = Field(description="The entity's name exactly as written in the chunk, e.g. 'Motor A'.")
    type: str = Field(description="One of: component, system, person, organization, document, process, concept, parameter, other.")
    description: str = Field(description="One short sentence (max 30 words) saying what the chunk states about this entity, using only the chunk. Empty string if the chunk says nothing about it.")


class ExtractedRelationship(BaseModel):
    subject: str = Field(description="Entity name exactly as written in the chunk (what the fact is about).")
    predicate: str = Field(description="Short snake_case verb phrase that keeps the direction written in the text, e.g. controlled_by, powered_by, connected_to, part_of.")
    object: str = Field(description="Entity name exactly as written in the chunk (the other end of the relationship).")
    evidence: str = Field(description="The shortest exact quote (one sentence or less) copied from the chunk that mentions BOTH the subject and the object and states this relationship.")
    description: str = Field(description="One short sentence (max 30 words) describing the relationship, using only the chunk.")


class ChunkExtraction(BaseModel):
    """The raw structured answer we ask Gemini for, per chunk."""
    entities: list[ExtractedEntity] = Field(description="Named entities explicitly present in the chunk. Empty list if none.")
    relationships: list[ExtractedRelationship] = Field(description="Relationships explicitly stated in the chunk. Empty list if none.")


class GraphEntity(BaseModel):
    """A validated entity. `key` is the normalized name that will later be
    used to merge the same entity found in different chunks into one node."""
    name: str
    key: str
    type: str
    description: str = ""    # LLM-written short summary of what the chunk says about the entity


class GraphTriple(BaseModel):
    """One validated fact (subject -[predicate]-> object) plus provenance."""
    subject: str
    predicate: str
    object: str
    subject_key: str
    object_key: str
    evidence: str            # exact quote from the chunk supporting this fact
    source: str              # PDF file name
    page: Optional[int]      # 1-based page number (human page), or None
    chunk_id: str            # stable id of the chunk this fact came from
    description: str = ""    # LLM-written short description of the relationship


class ChunkGraphData(BaseModel):
    """Everything extracted from ONE chunk, self-contained: the original
    chunk text is kept so graph facts can later be shown next to the text
    they came from."""
    chunk_id: str
    source: str
    page: Optional[int]      # 1-based page number, or None
    chunk_text: str
    entities: list[GraphEntity]
    triples: list[GraphTriple]
    status: str              # "ok" | "empty" | "skipped" | "error"
    error: Optional[str]
    dropped: list[str]       # facts the LLM proposed that failed a support check


# --------------------------------------------------------------------------- #
# Small deterministic helpers (no LLM involved)
# --------------------------------------------------------------------------- #

def normalize_text(text: str) -> str:
    """Lowercase, and turn every run of punctuation/whitespace into one space.
    'PSU-1', 'psu 1' and 'PSU  1' all become 'psu 1'. Used both to compare
    text and to build entity keys."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _clean_desc(text: Optional[str]) -> str:
    """Descriptions are LLM summaries (NOT verbatim quotes), so they are only
    whitespace-normalised and length-capped. The evidence quote is the part
    that is checked against the chunk text."""
    return " ".join((text or "").split())[:300]


def _appears_in(needle_norm: str, haystack_norm: str) -> bool:
    """True if `needle_norm` occurs in `haystack_norm` as whole tokens
    (so 'a' does not match inside 'motor a')."""
    return bool(needle_norm) and f" {needle_norm} " in f" {haystack_norm} "


def make_chunk_id(chunk: Document) -> str:
    """Deterministic id like 'manual.pdf|p3|9f2c41ab'. Built from the chunk's
    own source, page and text, so the same chunk always gets the same id and
    the existing chunks/metadata do not need to be modified."""
    source = chunk.metadata.get("source", "document")
    page0 = chunk.metadata.get("page")           # 0-based in the existing app
    page_label = f"p{page0 + 1}" if page0 is not None else "p?"
    digest = hashlib.sha1(f"{source}|{page0}|{chunk.page_content}".encode("utf-8")).hexdigest()[:8]
    return f"{source}|{page_label}|{digest}"


def _provenance(chunk: Document) -> tuple[str, str, Optional[int]]:
    """(chunk_id, source, 1-based page). The existing app stores pages
    0-based in metadata and adds 1 only when displaying; here we store the
    human page number directly so later steps never have to remember +1."""
    source = chunk.metadata.get("source", "document")
    page0 = chunk.metadata.get("page")
    page = page0 + 1 if page0 is not None else None
    return make_chunk_id(chunk), source, page


def _result(chunk: Document, status: str, error: Optional[str] = None) -> ChunkGraphData:
    """An empty result (nothing extracted) for a chunk."""
    chunk_id, source, page = _provenance(chunk)
    return ChunkGraphData(
        chunk_id=chunk_id, source=source, page=page, chunk_text=chunk.page_content,
        entities=[], triples=[], status=status, error=error, dropped=[],
    )


# --------------------------------------------------------------------------- #
# Validation: turn the LLM's raw answer into trusted graph data
# --------------------------------------------------------------------------- #

def validate_extraction(raw: ChunkExtraction, chunk: Document) -> ChunkGraphData:
    """Keeps only what the chunk text actually supports, attaches provenance,
    removes duplicates and sorts everything (so the output is stable).

    A proposed relationship is DROPPED unless ALL of these hold:
      - subject, predicate and object are non-empty, and not pronouns
      - subject and object are different entities
      - the predicate is a short verb phrase, and not a negative one
        ('unrelated_to', 'is_not_part_of' would wrongly connect entities)
      - subject and object both appear in the chunk text
      - the evidence quote appears in the chunk text
      - the evidence quote mentions BOTH subject and object
    The last check is what blocks "inferred" facts: from 'A is controlled by B.
    B receives power from C.' a model might claim (A, powered_by, C); the
    quote for that would not mention A, so it is rejected.

    These are cheap guards that reduce unsupported facts; they do not make
    extraction perfect (see the limitations noted in the docs of this step).
    """
    chunk_id, source, page = _provenance(chunk)
    chunk_norm = normalize_text(chunk.page_content)
    dropped: list[str] = []

    # ---- entities -------------------------------------------------------
    entities: dict[str, GraphEntity] = {}
    for ent in raw.entities:
        key = normalize_text(ent.name)
        if not key or key in GENERIC_ENTITY_NAMES:
            dropped.append(f"entity '{ent.name}': empty or generic word")
            continue
        if not _appears_in(key, chunk_norm):
            dropped.append(f"entity '{ent.name}': not found in the chunk text")
            continue
        etype = ent.type.strip().lower()
        if etype not in ALLOWED_ENTITY_TYPES:
            etype = "other"
        entities.setdefault(key, GraphEntity(name=ent.name.strip(), key=key, type=etype,
                                             description=_clean_desc(ent.description)))

    # ---- relationships --------------------------------------------------
    triples: dict[tuple[str, str, str], GraphTriple] = {}
    for rel in raw.relationships:
        label = f"{rel.subject} -[{rel.predicate}]-> {rel.object}"
        s_key = normalize_text(rel.subject)
        o_key = normalize_text(rel.object)
        predicate = re.sub(r"[\W_]+", "_", rel.predicate.lower()).strip("_")
        evidence_norm = normalize_text(rel.evidence)

        reason = None
        if not s_key or not o_key or not predicate:
            reason = "empty subject, predicate or object"
        elif s_key in GENERIC_ENTITY_NAMES or o_key in GENERIC_ENTITY_NAMES:
            reason = "subject or object is a pronoun / generic word"
        elif s_key == o_key:
            reason = "subject and object are the same entity"
        elif any(word in NEGATIVE_PREDICATE_WORDS for word in predicate.split("_")):
            reason = "negative relationship (an edge would wrongly link the two entities)"
        elif len(predicate) > MAX_PREDICATE_CHARS:
            reason = "predicate is too long to be a short verb phrase"
        elif not (_appears_in(s_key, chunk_norm) and _appears_in(o_key, chunk_norm)):
            reason = "subject or object does not appear in the chunk text"
        elif not evidence_norm or not _appears_in(evidence_norm, chunk_norm):
            reason = "evidence is not a quote from the chunk"
        elif not (_appears_in(s_key, evidence_norm) and _appears_in(o_key, evidence_norm)):
            reason = "evidence does not mention both subject and object"

        if reason:
            dropped.append(f"{label}: {reason}")
            continue

        triples.setdefault(
            (s_key, predicate, o_key),
            GraphTriple(
                subject=rel.subject.strip(), predicate=predicate, object=rel.object.strip(),
                subject_key=s_key, object_key=o_key, evidence=rel.evidence.strip(),
                source=source, page=page, chunk_id=chunk_id,
                description=_clean_desc(rel.description),
            ),
        )
        # Every fact's endpoints must exist as entities, even if the LLM
        # forgot to list them, so graph construction never meets a dangling edge.
        for name, key in ((rel.subject, s_key), (rel.object, o_key)):
            entities.setdefault(key, GraphEntity(name=name.strip(), key=key, type="other"))

    entity_list = [entities[k] for k in sorted(entities)]
    triple_list = [triples[k] for k in sorted(triples)]
    return ChunkGraphData(
        chunk_id=chunk_id, source=source, page=page, chunk_text=chunk.page_content,
        entities=entity_list, triples=triple_list,
        status="ok" if (entity_list or triple_list) else "empty",
        error=None, dropped=sorted(dropped),
    )


# --------------------------------------------------------------------------- #
# The Gemini extraction call
# --------------------------------------------------------------------------- #

EXTRACTION_PROMPT = """You extract a small knowledge graph from ONE chunk of a technical document.

Rules:
1. Extract ONLY what the chunk explicitly states. Never use outside knowledge and never infer
   or combine facts (do not chain 'A -> B' and 'B -> C' into 'A -> C').
2. Entities: specific named things (components, systems, people, organizations, documents,
   processes, concepts, parameters). Write each name exactly as it appears in the chunk.
   Skip pronouns and generic words such as 'it', 'this', 'the system'.
3. Relationships: (subject, predicate, object). Subject and object must be entities named
   explicitly in the chunk. If the subject is a pronoun, skip the relationship.
4. Predicate: a short lowercase snake_case verb phrase that keeps the direction written in the
   text. 'Motor A is controlled by Controller B' -> subject 'Motor A', predicate
   'controlled_by', object 'Controller B'. Examples: controlled_by, powered_by, connected_to,
   part_of, contains, depends_on, causes, uses, defined_as, located_in.
5. Evidence: copy the shortest exact quote (one sentence or less) from the chunk that mentions
   BOTH the subject and the object and states the relationship.
6. Extract only POSITIVE relationships that connect two things. Do NOT extract negative
   statements such as 'X is unrelated to Y' or 'X does not use Y'; skip them.
7. If the chunk has no meaningful entities or relationships (headings, page furniture,
   instructions, questions, marks, boilerplate), return empty lists. An empty result is
   correct and is preferred over guessing.
8. Entity description: one short sentence (max 30 words) saying what the chunk states about that
   entity, using only the chunk's own information. Use an empty string if the chunk says nothing.
9. Relationship description: one short sentence (max 30 words) describing the relationship,
   using only the chunk's own information.

Chunk:
\"\"\"
{chunk_text}
\"\"\""""


def build_extraction_chain(model_name: str, api_key: str):
    """prompt | Gemini with structured output (ChunkExtraction). Same pattern
    as the judge in app.py. temperature=0 is requested for repeatability;
    some Gemini models ignore it, which is why validate_extraction also
    sorts and de-duplicates everything."""
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    structured_model = model.with_structured_output(ChunkExtraction)
    return ChatPromptTemplate.from_template(EXTRACTION_PROMPT) | structured_model


def extract_from_chunk(chunk: Document, chain, retries: int = 2) -> ChunkGraphData:
    """Extracts graph data from a single chunk. Never raises: failures are
    returned as status='error' so one bad chunk cannot stop a whole run.

    - too-short chunk        -> status 'skipped' (no LLM call)
    - nothing valid found    -> status 'empty'
    - LLM/API failure        -> retried, then status 'error'
    """
    if len(chunk.page_content.strip()) < MIN_CHUNK_CHARS:
        return _result(chunk, "skipped")

    raw = None
    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            raw = chain.invoke({"chunk_text": chunk.page_content})
            if raw is not None:
                break
            last_error = "model returned no structured output"
        except Exception as e:  # rate limit, network, parse error, ...
            last_error = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))  # gentle backoff for free-tier rate limits

    if raw is None:
        return _result(chunk, "error", last_error)
    return validate_extraction(raw, chunk)


class _RateLimiter:
    """Shared by all extraction workers: guarantees at least `min_interval` seconds between the
    START of two Gemini calls, however many workers there are. With min_interval=4 that is at
    most 15 requests per minute (the usual free-tier limit). min_interval=0 disables it."""

    def __init__(self, min_interval: float, clock=time.monotonic, sleep=time.sleep):
        self.min_interval, self._clock, self._sleep = max(0.0, float(min_interval)), clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:                       # reserve the next start slot, then wait for it outside the lock
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.min_interval
        if start > now:
            self._sleep(start - now)


def _needs_llm(chunk: Document) -> bool:
    return len(chunk.page_content.strip()) >= MIN_CHUNK_CHARS


def extract_graph_data(
    chunks: list[Document],
    model_name: str,
    api_key: str,
    max_chunks: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, ChunkGraphData], None]] = None,
    chain=None,
    cache: Optional[dict] = None,
    delay_between_calls: float = 0.0,
    max_workers: int = 1,
    on_checkpoint: Optional[Callable[[], None]] = None,
    checkpoint_every: int = 25,
) -> list[ChunkGraphData]:
    """Runs extraction over many chunks (one Gemini call per chunk) and returns the results in
    the SAME ORDER as `chunks`.

    max_chunks   limit the run while testing, to save API calls
    on_progress  called as on_progress(done, total, result) whenever a chunk finishes (cached
                 chunks count immediately). Always called from the CALLING thread, never from a
                 worker, so it is safe to update a Streamlit progress bar from it.
    chain        optional pre-built chain (used for testing without an API)
    cache        optional dict chunk_id -> ChunkGraphData. A chunk already in the cache is NOT
                 sent to Gemini again. New successful results (status ok/empty/skipped) are
                 added. Failed chunks (status 'error') are never cached, so running the same call
                 again retries exactly the failed ones. Only the calling thread writes to it.
    delay_between_calls  minimum seconds between the START of two Gemini calls, shared by all
                 workers (never applied to cached or skipped chunks). Free tier: ~15 requests
                 per minute -> use 4.
    max_workers  chunks extracted at the same time (threads). 1 = one after another. More is
                 faster until the per-minute quota is hit; then you get 429 failures instead.
    on_checkpoint  called every `checkpoint_every` newly extracted chunks and once at the end
                 (also when the run is interrupted, e.g. the user presses Stop) so progress can
                 be saved to disk.
    """
    selected = chunks[:max_chunks] if max_chunks else chunks
    total = len(selected)
    results: list = [None] * total
    done = 0
    fresh = 0

    def finish(i: int, result: ChunkGraphData, cid: str, was_new: bool) -> None:
        nonlocal done, fresh
        results[i] = result
        if was_new:
            fresh += 1
            if cache is not None and result.status != "error":
                cache[cid] = result
        done += 1
        if on_progress:
            on_progress(done, total, result)
        if was_new and on_checkpoint and checkpoint_every and fresh % checkpoint_every == 0:
            on_checkpoint()

    # 1) cached chunks are finished immediately; 2) the rest are queued for extraction
    todo: list = []
    for i, chunk in enumerate(selected):
        cid = make_chunk_id(chunk)
        if cache is not None and cid in cache:
            finish(i, cache[cid], cid, was_new=False)
        else:
            todo.append((i, chunk, cid))

    try:
        if todo:
            if chain is None:
                chain = build_extraction_chain(model_name, api_key)
            limiter = _RateLimiter(delay_between_calls)

            def work(chunk: Document) -> ChunkGraphData:
                if _needs_llm(chunk):
                    limiter.wait()
                return extract_from_chunk(chunk, chain)

            workers = max(1, int(max_workers))
            if workers == 1:
                for i, chunk, cid in todo:
                    finish(i, work(chunk), cid, was_new=True)
            else:
                pool = ThreadPoolExecutor(max_workers=workers)
                try:
                    futures = {pool.submit(work, chunk): (i, cid) for i, chunk, cid in todo}
                    for fut in as_completed(futures):
                        i, cid = futures[fut]
                        finish(i, fut.result(), cid, was_new=True)
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)     # on Stop: don't start queued chunks
    finally:
        if on_checkpoint and fresh:
            on_checkpoint()
    return results


def split_cached(chunks: list[Document], cache: Optional[dict]) -> tuple[int, int]:
    """(already_cached, needs_extraction) for a list of chunks."""
    if not cache:
        return 0, len(chunks)
    hit = sum(make_chunk_id(c) in cache for c in chunks)
    return hit, len(chunks) - hit


def failed_chunks(results: list[ChunkGraphData]) -> list[str]:
    """chunk_ids whose extraction failed (API error, malformed output...)."""
    return [r.chunk_id for r in results if r.status == "error"]


def load_cache(path: str, version: str, model_cls) -> dict:
    """Loads a JSON cache file written by save_cache(). Returns {} if the file
    is missing, unreadable or from a different cache version — a broken cache
    must never crash the app, it just means the work is redone."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("version") != version:
            return {}
        return {k: model_cls.model_validate(v) for k, v in payload["items"].items()}
    except Exception:
        return {}


def save_cache(cache: dict, path: str, version: str) -> None:
    """Best-effort write (a read-only disk only loses the persistence)."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": version, "items": {k: v.model_dump() for k, v in cache.items()}}, f)
        os.replace(tmp, path)
    except Exception:
        pass


def summarize_extraction(results: list[ChunkGraphData]) -> dict:
    """Quick counts for a run. 'entity_mentions' counts entities per chunk,
    BEFORE the same entity from different chunks is merged into one node
    (that merge happens in the graph-construction step)."""
    return {
        "chunks": len(results),
        "ok": sum(r.status == "ok" for r in results),
        "empty": sum(r.status == "empty" for r in results),
        "skipped": sum(r.status == "skipped" for r in results),
        "error": sum(r.status == "error" for r in results),
        "entity_mentions": sum(len(r.entities) for r in results),
        "triples": sum(len(r.triples) for r in results),
        "dropped_as_unsupported": sum(len(r.dropped) for r in results),
    }


# --------------------------------------------------------------------------- #
# STEP 2: GRAPH CONSTRUCTION
# --------------------------------------------------------------------------- #
#
# Merges every chunk's ChunkGraphData into ONE graph:
#   - one node per normalized entity key (an entity mentioned in 3 chunks
#     becomes ONE node with 3 provenance entries, not 3 separate nodes)
#   - one edge per unique (subject_key, predicate, object_key); if the same
#     fact is stated in more than one chunk, that's still one edge, with
#     more than one provenance entry
#
# Design choice: a node/edge's DISPLAY name is whichever exact wording was
# most common across all its mentions (ties broken alphabetically, so the
# result is deterministic and reproducible — important given Step 1's
# output can vary run to run, per your test results).

@dataclass
class KnowledgeGraph:
    """Thin wrapper around a NetworkX graph plus a chunk lookup.

    graph: nx.MultiDiGraph — nodes keyed by normalized entity `key`
      (node attrs: name, type, mentions, descriptions);
      edges keyed by `predicate` (edge attrs: predicate, provenance — a list
      of {source, page, chunk_id, evidence, description} dicts, one per chunk that stated
      this fact).
    chunk_lookup: chunk_id -> the ChunkGraphData it came from (every chunk
      passed to build_knowledge_graph, not just ones with facts) — this is
      what lets a later step recover original chunk TEXT from a fact's
      provenance, per your "original source chunks" requirement.
    """
    graph: nx.MultiDiGraph
    chunk_lookup: dict[str, ChunkGraphData]


def _pick_majority(counter: "Counter[str]", default: str) -> str:
    """Most common value; ties broken alphabetically, so re-running on
    identical input always picks the same one."""
    if not counter:
        return default
    best = max(counter.values())
    return sorted(k for k, v in counter.items() if v == best)[0]


def build_knowledge_graph(results: list[ChunkGraphData]) -> KnowledgeGraph:
    """Builds one KnowledgeGraph from many chunks' extraction results.

    Chunks with status 'error' or 'skipped' contribute nothing (they have no
    entities/triples to add), but are still kept in chunk_lookup so their
    text can be recovered if ever needed. Re-running this on the same
    `results` always produces an identical graph (see the ordering/tie-break
    choices below) — needed because we'll cache/save this graph rather than
    rebuild it on every question.
    """
    graph = nx.MultiDiGraph()
    chunk_lookup: dict[str, ChunkGraphData] = {}

    entity_names: dict[str, Counter] = {}
    entity_types: dict[str, Counter] = {}
    entity_mentions: dict[str, set] = {}
    entity_desc: dict[str, dict[str, str]] = {}      # key -> {description text: chunk_id}
    edge_provenance: dict[tuple[str, str, str], list[dict]] = {}

    def register_entity(key: str, name: str, source: str, page, chunk_id: str, etype: Optional[str] = None,
                        description: str = ""):
        if description:
            prev = entity_desc.setdefault(key, {}).get(description)
            if prev is None or chunk_id < prev:
                entity_desc[key][description] = chunk_id
        entity_names.setdefault(key, Counter())[name] += 1
        if etype:
            entity_types.setdefault(key, Counter())[etype] += 1
        else:
            entity_types.setdefault(key, Counter())
        entity_mentions.setdefault(key, set()).add((source, page, chunk_id))

    for res in results:
        chunk_lookup[res.chunk_id] = res
        for ent in res.entities:
            register_entity(ent.key, ent.name, res.source, res.page, res.chunk_id, ent.type, ent.description)
        for tri in res.triples:
            # Every triple's endpoints are registered as entities too (with
            # no type vote), in case a fact's subject/object was for some
            # reason absent from res.entities — this guarantees every edge
            # in the final graph always has both its endpoint nodes.
            register_entity(tri.subject_key, tri.subject, tri.source, tri.page, tri.chunk_id)
            register_entity(tri.object_key, tri.object, tri.source, tri.page, tri.chunk_id)
            k = (tri.subject_key, tri.predicate, tri.object_key)
            edge_provenance.setdefault(k, []).append(
                {"source": tri.source, "page": tri.page, "chunk_id": tri.chunk_id, "evidence": tri.evidence,
                 "description": tri.description}
            )

    for key in sorted(entity_names):
        name = _pick_majority(entity_names[key], key)
        etype = _pick_majority(entity_types.get(key, Counter()), "other")
        mentions = sorted(entity_mentions.get(key, set()), key=lambda m: (m[0], m[1] if m[1] is not None else -1, m[2]))
        descriptions = [{"text": t, "chunk_id": c} for t, c in
                        sorted(entity_desc.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]))][:5]
        graph.add_node(key, name=name, type=etype, mentions=mentions, descriptions=descriptions)

    for (s_key, predicate, o_key) in sorted(edge_provenance):
        prov = edge_provenance[(s_key, predicate, o_key)]
        seen = set()
        deduped = []
        for p in sorted(prov, key=lambda p: (p["chunk_id"], p["evidence"])):
            sig = (p["chunk_id"], p["evidence"])
            if sig not in seen:
                seen.add(sig)
                deduped.append(p)
        graph.add_edge(s_key, o_key, key=predicate, predicate=predicate, provenance=deduped)

    return KnowledgeGraph(graph=graph, chunk_lookup=chunk_lookup)


def graph_stats(kg: KnowledgeGraph) -> dict:
    """Summary numbers to sanity-check a built graph (used by --build-graph
    and, later, by the Streamlit 'Build Knowledge Graph' progress display)."""
    g = kg.graph
    isolated = [n for n in g.nodes if g.degree(n) == 0]
    components = list(nx.connected_components(g.to_undirected()))
    degrees = sorted(
        ((g.in_degree(n) + g.out_degree(n)), g.nodes[n]["name"]) for n in g.nodes
    )
    top_entities = [(name, deg) for deg, name in reversed(degrees[-5:])]
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "isolated_nodes": len(isolated),
        "connected_components": len(components),
        "largest_component_size": max((len(c) for c in components), default=0),
        "top_entities_by_degree": top_entities,
    }


# --------------------------------------------------------------------------- #
# STEP 3A: GRAPH RETRIEVAL — question entity matching + neighborhood traversal
# --------------------------------------------------------------------------- #
#
# Two-stage, both deterministic (no LLM call — cheap, and needed since this
# runs on every question, unlike extraction which runs once per chunk):
#
#   1. find_matching_nodes()   question text -> which graph nodes does it
#                               name? Tries exact/substring whole-word
#                               matching FIRST; only falls back to fuzzy
#                               matching if nothing matched directly, and
#                               marks fuzzy matches as such (match_type)
#                               rather than silently treating them the same
#                               as a real match.
#
#   2. get_neighborhood_facts() from those starting nodes, walk outward up
#                               to `hops` steps, collecting every edge
#                               touched. Traversal is BIDIRECTIONAL — it
#                               follows edges in both directions (in_edges
#                               AND out_edges) — because a question naming
#                               PSU-1 (the OBJECT of "Controller B
#                               -[receives_power_from]-> PSU-1") must still
#                               find that fact; a directed-only walk
#                               starting at PSU-1 would find nothing, since
#                               PSU-1 has no outgoing edges in that fact.

@dataclass
class MatchedEntity:
    node_key: str
    display_name: str
    match_type: str    # "exact" | "substring" | "fuzzy"
    matched_text: str  # the question text (or node key, for fuzzy) that triggered this match


@dataclass
class RetrievedFact:
    subject: str
    predicate: str
    object: str
    subject_key: str
    object_key: str
    hop: int            # 1 = directly touches a starting node, 2 = one step further out
    provenance: list[dict]


def find_matching_nodes(question: str, kg: KnowledgeGraph, fuzzy_threshold: float = 0.72) -> list[MatchedEntity]:
    """Finds which graph nodes the question is talking about.

    Stage 1 (exact/substring): every node key is checked as a WHOLE-WORD
    substring of the normalized question (reusing the same `_appears_in`
    whole-word check Step 1 uses to verify entities against chunk text) —
    so a node key 'ai' will not wrongly match inside the question word
    'again'. Longer keys are checked... but since matches are keyed by
    node_key in a dict, a short key and a long key that both match just
    both end up in the result; there's no "longer wins" suppression needed
    since e.g. 'controller b' matching doesn't prevent 'psu 1' from also
    matching elsewhere in the same question, which is exactly wanted for a
    question naming two entities.

    Stage 2 (fuzzy, ONLY if stage 1 found nothing): guards against a typo
    ("Contorller B") losing the match entirely. Uses Python's built-in
    difflib against the question's individual words and short word-phrases,
    above `fuzzy_threshold` similarity. Every fuzzy match is tagged
    match_type='fuzzy' so a caller can choose to warn the user or require
    confirmation — this is the "do not silently match unrelated entities"
    requirement: fuzzy hits are visible, not indistinguishable from real ones.
    """
    q_norm = normalize_text(question)
    matches: dict[str, MatchedEntity] = {}

    for key in kg.graph.nodes:
        if key and _appears_in(key, q_norm):
            match_type = "exact" if key == q_norm else "substring"
            matches[key] = MatchedEntity(
                node_key=key, display_name=kg.graph.nodes[key]["name"],
                match_type=match_type, matched_text=key,
            )

    if matches:
        return sorted(matches.values(), key=lambda m: m.node_key)

    import difflib
    q_tokens = q_norm.split()
    phrases = list(q_tokens)
    phrases += [" ".join(q_tokens[i:i + 3]) for i in range(len(q_tokens))]  # short phrases, for multi-word keys
    for key in kg.graph.nodes:
        if not key:
            continue
        best = difflib.get_close_matches(key, phrases, n=1, cutoff=fuzzy_threshold)
        if best:
            matches[key] = MatchedEntity(
                node_key=key, display_name=kg.graph.nodes[key]["name"],
                match_type="fuzzy", matched_text=best[0],
            )
    return sorted(matches.values(), key=lambda m: m.node_key)


def get_neighborhood_facts(kg: KnowledgeGraph, matched: list[MatchedEntity], hops: int = 2) -> list[RetrievedFact]:
    """Bidirectional breadth-first walk outward from the matched nodes, up
    to `hops` steps, returning every edge encountered along the way,
    tagged with how many hops out it was found."""
    if hops not in (1, 2):
        raise ValueError("hops must be 1 or 2 — unrestricted traversal is out of scope for this POC")

    g = kg.graph
    start_keys = {m.node_key for m in matched if m.node_key in g.nodes}
    visited = set(start_keys)
    frontier = set(start_keys)
    facts: dict[tuple[str, str, str], RetrievedFact] = {}

    for hop in range(1, hops + 1):
        next_frontier = set()
        for node in frontier:
            for _, nbr, data in g.out_edges(node, data=True):
                key = (node, data["predicate"], nbr)
                facts.setdefault(key, RetrievedFact(
                    subject=g.nodes[node]["name"], predicate=data["predicate"], object=g.nodes[nbr]["name"],
                    subject_key=node, object_key=nbr, hop=hop, provenance=data["provenance"],
                ))
                if nbr not in visited:
                    next_frontier.add(nbr)
            for pred_node, _, data in g.in_edges(node, data=True):
                key = (pred_node, data["predicate"], node)
                facts.setdefault(key, RetrievedFact(
                    subject=g.nodes[pred_node]["name"], predicate=data["predicate"], object=g.nodes[node]["name"],
                    subject_key=pred_node, object_key=node, hop=hop, provenance=data["provenance"],
                ))
                if pred_node not in visited:
                    next_frontier.add(pred_node)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    return sorted(facts.values(), key=lambda f: (f.hop, f.subject_key, f.predicate, f.object_key))


def graph_retrieve(
    question: str, kg: KnowledgeGraph, hops: int = 2, fuzzy_threshold: float = 0.72
) -> tuple[list[MatchedEntity], list[RetrievedFact]]:
    """Convenience wrapper: match, then traverse. Returns ([], []) if the
    question doesn't name anything in the graph — this is the signal the
    Streamlit layer (Step 5) will use to say 'nothing relevant found' rather
    than silently returning an empty answer."""
    matched = find_matching_nodes(question, kg, fuzzy_threshold)
    facts = get_neighborhood_facts(kg, matched, hops) if matched else []
    return matched, facts


# --------------------------------------------------------------------------- #
# STEP 3B: GRAPH CONTEXT + GRAPH-RAG ANSWER GENERATION
# --------------------------------------------------------------------------- #
#
# graph_retrieve() (Step 3A) returns matched entities and 1/2-hop facts. This
# step turns them into text Gemini can read, then asks Gemini to answer using
# ONLY that text. The context contains, for every fact: the triple, its hop,
# the evidence quote, source file and page; plus the ORIGINAL CHUNK TEXT the
# facts came from (looked up via kg.chunk_lookup), so Gemini never sees a bare
# "MLOps -> DataOps".

NO_GRAPH_MATCH_MESSAGE = "No relevant entities were found in the knowledge graph for this question."

GRAPH_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You answer questions using ONLY the GraphRAG context below. The context was retrieved
from a knowledge graph (entities and relationships extracted from documents) and includes,
for every graph fact, its evidence quote, source document, page number, and the original
source text.

Rules:
1. Use only the supplied context. Do NOT use outside knowledge.
2. Do NOT invent relationships and do NOT infer facts the context does not support.
3. Base your answer on the evidence quotes and the source chunk text.
4. If the context is insufficient to answer, say clearly that the information was not found
   in the graph/source context. Do not guess.
5. Mention the source document and page number when appropriate.

GraphRAG context:
{context}

Question: {question}

Answer:"""
)


@dataclass
class GraphRAGResult:
    """Everything the UI needs to show for one GraphRAG question.
    answer is None when no graph entity matched (Gemini is NOT called then)."""
    question: str
    matched: list          # list[MatchedEntity]
    facts: list            # list[RetrievedFact]
    source_chunks: list    # list[ChunkGraphData], the original chunks used as context
    context: str
    answer: Optional[str]


def chunks_signature(chunks: list[Document]) -> str:
    """Stable fingerprint of a list of chunks (order-independent). Built from
    make_chunk_id(), which already hashes source + page + text. If the user
    adds/removes a PDF or changes chunk size/overlap, the signature changes,
    so a stored graph can be recognised as stale."""
    ids = sorted(make_chunk_id(c) for c in chunks)
    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()


def _page_label(page: Optional[int]) -> str:
    return f"page {page}" if page is not None else "page unknown"


def collect_source_chunks(
    kg: KnowledgeGraph, matched: list, facts: list, max_chunks: int = 6
) -> list[ChunkGraphData]:
    """Original chunks behind the retrieved graph. Order: chunks that
    support retrieved facts first (closest hop first), then chunks where a
    matched entity was merely mentioned (so 'What is X?' still gets source
    text even if X has few edges). De-duplicated, capped at max_chunks."""
    ordered_ids: list[str] = []

    def add(chunk_id: str) -> None:
        if chunk_id in kg.chunk_lookup and chunk_id not in ordered_ids:
            ordered_ids.append(chunk_id)

    for f in facts:  # facts are already sorted by hop
        for p in f.provenance:
            add(p["chunk_id"])
    for m in matched:
        if m.node_key in kg.graph.nodes:
            for (_source, _page, chunk_id) in kg.graph.nodes[m.node_key].get("mentions", []):
                add(chunk_id)
    return [kg.chunk_lookup[cid] for cid in ordered_ids[:max_chunks]]


def format_graph_context(
    kg: KnowledgeGraph, matched: list, facts: list,
    max_chunks: int = 6, max_chunk_chars: int = 1500,
) -> tuple[str, list[ChunkGraphData]]:
    """Builds the LLM-readable GraphRAG context. Returns (context_text,
    source_chunks). context_text has three parts: matched entities, graph
    facts (triple + hop + evidence + source + page), and original chunk text."""
    source_chunks = collect_source_chunks(kg, matched, facts, max_chunks)

    lines: list[str] = ["MATCHED ENTITIES:"]
    for m in matched:
        lines.append(f"- {m.display_name} ({m.match_type} match)")

    lines += ["", "GRAPH FACTS:"]
    if not facts:
        lines.append("(none: the matched entities have no relationships in the graph)")
    for i, f in enumerate(facts, start=1):
        lines.append(f"[Fact {i} | hop {f.hop}] {f.subject} -[{f.predicate}]-> {f.object}")
        for p in f.provenance:
            lines.append(f'  Evidence: "{p["evidence"]}"')
            lines.append(f"  Source: {p['source']}, {_page_label(p['page'])}")

    lines += ["", "ORIGINAL SOURCE CHUNKS:"]
    if not source_chunks:
        lines.append("(none)")
    for i, c in enumerate(source_chunks, start=1):
        text = " ".join(c.chunk_text.split())
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + " ..."
        lines.append(f"[Source chunk {i} | {c.source}, {_page_label(c.page)}]")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", source_chunks


def generate_graph_answer(
    question: str, kg: KnowledgeGraph, model_name: str, api_key: str, hops: int = 2
) -> GraphRAGResult:
    """question -> graph_retrieve() (deterministic, no LLM) -> graph context
    -> Gemini -> answer. If nothing in the graph matches the question, Gemini
    is NOT called (an empty context would invite a hallucinated answer) and
    result.answer is None."""
    matched, facts = graph_retrieve(question, kg, hops=hops)
    if not matched:
        return GraphRAGResult(question, [], [], [], "", None)

    context, source_chunks = format_graph_context(kg, matched, facts)

    # Matched entities exist but there is nothing to ground an answer on.
    if not facts and not source_chunks:
        return GraphRAGResult(
            question, matched, facts, source_chunks, context,
            "The question matched entities in the knowledge graph, but no relationships or "
            "source text were found for them, so the information was not found in the graph/source context.",
        )

    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)
    chain = GRAPH_RAG_PROMPT | model | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return GraphRAGResult(question, matched, facts, source_chunks, context, answer)


# --------------------------------------------------------------------------- #
# STEP 4: COMMUNITY DETECTION  (Microsoft GraphRAG concept: hierarchical Leiden)
# --------------------------------------------------------------------------- #
#
# Microsoft GraphRAG clusters the entity graph with *hierarchical Leiden*
# (graspologic's `hierarchical_leiden`, default max_cluster_size=10): Leiden is
# run on the whole graph, and every community bigger than max_cluster_size is
# clustered again, recursively. We call the same algorithm through the small
# `graspologic-native` package (the Rust core that graspologic / GraphRAG use),
# instead of installing the whole graspologic stack.
#
# Differences from Microsoft's indexer (deliberate simplifications):
#   - we run on the undirected, weighted projection of our MultiDiGraph
#     (weight = number of supporting provenance records, parallel edges summed);
#   - we cluster ALL connected components (Microsoft's default `use_lcc=true`
#     keeps only the largest connected component, which would silently drop
#     entities from small documents);
#   - isolated entities (no relationship) are not part of any community.

DEFAULT_MAX_CLUSTER_SIZE = 10
LEIDEN_SEED = 0xDEADBEEF     # same default seed as Microsoft's implementation


class Community(BaseModel):
    community_id: str          # e.g. "L0-C3" = level 0, cluster 3
    level: int                 # 0 = coarsest; deeper levels split oversized communities
    parent: Optional[str]      # community_id of the parent community (None at level 0)
    node_keys: list[str]       # normalized entity keys, sorted


def _leiden_edge_list(kg: KnowledgeGraph) -> list[tuple[str, str, float]]:
    weights: dict[tuple[str, str], float] = {}
    for u, v, data in kg.graph.edges(data=True):
        if u == v:
            continue
        a, b = sorted((u, v))
        weights[(a, b)] = weights.get((a, b), 0.0) + float(max(len(data.get("provenance", [])), 1))
    return [(a, b, w) for (a, b), w in sorted(weights.items())]


def detect_communities(
    kg: KnowledgeGraph,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    seed: int = LEIDEN_SEED,
) -> list[Community]:
    """
    Detect hierarchical Leiden communities component-by-component.

    Why component-by-component?
    The knowledge graph can contain many disconnected components.
    Running hierarchical Leiden once over the entire sparse graph can be
    unnecessarily slow. We therefore:

      1. Build the same undirected weighted projection.
      2. Find connected components.
      3. Handle small components directly as one community.
      4. Run hierarchical Leiden only on components larger than
         max_cluster_size.
      5. Remap local Leiden cluster IDs to globally unique IDs.

    This still uses the Microsoft GraphRAG concept of hierarchical Leiden.
    """

    try:
        import graspologic_native as gn
    except ImportError as e:
        raise RuntimeError(
            "Community detection needs the Leiden implementation used by "
            "Microsoft GraphRAG: pip install graspologic-native"
        ) from e

    # ------------------------------------------------------------------
    # 1. Build the same weighted undirected edge list used previously
    # ------------------------------------------------------------------
    edges = _leiden_edge_list(kg)

    if not edges:
        return []

    # ------------------------------------------------------------------
    # 2. Convert edge list into an undirected NetworkX graph.
    #
    #    This is ONLY used to identify disconnected components.
    #    Leiden itself is still performed by graspologic-native.
    # ------------------------------------------------------------------
    component_graph = nx.Graph()

    for u, v, weight in edges:
        component_graph.add_edge(u, v, weight=weight)

    components = list(nx.connected_components(component_graph))

    # Process larger components first.
    components.sort(key=lambda nodes: (-len(nodes), sorted(nodes)[0]))

    communities: list[Community] = []

    # Global cluster counter so IDs from different components never clash.
    global_cluster_id = 0

    # ------------------------------------------------------------------
    # 3. Process each connected component independently
    # ------------------------------------------------------------------
    for component_index, component_nodes in enumerate(components):

        component_nodes = set(component_nodes)

        # --------------------------------------------------------------
        # Small component:
        # If it already fits inside max_cluster_size, there is no need
        # to run Leiden. The whole connected component is one community.
        # --------------------------------------------------------------
        if len(component_nodes) <= max_cluster_size:

            if len(component_nodes) >= 2:
                community_id = f"L0-C{global_cluster_id}"
                global_cluster_id += 1

                communities.append(
                    Community(
                        community_id=community_id,
                        level=0,
                        parent=None,
                        node_keys=sorted(component_nodes),
                    )
                )

            continue

        # --------------------------------------------------------------
        # Large component:
        # Run hierarchical Leiden ONLY on this connected component.
        # --------------------------------------------------------------
        component_edges = [
            (u, v, weight)
            for u, v, weight in edges
            if u in component_nodes and v in component_nodes
        ]

        if not component_edges:
            continue

        records = gn.hierarchical_leiden(
            edges=component_edges,
            max_cluster_size=max_cluster_size,
            seed=seed,
            starting_communities=None,
            resolution=1.0,
            randomness=0.001,
            use_modularity=True,
            iterations=1,
        )

        # --------------------------------------------------------------
        # 4. Group Leiden records by local cluster ID
        # --------------------------------------------------------------
        grouped: dict[int, dict] = {}

        for r in records:
            grouped.setdefault(
                r.cluster,
                {
                    "level": r.level,
                    "parent": r.parent_cluster,
                    "nodes": set(),
                },
            )

            grouped[r.cluster]["nodes"].add(r.node)

        # --------------------------------------------------------------
        # 5. Give each local Leiden cluster a globally unique ID.
        #
        #    parent_cluster refers to the LOCAL cluster ID returned by
        #    this Leiden call, so we first create a mapping.
        # --------------------------------------------------------------
        cluster_id_map: dict[int, str] = {}

        for local_cluster_id in sorted(grouped):
            level = grouped[local_cluster_id]["level"]

            cluster_id_map[local_cluster_id] = (
                f"L{level}-C{global_cluster_id}"
            )

            global_cluster_id += 1

        # --------------------------------------------------------------
        # 6. Create Community objects and preserve parent relationships
        # --------------------------------------------------------------
        for local_cluster_id, group in grouped.items():

            nodes = group["nodes"]

            # Ignore singleton communities, same as the original code.
            if len(nodes) < 2:
                continue

            level = group["level"]
            parent_local = group["parent"]

            parent = (
                cluster_id_map[parent_local]
                if parent_local is not None
                and parent_local in cluster_id_map
                else None
            )

            communities.append(
                Community(
                    community_id=cluster_id_map[local_cluster_id],
                    level=level,
                    parent=parent,
                    node_keys=sorted(nodes),
                )
            )

    # ------------------------------------------------------------------
    # 7. Stable deterministic ordering
    # ------------------------------------------------------------------
    return sorted(
        communities,
        key=lambda c: (
            c.level,
            -len(c.node_keys),
            c.community_id,
        ),
    )


def community_edges(kg: KnowledgeGraph, community: Community) -> list[tuple[str, str, str, dict]]:
    """Every relationship (subject_key, predicate, object_key, edge_data) whose
    two endpoints are both members of the community."""
    members = set(community.node_keys)
    return sorted(
        ((u, d["predicate"], v, d) for u, v, d in kg.graph.edges(data=True) if u in members and v in members),
        key=lambda e: (e[0], e[1], e[2]),
    )


def community_stats(communities: list[Community], kg: KnowledgeGraph) -> dict:
    """Numbers for the UI. 'coverage' = share of graph entities that belong to
    at least one community (isolated entities never do)."""
    covered = {n for c in communities for n in c.node_keys}
    total = kg.graph.number_of_nodes()
    return {
        "communities": len(communities),
        "levels": (max((c.level for c in communities), default=-1) + 1),
        "top_level_communities": sum(c.level == 0 for c in communities),
        "largest_community": max((len(c.node_keys) for c in communities), default=0),
        "entities_in_communities": len(covered),
        "coverage": (len(covered) / total) if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# STEP 5: COMMUNITY REPORTS  (Microsoft GraphRAG concept: community summaries)
# --------------------------------------------------------------------------- #
#
# For each community Gemini writes a report with the same fields Microsoft's
# reports have: title, summary, rating (+ explanation) and findings. The LLM
# only sees the community's own entities/relationships (with descriptions and
# one evidence quote each). The report's PROVENANCE (which PDFs / pages /
# chunks it rests on) is computed by our code from the graph, never by the LLM.
#
# Simplification vs Microsoft: every community is summarised directly from its
# own entities/relationships (capped below); Microsoft substitutes sub-community
# reports when a parent community's context is too large.

REPORT_MAX_ENTITIES = 40
DEFAULT_MAX_REPORTS = 30        # communities that get a report (see generate_community_reports)
REPORT_MAX_RELATIONSHIPS = 60


class ReportFinding(BaseModel):
    summary: str = Field(description="One-sentence headline of this finding.")
    explanation: str = Field(description="2-3 sentences explaining the finding, using only the supplied entities and relationships.")


class CommunityReportOut(BaseModel):
    """What Gemini returns for one community."""
    title: str = Field(description="Short title naming the community's main topic.")
    summary: str = Field(description="2-4 sentence executive summary of the community.")
    rating: float = Field(description="0-10: how important this community is for understanding the documents.")
    rating_explanation: str = Field(description="One sentence justifying the rating.")
    findings: list[ReportFinding] = Field(description="3-6 key findings, each grounded in the supplied data.")


class CommunityReport(BaseModel):
    """A validated report + code-computed provenance."""
    community_id: str
    level: int
    size: int
    title: str
    summary: str
    rating: float
    rating_explanation: str
    findings: list[dict]        # [{"summary":..., "explanation":...}]
    node_keys: list[str]
    node_names: list[str]
    relationships: list[str]    # "A -[predicate]-> B"
    chunk_ids: list[str]        # every chunk that supports an entity/relationship used
    locations: list[dict]       # [{"source","page_number","chunk_id"}], sorted
    signature: str              # hash of the exact context the report was written from


REPORT_PROMPT = """You write a report about ONE community (a cluster of closely related entities) found in a set of documents.

Use ONLY the entities and relationships listed below. Do not add outside knowledge, and do not
claim relationships that are not listed. If the data is thin, write a short, modest report.

ENTITIES (id: name (type): description)
{entities}

RELATIONSHIPS (id: subject -[predicate]-> object :: description :: evidence quote)
{relationships}

Return: a title, a summary, a 0-10 importance rating with a one-sentence explanation, and 3-6
findings (each a headline plus a short explanation)."""


def build_report_chain(model_name: str, api_key: str):
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    return ChatPromptTemplate.from_template(REPORT_PROMPT) | model.with_structured_output(CommunityReportOut)


def _report_context(kg: KnowledgeGraph, community: Community):
    """Deterministic, size-capped context for one community. Returns
    (entities_text, relationships_text, used_edges, used_node_keys)."""
    g = kg.graph
    edges = community_edges(kg, community)
    edges = sorted(edges, key=lambda e: (-len(e[3].get("provenance", [])), e[0], e[1], e[2]))[:REPORT_MAX_RELATIONSHIPS]
    nodes = sorted(community.node_keys, key=lambda n: (-g.degree(n), n))[:REPORT_MAX_ENTITIES]
    ent_lines = []
    for i, n in enumerate(nodes, start=1):
        d = g.nodes[n]
        desc = (d.get("descriptions") or [{"text": ""}])[0]["text"]
        ent_lines.append(f"N{i}: {d['name']} ({d.get('type', 'other')}): {desc}")
    rel_lines = []
    for j, (u, p, v, data) in enumerate(edges, start=1):
        prov = data.get("provenance", [])
        desc = next((x.get("description", "") for x in prov if x.get("description")), "")
        quote = (prov[0]["evidence"] if prov else "")[:200]
        rel_lines.append(f"R{j}: {g.nodes[u]['name']} -[{p}]-> {g.nodes[v]['name']} :: {desc} :: \"{quote}\"")
    return "\n".join(ent_lines), "\n".join(rel_lines), edges, nodes


def _report_provenance(kg: KnowledgeGraph, edges, nodes) -> tuple[list[str], list[dict]]:
    locs: dict[str, dict] = {}
    for (_u, _p, _v, data) in edges:
        for pr in data.get("provenance", []):
            locs[pr["chunk_id"]] = {"source": pr["source"], "page_number": pr["page"], "chunk_id": pr["chunk_id"]}
    for n in nodes:
        for (source, page, chunk_id) in kg.graph.nodes[n].get("mentions", []):
            locs.setdefault(chunk_id, {"source": source, "page_number": page, "chunk_id": chunk_id})
    ordered = sorted(locs.values(), key=lambda l: (l["source"], l["page_number"] if l["page_number"] is not None else -1, l["chunk_id"]))
    return [l["chunk_id"] for l in ordered], ordered


def generate_community_reports(
    kg: KnowledgeGraph,
    communities: list[Community],
    model_name: str,
    api_key: str,
    cache: Optional[dict] = None,
    on_progress: Optional[Callable[[int, int, "CommunityReport", bool], None]] = None,
    chain=None,
    retries: int = 2,
    levels: Optional[tuple] = (0,),
    max_reports: Optional[int] = DEFAULT_MAX_REPORTS,
    max_workers: int = 1,
    delay_between_calls: float = 0.0,
    on_checkpoint: Optional[Callable[[], None]] = None,
    checkpoint_every: int = 5,
) -> tuple[list[CommunityReport], list[tuple[str, str]]]:
    """One Gemini call per community that has at least one relationship — but only for
    the communities the searches can actually use:

      levels       which hierarchy levels get a report (default: level 0 only). Global
                   Search reads level 0 and Local Search adds level-0 reports, so reports of
                   deeper levels would cost a Gemini call each and never be read.
                   Pass None for every level.
      max_reports  only the N largest communities get a report (default 30). Global Search
                   uses at most 20 reports and Local Search at most 2 per question, while a
                   real graph can have hundreds of tiny communities (2-3 entities each).
                   Pass None for no limit.
      max_workers / delay_between_calls  same meaning as in extract_graph_data: reports are
                   written concurrently, and the pause is the minimum gap between the START
                   of two Gemini calls, shared by all workers.
      on_checkpoint  called every `checkpoint_every` new reports and once at the end (also
                   when interrupted) so the report cache can be saved.

    cache  dict signature -> CommunityReport. The signature hashes the exact
           context text, so an unchanged community is never re-summarised
           (adding a PDF only regenerates the communities it changed).
    Returns (reports, failures) — reports in largest-first order; failures =
    [(community_id, error)]. A failed community is skipped, it does not stop the others.
    on_progress(done, total, report, reused) is always called from the calling thread."""
    cache = {} if cache is None else cache
    todo = [c for c in communities if (levels is None or c.level in levels) and community_edges(kg, c)]
    todo.sort(key=lambda c: (-len(c.node_keys), c.community_id))          # largest communities first
    if max_reports:
        todo = todo[:max_reports]
    total = len(todo)

    slots: list = [None] * total
    failures: list[tuple[str, str]] = []
    state = {"done": 0, "fresh": 0}

    def finish(idx: int, rep: CommunityReport, reused: bool) -> None:
        slots[idx] = rep
        state["done"] += 1
        if not reused:
            state["fresh"] += 1
        if on_progress:
            on_progress(state["done"], total, rep, reused)
        if not reused and on_checkpoint and checkpoint_every and state["fresh"] % checkpoint_every == 0:
            on_checkpoint()

    queue: list = []
    for idx, com in enumerate(todo):
        ents, rels, edges, nodes = _report_context(kg, com)
        sig = hashlib.sha1(f"{REPORT_VERSION}\n{ents}\n{rels}".encode("utf-8")).hexdigest()
        if sig in cache:
            finish(idx, cache[sig].model_copy(update={"community_id": com.community_id, "level": com.level}), True)
        else:
            queue.append((idx, com, ents, rels, edges, nodes, sig))

    def build(item, out) -> CommunityReport:
        idx, com, ents, rels, edges, nodes, sig = item
        chunk_ids, locations = _report_provenance(kg, edges, nodes)
        rep = CommunityReport(
            community_id=com.community_id, level=com.level, size=len(com.node_keys),
            title=out.title.strip() or com.community_id, summary=out.summary.strip(),
            rating=max(0.0, min(10.0, float(out.rating))), rating_explanation=out.rating_explanation.strip(),
            findings=[{"summary": f.summary.strip(), "explanation": f.explanation.strip()} for f in out.findings],
            node_keys=list(com.node_keys), node_names=[kg.graph.nodes[n]["name"] for n in com.node_keys],
            relationships=[f"{kg.graph.nodes[u]['name']} -[{p}]-> {kg.graph.nodes[v]['name']}" for u, p, v, _ in edges],
            chunk_ids=chunk_ids, locations=locations, signature=sig,
        )
        cache[sig] = rep                                  # only the calling thread touches the cache
        return rep

    def settle(item, out, err) -> None:
        if out is None:
            failures.append((item[1].community_id, err or "unknown error"))
            state["done"] += 1
        else:
            finish(item[0], build(item, out), False)

    try:
        if queue:
            report_chain = chain if chain is not None else build_report_chain(model_name, api_key)
            limiter = _RateLimiter(delay_between_calls)

            def call_llm(item):
                out, err = None, None
                for attempt in range(retries + 1):
                    limiter.wait()
                    try:
                        out = report_chain.invoke({"entities": item[2], "relationships": item[3]})
                        if out is not None:
                            break
                        err = "model returned no structured output"
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                    if attempt < retries:
                        time.sleep(2 * (attempt + 1))
                return out, err

            workers = max(1, int(max_workers))
            if workers == 1:
                for item in queue:
                    settle(item, *call_llm(item))
            else:
                pool = ThreadPoolExecutor(max_workers=workers)
                try:
                    futures = {pool.submit(call_llm, item): item for item in queue}
                    for fut in as_completed(futures):
                        settle(futures[fut], *fut.result())
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
    finally:
        if on_checkpoint and state["fresh"]:
            on_checkpoint()
    return [r for r in slots if r is not None], sorted(failures)


# --------------------------------------------------------------------------- #
# GRAPH VISUALIZATION — self-contained SVG (no extra dependencies, no Streamlit)
# --------------------------------------------------------------------------- #

_TYPE_COLORS = {
    "component": "#5B5FEF", "system": "#8B5CF6", "person": "#EC4899",
    "organization": "#F59E0B", "document": "#64748B", "process": "#10B981",
    "concept": "#3B82F6", "parameter": "#EF4444", "other": "#94A3B8",
}


def render_graph_svg(
    kg: KnowledgeGraph,
    matched: Optional[list] = None,
    facts: Optional[list] = None,
    include_isolated: bool = False,
    width: int = 960,
    height: int = 560,
    max_nodes: int = 120,
) -> tuple[str, int]:
    """Returns (html, height_px): an HTML string (legend + inline SVG) drawing the graph.

    - Default (facts=None): the whole knowledge graph. Isolated nodes are
      hidden unless include_isolated=True (they add clutter, no structure).
    - With facts (from graph_retrieve): only the retrieved subgraph.
      Matched entities get a red outline; hop-1 edges are indigo, hop-2
      edges orange.
    Node colour = entity type, node size = number of connections, arrows
    follow the fact's direction, hovering a node shows its details.
    Returns ("", 0) if there is nothing to draw."""
    import html
    import math

    g = kg.graph
    matched_keys = {m.node_key for m in (matched or [])}

    if facts:
        edges = [(f.subject_key, f.predicate, f.object_key, f.hop) for f in facts]
        keep = set(matched_keys)
        for s, _p, o, _h in edges:
            keep |= {s, o}
    else:
        edges = [(u, d["predicate"], v, 0) for u, v, d in g.edges(data=True)]
        keep = set(g.nodes) if include_isolated else {n for e in edges for n in (e[0], e[2])}

    keep = {n for n in keep if n in g.nodes}
    if len(keep) > max_nodes:  # keep the best-connected nodes if the graph is huge
        keep = set(sorted(keep, key=lambda n: (-g.degree(n), n))[:max_nodes])
    edges = [e for e in edges if e[0] in keep and e[2] in keep and e[0] != e[2]]
    if not keep:
        return "", 0

    # ---- layout (deterministic: fixed seed) ------------------------------
    # Each connected component is laid out on its own and the components are
    # packed into a grid, so separate groups stay close together. Isolated
    # nodes (no edges) go in a compact band at the bottom.
    H = nx.Graph()
    H.add_nodes_from(sorted(keep))
    H.add_edges_from((s, o) for s, _p, o, _h in edges)
    comps = sorted((sorted(c) for c in nx.connected_components(H)), key=lambda c: (-len(c), c[0]))
    multi = [c for c in comps if len(c) > 1]
    singles = [c[0] for c in comps if len(c) == 1]

    pad_x, pad_top = 70, 30
    per_row = max(1, (width - 2 * pad_x) // 120)
    band_h = 0 if not singles else (math.ceil(len(singles) / per_row) * 46 + 24)

    # Big groups get a full-width row of their own, as tall as they need;
    # small groups share a grid below them.
    big = [c for c in multi if len(c) > 8]
    small = [c for c in multi if len(c) <= 8]
    big_h = [max(440, 40 * len(c)) for c in big]
    small_cols = max(1, min(4, len(small)))
    small_rows = math.ceil(len(small) / small_cols) if small else 0
    small_cell_h = 220
    top_h = (pad_top + sum(big_h) + small_rows * small_cell_h) if multi else 0
    total_h = top_h + band_h if (multi or singles) else height
    P: dict = {}

    def place(comp, ox, oy, cell_w, cell_h):
        sub = H.subgraph(comp)
        m = len(comp)
        pos = nx.spring_layout(sub, k=2.0 / math.sqrt(m), iterations=300, seed=42)
        xs_, ys_ = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
        x0, x1, y0, y1 = min(xs_), max(xs_), min(ys_), max(ys_)
        inner_x, inner_y = 55, 45
        for node, (px0, py0) in pos.items():
            fx = 0.5 if x1 == x0 else (px0 - x0) / (x1 - x0)
            fy = 0.5 if y1 == y0 else (py0 - y0) / (y1 - y0)
            P[node] = (ox + inner_x + fx * max(cell_w - 2 * inner_x, 1),
                       oy + inner_y + fy * max(cell_h - 2 * inner_y, 1))

    y_cursor = pad_top
    for comp, ch in zip(big, big_h):
        place(comp, pad_x, y_cursor, width - 2 * pad_x, ch)
        y_cursor += ch
    if small:
        cell_w = (width - 2 * pad_x) / small_cols
        for idx, comp in enumerate(small):
            col, row = idx % small_cols, idx // small_cols
            place(comp, pad_x + col * cell_w, y_cursor + row * small_cell_h, cell_w, small_cell_h)
    for i, node in enumerate(singles):
        r_, c_ = divmod(i, per_row)
        P[node] = (pad_x + 50 + c_ * 120, top_h + 26 + r_ * 46)
    height = total_h
    R = {k: 7 + 2 * min(g.degree(k), 6) for k in keep}

    # ---- edges -----------------------------------------------------------
    edge_color = {0: "#9CA3AF", 1: "#5B5FEF", 2: "#F59E0B"}
    groups: dict = {}
    for i, (s, _p, o, _h) in enumerate(edges):
        groups.setdefault(tuple(sorted((s, o))), []).append(i)

    parts: list[str] = []
    defs = "".join(
        f'<marker id="arr{h}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for h, c in edge_color.items()
    )
    labels: list[str] = []
    for (a, b), idxs in groups.items():
        ax_, ay_ = P[a]
        bx_, by_ = P[b]
        dx, dy = bx_ - ax_, by_ - ay_
        length = math.hypot(dx, dy) or 1.0
        px_, py_ = -dy / length, dx / length          # unit perpendicular of a->b
        for j, i in enumerate(idxs):
            s, pred, o, hop = edges[i]
            off = (j - (len(idxs) - 1) / 2) * 44
            (x1, y1), (x2, y2) = P[s], P[o]
            cx, cy = (x1 + x2) / 2 + px_ * off * 2, (y1 + y2) / 2 + py_ * off * 2
            d1 = math.hypot(cx - x1, cy - y1) or 1.0
            d2 = math.hypot(x2 - cx, y2 - cy) or 1.0
            sx1, sy1 = x1 + (cx - x1) / d1 * R[s], y1 + (cy - y1) / d1 * R[s]
            ex, ey = x2 - (x2 - cx) / d2 * (R[o] + 3), y2 - (y2 - cy) / d2 * (R[o] + 3)
            col = edge_color.get(hop, edge_color[0])
            parts.append(
                f'<path d="M{sx1:.1f},{sy1:.1f} Q{cx:.1f},{cy:.1f} {ex:.1f},{ey:.1f}" fill="none" '
                f'stroke="{col}" stroke-width="{2 if hop else 1.5}" marker-end="url(#arr{hop if hop in edge_color else 0})"/>'
            )
            lx, ly = 0.25 * sx1 + 0.5 * cx + 0.25 * ex, 0.25 * sy1 + 0.5 * cy + 0.25 * ey
            ptxt = html.escape(pred.replace("_", " "))
            labels.append(
                f'<text x="{lx:.1f}" y="{ly - 4:.1f}" text-anchor="middle" font-size="10" fill="#fff" '
                f'stroke="#fff" stroke-width="4" stroke-linejoin="round">{ptxt}</text>'
                f'<text x="{lx:.1f}" y="{ly - 4:.1f}" text-anchor="middle" font-size="10" fill="{col}">{ptxt}</text>'
            )
    parts.extend(labels)

        # ---- nodes -----------------------------------------------------------
    types_present: set = set()
    for k in sorted(keep):
        data = g.nodes[k]
        etype = data.get("type", "other")
        types_present.add(etype)
        x, y = P[k]
        is_match = k in matched_keys
        stroke = ('stroke="#DC2626" stroke-width="3.5"' if is_match else 'stroke="#fff" stroke-width="2"')
        tip = f'{data["name"]} ({etype}) — {g.degree(k)} connection(s), {len(data.get("mentions", []))} mention(s)'
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{R[k]}" fill="{_TYPE_COLORS.get(etype, _TYPE_COLORS["other"])}" '
            f'{stroke}><title>{html.escape(tip)}</title></circle>'
        )
        label = data["name"] if len(data["name"]) <= 22 else data["name"][:21] + "…"
        ltxt = html.escape(label)
        ly2 = y + R[k] + 13
        parts.append(
            f'<text x="{x:.1f}" y="{ly2:.1f}" text-anchor="middle" font-size="11" '
            f'font-weight="{700 if is_match else 500}" fill="#fff" stroke="#fff" stroke-width="4" '
            f'stroke-linejoin="round">{ltxt}</text>'
            f'<text x="{x:.1f}" y="{ly2:.1f}" text-anchor="middle" font-size="11" '
            f'font-weight="{700 if is_match else 500}" fill="#1A1D29">{ltxt}</text>'
        )

    # ---- legend + wrapper ------------------------------------------------
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
        f'<span style="width:11px;height:11px;border-radius:50%;background:{_TYPE_COLORS.get(t, "#94A3B8")};'
        f'display:inline-block;margin-right:5px;"></span>{html.escape(t)}</span>'
        for t in sorted(types_present)
    )
    if facts:
        legend += ('<span style="margin-left:6px;color:#DC2626;">&#9711; matched entity</span>'
                   '<span style="margin-left:12px;color:#5B5FEF;">&#8212; hop 1</span>'
                   '<span style="margin-left:12px;color:#F59E0B;">&#8212; hop 2</span>')

    fs_style = (
        '<style>'
        '#kgwrap{background:#fff;}'
        '#kgwrap:fullscreen{padding:14px;overflow:auto;}'
        '#kgwrap:-webkit-full-screen{padding:14px;overflow:auto;}'
        '#kgwrap:fullscreen svg{height:calc(100vh - 70px);width:100%;}'
        '#kgwrap:-webkit-full-screen svg{height:calc(100vh - 70px);width:100%;}'
        '#kgfs{float:right;border:1px solid #D9DAFB;background:#EEF0FF;color:#5B5FEF;'
        'border-radius:8px;padding:4px 12px;font-size:12px;font-weight:600;cursor:pointer;}'
        '</style>'
    )
    fs_button = (
        '<button id="kgfs" type="button" onclick="var w=document.getElementById(\'kgwrap\');'
        'if(document.fullscreenElement){document.exitFullscreen();}'
        'else if(w.requestFullscreen){w.requestFullscreen();}'
        'else if(w.webkitRequestFullscreen){w.webkitRequestFullscreen();}">'
        '&#x26F6; Full screen</button>'
    )
    html_out = (
        f'{fs_style}'
        '<div id="kgwrap" style="font-family:Inter,Arial,sans-serif;font-size:12px;color:#374151;">'
        f'{fs_button}<div style="margin-bottom:6px;">{legend}</div>'
        f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" '
        'style="background:#fff;border:1px solid #E7E8EE;border-radius:12px;">'
        f'<defs>{defs}</defs>{"".join(parts)}</svg></div>'
    )
    return html_out, int(height * 0.95) + 50


# --------------------------------------------------------------------------- #
# Stand-alone test runner:  python graph_rag.py [--pdf file.pdf ...]
# --------------------------------------------------------------------------- #

DEMO_CHUNKS = [
    ("Motor A is controlled by Controller B.", 3),
    ("Controller B receives power from PSU-1.", 9),
]


def _print_result(res: ChunkGraphData) -> None:
    page = f"p{res.page}" if res.page is not None else "p?"
    text = " ".join(res.chunk_text.split())
    print("=" * 78)
    print(f"CHUNK   {res.chunk_id}    status={res.status}")
    print(f"TEXT    {text[:300]}{'...' if len(text) > 300 else ''}")
    if res.error:
        print(f"ERROR   {res.error}")
    if res.entities:
        print("ENTITIES")
        for e in res.entities:
            print(f"   - {e.name}  ({e.type})")
    if res.triples:
        print("TRIPLES  (subject -[predicate]-> object)")
        for t in res.triples:
            print(f"   - {t.subject} -[{t.predicate}]-> {t.object}")
            print(f"       evidence:   \"{t.evidence}\"")
            print(f"       provenance: {t.source}, {page}, chunk {t.chunk_id}")
    if res.dropped:
        print("DROPPED  (proposed by the model but failed a support check)")
        for d in res.dropped:
            print(f"   - {d}")


def _print_graph_stats(kg: KnowledgeGraph) -> None:
    stats = graph_stats(kg)
    print("=" * 78)
    print("KNOWLEDGE GRAPH")
    print(f"   nodes: {stats['nodes']}   edges: {stats['edges']}   "
          f"isolated nodes: {stats['isolated_nodes']}   "
          f"connected components: {stats['connected_components']} "
          f"(largest has {stats['largest_component_size']} node(s))")
    if stats["top_entities_by_degree"]:
        print("   top entities by degree:", ", ".join(f"{n} ({d})" for n, d in stats["top_entities_by_degree"]))
    for u, v, data in sorted(kg.graph.edges(data=True), key=lambda e: (e[0], e[1], e[2]["predicate"])):
        print(f"   {kg.graph.nodes[u]['name']} -[{data['predicate']}]-> {kg.graph.nodes[v]['name']}"
              f"   ({len(data['provenance'])} source(s))")


def _print_retrieval(question: str, matched: list[MatchedEntity], facts: list[RetrievedFact]) -> None:
    print("=" * 78)
    print(f"QUESTION   {question}")
    if not matched:
        print("   no graph entities matched this question")
        return
    print("MATCHED ENTITIES")
    for m in matched:
        print(f"   - {m.display_name}  (key={m.node_key}, match={m.match_type}, on \"{m.matched_text}\")")
    if not facts:
        print("   matched entities have no edges in the graph")
        return
    print("RETRIEVED FACTS")
    for f in facts:
        print(f"   [hop {f.hop}] {f.subject} -[{f.predicate}]-> {f.object}")
        for p in f.provenance:
            page = f"p{p['page']}" if p["page"] is not None else "p?"
            print(f"       evidence:   \"{p['evidence']}\"")
            print(f"       provenance: {p['source']}, {page}, chunk {p['chunk_id']}")


def _demo() -> None:
    import argparse
    import os
    import sys

    from dotenv import load_dotenv

    # Windows consoles can choke on characters found in PDFs.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Test Steps 1-3A: extraction, graph construction, graph retrieval.")
    parser.add_argument("--pdf", help="PDF to chunk (800/100, same as the app) and extract from. Default: built-in Motor A example (2 chunks, on different pages, to demonstrate cross-chunk merging).")
    parser.add_argument("--start", type=int, default=0, help="Index of the first chunk to process (PDFs only).")
    parser.add_argument("--max-chunks", type=int, default=5, help="How many chunks to process (each is one Gemini call).")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--build-graph", action="store_true", help="Also build the NetworkX knowledge graph (Step 2) from the extraction results and print its stats.")
    parser.add_argument("--ask", action="append", default=[], help="A question to run graph retrieval (Step 3A) against. Implies --build-graph. Repeatable.")
    args = parser.parse_args()
    if args.ask:
        args.build_graph = True

    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not found. Put it in your .env file (same as the app).")

    if args.pdf:
        from pypdf import PdfReader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        name = os.path.basename(args.pdf)
        pages = [
            Document(page_content=p.extract_text() or "", metadata={"source": name, "page": i})
            for i, p in enumerate(PdfReader(args.pdf).pages)
        ]
        pages = [p for p in pages if p.page_content.strip()]
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(pages)
        print(f"{name}: {len(chunks)} chunks total; processing chunks {args.start}..{args.start + args.max_chunks - 1}")
        chunks = chunks[args.start:]
    else:
        chunks = [Document(page_content=text, metadata={"source": "demo.txt", "page": page}) for text, page in DEMO_CHUNKS]

    def progress(done: int, total: int, res: ChunkGraphData) -> None:
        print(f"[{done}/{total}] {res.chunk_id} -> {res.status}")

    results = extract_graph_data(chunks, args.model, api_key, max_chunks=args.max_chunks, on_progress=progress)
    for res in results:
        _print_result(res)
    print("=" * 78)
    print("SUMMARY", summarize_extraction(results))

    if args.build_graph:
        kg = build_knowledge_graph(results)
        _print_graph_stats(kg)
        for question in args.ask:
            matched, facts = graph_retrieve(question, kg)
            _print_retrieval(question, matched, facts)


if __name__ == "__main__":
    _demo()