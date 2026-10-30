"""

evidence.py — the shared evidence model for every retrieval mode.



No Streamlit, no Gemini, no ChromaDB in here: pure Python, so it is easy to test.



Every retrieval mode (Traditional RAG, Local Search, Global Search, Hybrid)

turns what it found into a list of `RetrievedEvidence`. Everything after that

— de-duplication, ranking, the [E1]/[E2] labels sent to Gemini, checking

Gemini's citations, and the UI — works on this one type.



The rule that keeps citations honest:



    source / page / chunk_id are copied from metadata by OUR code.

    Gemini only ever writes "[E3]"; the app looks E3 up in this list.



An evidence item is one of two kinds:

    chunk items      a text chunk of a PDF (source, page, chunk_id are set)

    community items  a community report (retrieval_type == "community"); it has

                     no single page — `provenance` lists every PDF/page/chunk

                     the report was built from.

"""



import re

from dataclasses import dataclass, field

from typing import Optional



INSUFFICIENT_EVIDENCE = "The available documents do not provide sufficient evidence to answer this question."



# Near-duplicate threshold on word 3-shingles (Jaccard). Deliberately high: two

# chunks that merely say similar things in different words must stay separate.

NEAR_DUPLICATE_THRESHOLD = 0.9





@dataclass

class RetrievedEvidence:

    text: str

    retrieval_type: str                      # "vector" | "graph" | "vector+graph" | "community"

    source: Optional[str] = None             # PDF name (chunk items)

    page: Optional[int] = None               # 0-based page index (as stored in Chroma)

    page_number: Optional[int] = None        # 1-based page for humans (what the UI shows)

    chunk_id: Optional[str] = None

    evidence_id: str = ""                    # "E1", "E2"... assigned just before prompting

    distance: Optional[float] = None         # Chroma distance (vector items only; lower = closer)

    graph_hop: Optional[int] = None          # 0 = chunk mentions a matched entity, 1/2 = hops away

    graph_links: int = 0                     # how many matched entities/facts point at this chunk

    entity_ids: list = field(default_factory=list)          # normalized entity keys

    relationship_ids: list = field(default_factory=list)    # "subject|predicate|object"

    graph_facts: list = field(default_factory=list)         # readable facts supported by this chunk

    community_id: Optional[str] = None

    title: Optional[str] = None              # community report title

    provenance: list = field(default_factory=list)          # community items: [{source,page_number,chunk_id}]

    other_locations: list = field(default_factory=list)     # near-identical copies: [{source,page_number,chunk_id}]

    citation_status: str = "not_cited"       # "cited" | "not_cited" (set after Gemini answers)



    @property

    def is_community(self) -> bool:

        return self.retrieval_type == "community"



    def location(self) -> str:

        if self.is_community:

            return f"Community report {self.community_id}"

        page = f"Page {self.page_number}" if self.page_number is not None else "Page unknown"

        return f"{self.source} — {page}"





def evidence_from_document(doc, distance: Optional[float] = None, retrieval_type: str = "vector") -> RetrievedEvidence:

    """Builds a chunk item from a LangChain Document (as stored in ChromaDB)."""

    md = doc.metadata

    page = md.get("page")

    page_number = md.get("page_number", page + 1 if isinstance(page, int) else None)

    return RetrievedEvidence(

        text=doc.page_content, retrieval_type=retrieval_type, source=md.get("source"),

        page=page, page_number=page_number, chunk_id=md.get("chunk_id"), distance=distance,

    )





# --------------------------------------------------------------------------- #

# Near-duplicate handling

# --------------------------------------------------------------------------- #



def _shingles(text: str, n: int = 3) -> set:

    words = re.sub(r"[\W_]+", " ", text.lower()).split()

    if len(words) < n:

        return {" ".join(words)} if words else set()

    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}





def near_duplicate(a: str, b: str, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> bool:

    """True only when the two texts are almost word-for-word the same."""

    sa, sb = _shingles(a), _shingles(b)

    if not sa or not sb:

        return False

    return len(sa & sb) / len(sa | sb) >= threshold





def _loc(e: RetrievedEvidence) -> dict:

    return {"source": e.source, "page_number": e.page_number, "chunk_id": e.chunk_id}





def collapse_duplicates(items: list) -> list:

    """Sends near-identical chunk texts to Gemini once, WITHOUT losing where

    they came from: the first item is kept and every later copy is recorded in

    its `other_locations`. Community items are never collapsed. Chunks that

    only say similar things in different words are never collapsed."""

    kept: list = []

    for item in items:

        if item.is_community:

            kept.append(item)

            continue

        twin = next((k for k in kept if not k.is_community and near_duplicate(k.text, item.text)), None)

        if twin is None:

            kept.append(item)

            continue

        if _loc(item) != _loc(twin) and _loc(item) not in twin.other_locations:

            twin.other_locations.append(_loc(item))

        for loc in item.other_locations:

            if loc not in twin.other_locations:

                twin.other_locations.append(loc)

    return kept





# --------------------------------------------------------------------------- #

# Merge + rank (transparent, no artificial combined score)

# --------------------------------------------------------------------------- #



def merge_same_chunk(vector_items: list, graph_items: list) -> tuple[list, list, list]:

    """A chunk found by BOTH channels becomes one item ("vector+graph") that

    keeps its distance AND its graph information. Returns (both, vector_only,

    graph_only)."""

    graph_by_id = {g.chunk_id: g for g in graph_items if g.chunk_id}

    both, vector_only, used = [], [], set()

    for v in vector_items:

        g = graph_by_id.get(v.chunk_id)

        if g is None:

            vector_only.append(v)

            continue

        v.retrieval_type = "vector+graph"

        v.graph_hop, v.graph_links = g.graph_hop, g.graph_links

        v.entity_ids, v.relationship_ids, v.graph_facts = g.entity_ids, g.relationship_ids, g.graph_facts

        both.append(v)

        used.add(v.chunk_id)

    graph_only = [g for g in graph_items if g.chunk_id not in used]

    return both, vector_only, graph_only





def rank_and_limit(vector_items: list, graph_items: list, community_items: Optional[list] = None,

                   max_items: int = 10) -> tuple[list, int]:

    """Builds the final evidence list for Gemini. Returns (items, total_candidates).



    Vector distances and graph hops are different kinds of number, so they are

    never compared. The order is simply:

      1. chunks found by both channels (closest vector distance first)

      2. vector-only and graph-only chunks, interleaved (vector: closest first;

         graph: most linked / fewest hops first)

      3. community reports last (they are summaries, not source text)

    Near-identical chunks are collapsed (all locations kept) BEFORE the limit

    is applied, so duplicates do not waste slots.

    """

    community_items = community_items or []

    total = len(vector_items) + len(graph_items) + len(community_items)



    both, v_only, g_only = merge_same_chunk(list(vector_items), list(graph_items))

    both.sort(key=lambda e: (e.distance if e.distance is not None else 9e9))

    v_only.sort(key=lambda e: (e.distance if e.distance is not None else 9e9))

    g_only.sort(key=lambda e: (-e.graph_links, e.graph_hop if e.graph_hop is not None else 9, e.chunk_id or ""))



    ordered = list(both)

    for i in range(max(len(v_only), len(g_only))):

        if i < len(v_only):

            ordered.append(v_only[i])

        if i < len(g_only):

            ordered.append(g_only[i])



    chunks = collapse_duplicates(ordered)

    n_comm = min(len(community_items), max(1, max_items // 5)) if community_items else 0

    final = chunks[:max_items - n_comm] + community_items[:n_comm]

    return final, total





def assign_ids(items: list) -> list:

    for i, item in enumerate(items, start=1):

        item.evidence_id = f"E{i}"

    return items





# --------------------------------------------------------------------------- #

# The text Gemini sees

# --------------------------------------------------------------------------- #



def _loc_text(loc: dict) -> str:

    page = f"page {loc['page_number']}" if loc.get("page_number") is not None else "page unknown"

    return f"{loc.get('source')}, {page}"





def format_context(items: list, max_chunk_chars: int = 1500) -> str:

    """One labelled block per evidence item. Only metadata that really exists

    is printed (no invented distances, pages or sources)."""

    blocks = []

    for e in items:

        lines = [f"[{e.evidence_id}]"]

        if e.is_community:

            lines.append(f"Type: community report {e.community_id} (a summary of a group of related entities)")

            lines.append(f"Title: {e.title}")

            if e.provenance:

                shown = "; ".join(_loc_text(l) for l in e.provenance[:6])

                more = f" (+{len(e.provenance) - 6} more)" if len(e.provenance) > 6 else ""

                lines.append(f"Built from: {shown}{more}")

            lines.append("Report:")

            lines.append(e.text)

        else:

            lines.append(f"Source: {e.source}")

            lines.append(f"Page: {e.page_number if e.page_number is not None else 'unknown'}")

            lines.append(f"Chunk ID: {e.chunk_id}")

            if e.other_locations:

                lines.append("Near-identical text also appears in: " + "; ".join(_loc_text(l) for l in e.other_locations))

            if e.graph_facts:

                lines.append("Knowledge-graph facts extracted from this text:")

                lines += [f"  - {f}" for f in e.graph_facts]

            text = " ".join(e.text.split())

            lines.append("Text:")

            lines.append(text[:max_chunk_chars] + (" ..." if len(text) > max_chunk_chars else ""))

        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)





# --------------------------------------------------------------------------- #

# Citations: never trust the model for metadata

# --------------------------------------------------------------------------- #



_CITE_GROUP = re.compile(
    r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]",
    re.IGNORECASE
)





def map_citations(answer: str, items: list) -> tuple[str, list, list]:

    """Checks every [E#] the model wrote against the evidence we really sent.



    Returns (clean_answer, cited_ids, unknown_ids):

      - unknown ids (e.g. [E99]) are REMOVED from the displayed text and listed;

      - the valid ones mark the matching items as citation_status == "cited".

    """

    valid = {e.evidence_id for e in items}

    for e in items:

        e.citation_status = "not_cited"

    cited: list = []

    unknown: list = []



    def _fix(match: "re.Match") -> str:

        ids = [x.strip().upper() for x in re.split(r"[,;]", match.group(1))]

        good = [i for i in ids if i in valid]

        unknown.extend(i for i in ids if i not in valid and i not in unknown)

        for i in good:

            if i not in cited:

                cited.append(i)

        return "".join(f"[{i}]" for i in good)



    clean = _CITE_GROUP.sub(_fix, answer)

    clean = re.sub(r"[ **\t**]+([.,;:])", r"\1", clean)

    for e in items:

        if e.evidence_id in cited:

            e.citation_status = "cited"

    return clean, cited, unknown





@dataclass

class AnswerResult:

    """Everything the UI needs for one answered question (any mode)."""

    question: str

    mode: str

    answer: str

    evidence: list = field(default_factory=list)      # list[RetrievedEvidence], ids assigned

    cited_ids: list = field(default_factory=list)

    unknown_ids: list = field(default_factory=list)   # citations the model invented (removed from text)

    total_retrieved: int = 0                          # candidates before the limit

    insufficient: bool = False

    notes: list = field(default_factory=list)         # e.g. "Showing 10 of 23 retrieved evidence items."

    context: str = ""                                 # exact text sent to Gemini (also used by the judge)

    matched: list = field(default_factory=list)       # graph modes: MatchedEntity list

    facts: list = field(default_factory=list)         # graph modes: RetrievedFact list (for the subgraph view)

    findings: list = field(default_factory=list)      # global mode: intermediate map findings
