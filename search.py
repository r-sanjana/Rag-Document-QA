"""
search.py — the retrieval modes and grounded answer generation (no Streamlit).

    Traditional RAG   ChromaDB similarity search ─► evidence ─► Gemini
    Local Search      entities in the question ─► graph neighbourhood ─► linked text
                      units (+ community reports) ─► Gemini
                      [Microsoft GraphRAG "Local Search" concept, simplified]
    Global Search     community reports ─► MAP (scored points per batch) ─► REDUCE ─► Gemini
                      [Microsoft GraphRAG "Global Search" concept, simplified]
    Hybrid RAG        vector evidence + local-search graph evidence (+ reports) ─► merge ─► Gemini

All modes end in `answer_from_evidence()`: evidence gets [E1], [E2]... labels,
Gemini must cite those labels, and `evidence.map_citations()` checks them
against the real metadata afterwards.

DRIFT Search (Microsoft) is intentionally NOT implemented — see README.

Every function takes an optional `model` so tests can inject a fake LLM; in the
app it is a ChatGoogleGenerativeAI.
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

import vector_store as vs
import llm_cache
from evidence import (
    INSUFFICIENT_EVIDENCE, AnswerResult, RetrievedEvidence, assign_ids, format_context,
    map_citations, rank_and_limit,
)
from graph_rag import (
    NO_GRAPH_MATCH_MESSAGE, CommunityReport, KnowledgeGraph, find_matching_nodes, get_neighborhood_facts,
)

# Context limits (every limit that truncates is reported to the user).
MAX_EVIDENCE = 10                  # evidence items sent to Gemini (Hybrid / Local)
MAX_GRAPH_FACTS = 25               # graph facts kept after ranking
MAX_PROVENANCE_PER_FACT = 3        # supporting chunks followed per fact
MAX_LOCAL_CHUNKS = 8               # graph-derived text chunks
MAX_LOCAL_REPORTS = 2              # community reports added to Local/Hybrid context
GLOBAL_BATCH_SIZE = 5              # community reports per MAP call
GLOBAL_MAX_REPORTS = 20
GLOBAL_MAX_POINTS = 15             # intermediate points passed to REDUCE


# --------------------------------------------------------------------------- #
# LLM plumbing
# --------------------------------------------------------------------------- #

def chat_model(model_name: str, api_key: str, temperature: float = 0.2):
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)


def _text_of(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):      # newer Gemini models return a list of content blocks
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _model_cache_name(model) -> str:
    """Stable model identifier; never include API keys in cache keys."""
    return str(getattr(model, "model", None) or getattr(model, "model_name", None) or type(model).__name__)


def _call_text(model, prompt: str) -> str:
    payload = (_model_cache_name(model), prompt)
    cached = llm_cache.get("gemini_text_v1", payload)
    if cached is not None:
        llm_cache.set_last_cache_status("HIT")
        return cached
    result = _text_of(model.invoke(prompt))
    llm_cache.set("gemini_text_v1", payload, result)
    return result


def _call_structured(model, schema, prompt: str):
    payload = (_model_cache_name(model), f"{schema.__module__}.{schema.__qualname__}", prompt)
    cached = llm_cache.get("gemini_structured_v1", payload)
    if cached is not None:
        llm_cache.set_last_cache_status("HIT")
        return cached
    result = model.with_structured_output(schema).invoke(prompt)
    llm_cache.set("gemini_structured_v1", payload, result)
    return result


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

ANSWER_PROMPT = """You answer questions using ONLY the numbered evidence below. The evidence was retrieved
automatically from the user's PDF documents and may be irrelevant to the question.

Rules:
1. Use only facts stated in the evidence. Never use outside knowledge, and never guess.
2. Cite every claim with the ID(s) of the evidence that supports it, written exactly like [E1] or
   [E2][E4]. Use only IDs that appear in the evidence. Never invent an ID.
3. Do NOT write PDF names, page numbers or chunk IDs yourself — the application displays them from
   the evidence metadata. Refer to sources only through [E#] IDs.
4. If the evidence does not contain enough information to answer, reply with exactly this sentence
   and nothing else: {insufficient}
5. If several evidence items say similar things, combine them carefully and cite all of them. If
   they disagree, say so and cite both.
6. "Knowledge-graph facts" listed under an item are relationships extracted from that item's text;
   the text itself is the authority.

Evidence:
{context}

Question: {question}

Answer:"""

MAP_PROMPT = """You are given community reports. A community report summarises a group of closely related
entities found in the user's documents. Each report has an evidence ID such as [E3].

Using ONLY these reports, list the points that help answer the question. For every point give:
a short description, an importance score from 0 to 100 (0 = does not help), and the evidence
ID(s) it is based on (only IDs listed below). If no report helps, return an empty list.

Reports:
{context}

Question: {question}"""

REDUCE_PROMPT = """You answer a question about a whole set of documents using ONLY the key points below. Each
point comes from community reports and is labelled with evidence ID(s) such as [E3].

Rules:
1. Use only the points. Never use outside knowledge, and never guess.
2. Cite every claim with the ID(s) from the points, exactly like [E1] or [E2][E4]. Never invent IDs.
3. Do NOT write PDF names, page numbers or chunk IDs — the application shows them from metadata.
4. If the points do not answer the question, reply with exactly this sentence and nothing else:
   {insufficient}
5. Merge overlapping points, keep the most important themes first.

Key points (highest importance first):
{points}

Question: {question}

Answer:"""


# --------------------------------------------------------------------------- #
# Shared answer step
# --------------------------------------------------------------------------- #

def _truncation_note(shown: int, total: int, what: str = "retrieved evidence items") -> list:
    return [f"Showing {shown} of {total} {what}."] if total > shown else []


def answer_from_evidence(question: str, mode: str, evidence: list, model, total: int,
                         notes: Optional[list] = None) -> AnswerResult:
    """evidence -> [E#] labels -> Gemini -> citation check. With no evidence at
    all Gemini is not called."""
    notes = list(notes or []) + _truncation_note(len(evidence), total)
    assign_ids(evidence)
    if not evidence:
        return AnswerResult(question, mode, INSUFFICIENT_EVIDENCE, insufficient=True, total_retrieved=total,
                            notes=notes + ["No evidence was retrieved."])
    context = format_context(evidence)
    raw = _call_text(model, ANSWER_PROMPT.format(context=context, question=question,
                                                 insufficient=INSUFFICIENT_EVIDENCE))
    clean, cited, unknown = map_citations(raw, evidence)
    insufficient = INSUFFICIENT_EVIDENCE.lower()[:60] in clean.lower()
    if insufficient:
        for e in evidence:
            e.citation_status = "not_cited"
        cited = []
    return AnswerResult(question, mode, clean.strip(), evidence=evidence, cited_ids=cited, unknown_ids=unknown,
                        total_retrieved=total, insufficient=insufficient, notes=notes, context=context)


def _only_sources(items: list, sources: Optional[list]) -> list:
    return [i for i in items if i.is_community or not sources or i.source in sources]


# --------------------------------------------------------------------------- #
# Traditional RAG
# --------------------------------------------------------------------------- #

def basic_search(store, question: str, model, top_k: int = 6, max_per_pdf: int = 2,
                 sources: Optional[list] = None) -> AnswerResult:
    vec = vs.retrieve_vector_evidence(store, question, k=top_k, max_per_pdf=max_per_pdf, sources=sources)
    final, total = rank_and_limit(vec, [], [], max_items=top_k)
    return answer_from_evidence(question, "Traditional RAG", final, model, total)


# --------------------------------------------------------------------------- #
# Local Search (graph neighbourhood + linked text units + community reports)
# --------------------------------------------------------------------------- #

def report_to_text(r: CommunityReport) -> str:
    lines = [f"Summary: {r.summary}", f"Importance rating: {r.rating:.0f}/10 — {r.rating_explanation}"]
    lines += [f"Finding: {f['summary']} — {f['explanation']}" for f in r.findings]
    return "\n".join(lines)


def report_to_evidence(r: CommunityReport) -> RetrievedEvidence:
    return RetrievedEvidence(text=report_to_text(r), retrieval_type="community", community_id=r.community_id,
                             title=r.title, provenance=list(r.locations))


@dataclass
class LocalRetrieval:
    matched: list = field(default_factory=list)
    facts: list = field(default_factory=list)          # facts kept after ranking/limits
    facts_total: int = 0
    chunk_items: list = field(default_factory=list)
    community_items: list = field(default_factory=list)


def local_evidence(kg: KnowledgeGraph, question: str, hops: int = 2, entity_store=None,
                   reports: Optional[list] = None, max_facts: int = MAX_GRAPH_FACTS,
                   max_prov: int = MAX_PROVENANCE_PER_FACT, max_chunks: int = MAX_LOCAL_CHUNKS) -> LocalRetrieval:
    """Microsoft-GraphRAG-style local retrieval, adapted:
      1. entities: exact/substring/fuzzy name match (existing find_matching_nodes);
         if no name literally matches, entity-embedding matches from ChromaDB;
      2. bounded 1-2 hop bidirectional neighbourhood (existing get_neighborhood_facts);
      3. hub protection: facts ranked (fewest hops, most supporting chunks) and capped;
      4. text units: the chunks behind those facts + chunks mentioning the matched entities,
         ranked by how many matched entities/facts point at them;
      5. community reports of the matched entities (level 0)."""
    matched = find_matching_nodes(question, kg)
    if not any(m.match_type in ("exact", "substring") for m in matched):
        seen = {m.node_key for m in matched}
        matched += [m for m in vs.semantic_entity_matches(entity_store, question, kg) if m.node_key not in seen]
    if not matched:
        return LocalRetrieval()

    g = kg.graph
    all_facts = get_neighborhood_facts(kg, matched, hops)
    facts = sorted(all_facts, key=lambda f: (f.hop, -len(f.provenance), f.subject_key, f.predicate, f.object_key))[:max_facts]

    links: dict = {}

    def touch(cid: str, hop: int) -> dict:
        rec = links.setdefault(cid, {"links": 0, "hop": hop, "facts": [], "entities": set(), "rels": set()})
        rec["hop"] = min(rec["hop"], hop)
        rec["links"] += 1
        return rec

    for f in facts:
        for p in f.provenance[:max_prov]:
            rec = touch(p["chunk_id"], f.hop)
            text = f"{f.subject} -[{f.predicate}]-> {f.object} (hop {f.hop})"
            if p.get("description"):
                text += f": {p['description']}"
            if text not in rec["facts"]:
                rec["facts"].append(text)
            rec["entities"] |= {f.subject_key, f.object_key}
            rec["rels"].add(f"{f.subject_key}|{f.predicate}|{f.object_key}")
    for m in matched:
        for (_s, _p, cid) in g.nodes[m.node_key].get("mentions", []):
            touch(cid, 0)["entities"].add(m.node_key)

    chunk_items = []
    for cid, rec in sorted(links.items(), key=lambda kv: (-kv[1]["links"], kv[1]["hop"], kv[0])):
        c = kg.chunk_lookup.get(cid)
        if c is None:
            continue
        chunk_items.append(RetrievedEvidence(
            text=c.chunk_text, retrieval_type="graph", source=c.source,
            page=(c.page - 1) if c.page is not None else None, page_number=c.page, chunk_id=cid,
            graph_hop=rec["hop"], graph_links=rec["links"], entity_ids=sorted(rec["entities"]),
            relationship_ids=sorted(rec["rels"]), graph_facts=rec["facts"][:5],
        ))
    chunk_items = chunk_items[:max_chunks]

    mk = {m.node_key for m in matched}
    comm = sorted((r for r in (reports or []) if r.level == 0 and mk & set(r.node_keys)),
                  key=lambda r: (-r.size, r.community_id))[:MAX_LOCAL_REPORTS]
    return LocalRetrieval(matched, facts, len(all_facts), chunk_items, [report_to_evidence(r) for r in comm])


def _local_notes(r: LocalRetrieval) -> list:
    return _truncation_note(len(r.facts), r.facts_total, "retrieved graph facts")


def local_search(kg: KnowledgeGraph, question: str, model, entity_store=None, reports: Optional[list] = None,
                 sources: Optional[list] = None) -> AnswerResult:
    r = local_evidence(kg, question, entity_store=entity_store, reports=reports)
    if not r.matched:
        return AnswerResult(question, "Local Search", NO_GRAPH_MATCH_MESSAGE, insufficient=True,
                            notes=["Gemini was not called: no graph entity matched the question."])
    final, total = rank_and_limit([], _only_sources(r.chunk_items, sources), r.community_items, max_items=MAX_EVIDENCE)
    res = answer_from_evidence(question, "Local Search", final, model, total, _local_notes(r))
    res.matched, res.facts = r.matched, r.facts
    return res


# --------------------------------------------------------------------------- #
# Hybrid RAG
# --------------------------------------------------------------------------- #

def hybrid_search(store, kg: KnowledgeGraph, question: str, model, top_k: int = 6, max_per_pdf: int = 2,
                  sources: Optional[list] = None, entity_store=None, reports: Optional[list] = None) -> AnswerResult:
    """ChromaDB vector evidence + graph evidence (+ community reports), merged.
    A chunk found by both channels is merged into one item. If the question
    matches no graph entity, the result is vector-only and says so."""
    vec = vs.retrieve_vector_evidence(store, question, k=top_k, max_per_pdf=max_per_pdf, sources=sources)
    r = local_evidence(kg, question, entity_store=entity_store, reports=reports)
    notes = _local_notes(r)
    if not r.matched:
        notes.append("No knowledge-graph entity matched the question, so this answer uses vector evidence only.")
    final, total = rank_and_limit(vec, _only_sources(r.chunk_items, sources), r.community_items, max_items=MAX_EVIDENCE)
    res = answer_from_evidence(question, "Hybrid RAG", final, model, total, notes)
    res.matched, res.facts = r.matched, r.facts
    return res


# --------------------------------------------------------------------------- #
# Global Search (map-reduce over community reports)
# --------------------------------------------------------------------------- #

class MapPoint(BaseModel):
    description: str = Field(description="One point that helps answer the question, based only on the reports.")
    score: int = Field(description="Importance 0-100 (0 = not helpful).")
    evidence_ids: list[str] = Field(description="Evidence IDs (like E3) of the reports this point comes from.")


class MapResponse(BaseModel):
    points: list[MapPoint] = Field(description="Points that help answer the question. Empty list if none.")


def global_search(reports: list, question: str, model, level: int = 0, batch_size: int = GLOBAL_BATCH_SIZE,
                  max_reports: int = GLOBAL_MAX_REPORTS, max_points: int = GLOBAL_MAX_POINTS) -> AnswerResult:
    """Corpus-level question answering over community reports (not top-k chunks).

    MAP     reports are split into batches; for each batch Gemini returns scored
            points, each citing the report IDs it used (IDs not in the batch are dropped).
    REDUCE  points with score 0 are discarded, the rest are ranked and Gemini writes the
            final answer from the points only.
    Simplifications vs Microsoft: reports are not shuffled, no dynamic community selection,
    and the level is chosen by the caller (0 = coarsest)."""
    mode = "Global Search"
    chosen = sorted((r for r in reports if r.level == level), key=lambda r: (-r.size, -r.rating, r.community_id))
    if not chosen:
        return AnswerResult(question, mode, "No community reports are available. Build the knowledge graph first "
                            "(communities need entities connected by relationships).", insufficient=True,
                            notes=["Gemini was not called: no community reports."])
    total = len(chosen)
    chosen = chosen[:max_reports]
    items = assign_ids([report_to_evidence(r) for r in chosen])
    notes = _truncation_note(len(items), total, "community reports")

    points: list = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        ids = {e.evidence_id for e in batch}
        out = _call_structured(model, MapResponse, MAP_PROMPT.format(context=format_context(batch), question=question))
        for p in (out.points if out else []):
            valid = [x.strip().upper() for x in p.evidence_ids if x.strip().upper() in ids]
            if p.score > 0 and valid and p.description.strip():
                points.append({"description": p.description.strip(), "score": int(p.score), "evidence_ids": valid})
    points.sort(key=lambda p: -p["score"])
    kept = points[:max_points]
    notes.append(f"Map stage: {len(items)} report(s) in {-(-len(items) // batch_size)} batch(es) produced "
                 f"{len(points)} point(s)" + (f"; the {len(kept)} most important were used." if len(points) > len(kept) else "."))

    if not kept:
        return AnswerResult(question, mode, INSUFFICIENT_EVIDENCE, evidence=items, total_retrieved=total,
                            insufficient=True, notes=notes + ["No community report contained a helpful point."],
                            findings=[])
    used_ids = {i for p in kept for i in p["evidence_ids"]}
    used = [e for e in items if e.evidence_id in used_ids]
    def _tags(ids: list) -> str:
        return "".join("[" + i + "]" for i in ids)

    points_text = "\n".join(
        "{} (importance {}): {}".format(_tags(p["evidence_ids"]), p["score"], p["description"]) for p in kept
    )
    raw = _call_text(model, REDUCE_PROMPT.format(points=points_text, question=question, insufficient=INSUFFICIENT_EVIDENCE))
    clean, cited, unknown = map_citations(raw, used)
    insufficient = INSUFFICIENT_EVIDENCE.lower()[:60] in clean.lower()
    if insufficient:
        for e in used:
            e.citation_status = "not_cited"
        cited = []
    return AnswerResult(question, mode, clean.strip(), evidence=used, cited_ids=cited, unknown_ids=unknown,
                        total_retrieved=total, insufficient=insufficient, notes=notes, context=points_text,
                        findings=kept)
