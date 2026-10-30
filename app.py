"""
RAG Document Q&A — multi-PDF Traditional RAG + Microsoft-GraphRAG-inspired search.

Upload up to 10 PDFs. Everything is indexed together:

  PDFs ─► chunks (source, page, page_number, chunk_id, content_hash)
      ├─► ChromaDB (persistent vector database, incremental indexing)
      └─► Gemini entity/relationship extraction ─► NetworkX knowledge graph
            ─► hierarchical Leiden communities ─► community reports

Retrieval Mode:
  Traditional RAG   ChromaDB similarity search across all PDFs
  Local Search      entity neighbourhood + linked text chunks + community reports
  Global Search     map-reduce over community reports (corpus-level questions)
  Hybrid RAG        vector evidence + graph evidence, merged

Every answer cites evidence as [E1], [E2]... and the app (not Gemini) maps those
IDs to the real PDF / page / chunk. Backend modules: ingestion.py, vector_store.py,
graph_rag.py, search.py, evidence.py. See README.md for what is and is not
Microsoft GraphRAG.

Run with:
    streamlit run app.py

Requires a GOOGLE_API_KEY (Gemini API key) in a .env file (or pasted into the sidebar).
"""

import os
import json
import hashlib
from collections import OrderedDict
from typing import Optional
import llm_cache
import traceback

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import search
import llm_cache
import vector_store as vs
from graph_rag import (
    EXTRACTION_VERSION,
    REPORT_VERSION,
    ChunkGraphData,
    Community,
    CommunityReport,
    build_knowledge_graph,
    chunks_signature,
    community_stats,
    detect_communities,
    extract_graph_data,
    generate_community_reports,
    graph_stats,
    load_cache,
    render_graph_svg,
    save_cache,
    split_cached,
    summarize_extraction,
)
from ingestion import MAX_PDFS, prepare_uploads, selection_error


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

load_dotenv()

GEMINI_MODEL_OPTIONS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]

# Extraction / report caches survive restarts (plain JSON files, keyed by chunk_id /
# report-context hash). Delete the files to force a full re-extraction.
GRAPH_CACHE_PATH = ".graph_extraction_cache.json"
REPORT_CACHE_PATH = ".graph_report_cache.json"
COMMUNITY_CACHE_PATH = ".graph_community_cache.json"
ENTITY_INDEX_META_PATH = ".graph_entity_index_meta.json"

MODES = ("Traditional RAG", "Local Search", "Global Search", "Hybrid RAG")
GRAPH_MODES = MODES[1:]

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")


def show_html(html: str, height: int):
    """Shows a self-generated HTML/SVG snippet. Newer Streamlit deprecates
    components.html in favour of st.iframe; use whichever this install has."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=True)


# --------------------------------------------------------------------------- #
# Look & feel — shared CSS so the app doesn't look like Streamlit defaults
# --------------------------------------------------------------------------- #

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

        :root {
            --accent: #5B5FEF;
            --accent-2: #8B5CF6;
            --accent-light: #EEF0FF;
            --ink: #1A1D29;
            --muted: #6B7280;
            --border: #E7E8EE;
            --good: #1A7F37;
            --good-bg: #DAFBE1;
            --warn: #B54708;
            --warn-bg: #FFF1E0;
        }

        /* Pull content flush after the sidebar instead of leaving a big
           centered gap on wide screens, and give it room to breathe. */
        section.main > div.block-container {
            max-width: 1280px !important;
            margin-left: 0 !important;
            padding-top: 1.4rem !important;
            padding-left: 2.2rem !important;
            padding-right: 2.2rem !important;
        }
        div[data-testid="stAppViewContainer"] > .main { padding-left: 0 !important; }

        /* Header */
        .app-header-left { display: flex; align-items: center; gap: 14px; }
        .app-header-left .icon {
            font-size: 2.2rem; line-height: 1;
            filter: drop-shadow(0 2px 4px rgba(91,95,239,0.25));
        }
        .app-header-left h1 {
            font-size: 1.65rem; font-weight: 800; margin: 0; color: var(--ink);
            letter-spacing: -0.02em;
        }
        .app-subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 3px; }

        /* Section spacing */
        .section-block { margin-top: 1.6rem; margin-bottom: 0.6rem; }
        .section-block h3 { margin-bottom: 0.2rem; font-weight: 700; letter-spacing: -0.01em; }
        hr.soft-divider { border: none; border-top: 1px solid var(--border); margin: 1.6rem 0; }

        /* Upload card */
        div[data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(135deg, #FAFAFF, #F5F6FF) !important;
            border: 1.5px dashed #C7C9FA !important;
            border-radius: 16px !important;
            transition: border-color 0.15s ease;
        }
        div[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

        /* Document chips */
        .doc-chip-row { margin: 0.9rem 0 0.3rem 0; }
        .doc-chip {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--accent-light); color: var(--accent);
            border: 1px solid #D9DAFB; border-radius: 999px;
            padding: 6px 14px; margin: 3px 8px 3px 0;
            font-size: 0.82rem; font-weight: 600;
            box-shadow: 0 1px 2px rgba(91,95,239,0.08);
        }
        .doc-chip .count {
            background: var(--accent); color: white; border-radius: 999px;
            padding: 0 7px; font-size: 0.72rem; font-weight: 700;
        }

        /* Buttons */
        div[data-testid="stButton"] button {
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            font-size: 0.85rem !important;
            padding: 0.55rem 1rem !important;
            transition: all 0.12s ease !important;
        }
        div[data-testid="stButton"] button:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            box-shadow: 0 2px 6px rgba(91,95,239,0.15);
        }
        .primary-btn button {
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(91,95,239,0.3);
        }
        .primary-btn button:hover { filter: brightness(1.06); color: white !important; }
        .clear-btn button {
            background: #FEF2F2 !important;
            border-color: #FECACA !important;
            color: #B42318 !important;
        }
        .clear-btn button:hover { border-color: #B42318 !important; color: #B42318 !important; }

        /* Compare feature card */
        .compare-banner {
            background: linear-gradient(135deg, #EEF0FF, #F8F5FF);
            border: 1px solid #D9DAFB; border-radius: 18px;
            padding: 20px 22px; margin: 0.6rem 0 1rem 0;
            box-shadow: 0 2px 10px rgba(91,95,239,0.08);
        }
        .compare-banner .title { font-weight: 800; color: var(--ink); font-size: 1.1rem; letter-spacing: -0.01em; }
        .compare-banner .desc { color: var(--muted); font-size: 0.85rem; margin-top: 3px; }

        .compare-doc-card {
            border: 1px solid var(--border); border-radius: 14px;
            padding: 16px 18px; height: 100%; background: #FAFAFB;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
        .compare-doc-card .doc-name { font-weight: 700; color: var(--accent); margin-bottom: 6px; }

        .diff-card { border-radius: 14px; padding: 16px 18px; height: 100%; }
        .diff-card.similar { background: var(--good-bg); border: 1px solid #BEF0CE; }
        .diff-card.differ { background: var(--warn-bg); border: 1px solid #FDDCAB; }
        .diff-card .diff-title { font-weight: 700; margin-bottom: 6px; }
        .diff-card.similar .diff-title { color: var(--good); }
        .diff-card.differ .diff-title { color: var(--warn); }

        /* Answer container */
        .answer-card { border: 1px solid var(--border); border-radius: 14px; padding: 4px 2px; }
        .source-tag {
            display: inline-block; background: #F3F4F6; color: #374151;
            border-radius: 6px; padding: 1px 8px; font-size: 0.72rem;
            font-weight: 600; margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Helpers shared with Compare Documents
# --------------------------------------------------------------------------- #

def format_chunks_for_prompt(chunks: list[Document], multi_doc: bool) -> str:
    """Joins retrieved chunks into one text block for the prompt, each
    labeled with its source document (when there's more than one) and page."""
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page")
        source = chunk.metadata.get("source", "document")
        bits = [f"Chunk {i}"]
        if multi_doc:
            bits.append(f"from '{source}'")
        if page is not None:
            bits.append(f"page {page + 1}")
        formatted.append(f"[{', '.join(bits)}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(formatted)


# --------------------------------------------------------------------------- #
# Step 5: LLM-as-a-Judge — evaluate each generated answer automatically
# --------------------------------------------------------------------------- #

class AnswerEvaluation(BaseModel):
    """The judge's structured verdict on one generated answer."""
    groundedness_score: int = Field(
        description="1-5: how well the answer is supported by the retrieved context, with no invented facts. 5 = fully supported, 1 = mostly unsupported/hallucinated."
    )
    groundedness_reasoning: str = Field(
        description="One or two sentences explaining the groundedness score, referencing specific context if relevant."
    )
    relevance_score: int = Field(
        description="1-5: how directly the answer addresses the actual question asked. 5 = fully answers it, 1 = off-topic or non-answer."
    )
    relevance_reasoning: str = Field(description="One or two sentences explaining the relevance score.")
    verdict: str = Field(description="Overall judgment: 'Pass' if both scores are 4 or higher, otherwise 'Needs review'.")


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are an impartial evaluator (an "LLM-as-a-judge") grading a RAG system's answer.

You are NOT answering the question yourself — you are critiquing whether the answer below is
well-grounded in the given context and actually relevant to the question. Be strict: if the
answer states something not present in the context, that is a groundedness failure, even if the
statement happens to be true in general.

Context that was given to the answering model:
{context}

Question that was asked:
{question}

Answer that was generated:
{answer}

Evaluate this answer now."""
)


def judge_answer(question: str, context: str, answer: str, model_name: str, api_key: str) -> AnswerEvaluation:
    """A second, independent Gemini call that grades the first one's answer
    strictly against the retrieved context (the LLM-as-a-judge pattern)."""
    cache_payload = (model_name, question.strip(), context, answer)
    cached = llm_cache.get("answer_judge_v1", cache_payload)
    if cached is not None:
        return cached
    judge_model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    structured_judge = judge_model.with_structured_output(AnswerEvaluation)
    judge_chain = JUDGE_PROMPT | structured_judge
    result = judge_chain.invoke({"context": context, "question": question, "answer": answer})
    llm_cache.set("answer_judge_v1", cache_payload, result)
    return result


# --------------------------------------------------------------------------- #
# Compare Documents — a dedicated feature for 2+ indexed PDFs
# --------------------------------------------------------------------------- #

class DocSummary(BaseModel):
    document: str = Field(description="The exact document/source name being summarized.")
    summary: str = Field(description="A concise 2-4 sentence summary of just this document's content.")


class DocumentComparison(BaseModel):
    """Structured, side-by-side comparison across every indexed document."""
    overview: str = Field(description="1-2 sentences framing how the documents relate to each other overall.")
    document_summaries: list[DocSummary] = Field(description="One summary per document, in the order given.")
    similarities: list[str] = Field(description="3-6 short bullet points on what the documents have in common.")
    differences: list[str] = Field(description="3-6 short bullet points on the key differences between the documents.")


COMPARE_PROMPT = ChatPromptTemplate.from_template(
    """You are comparing {num_docs} documents based ONLY on the context below. Each chunk is
labeled with the document it came from. Produce a structured comparison: a short overview,
a concise summary of each document individually, a list of similarities, and a list of key
differences. Be specific and cite document names when relevant. If the context is too sparse
to compare on some point, say so rather than guessing.

Context:
{context}

Documents being compared: {doc_names}
"""
)


def generate_comparison(vector_store: Chroma, doc_names: list[str], model_name: str, api_key: str) -> DocumentComparison:
    """Retrieves a generous, balanced sample of chunks from every indexed
    document and asks Gemini for a structured comparison, rather than a
    freeform chat answer — this is what powers the dedicated 'Compare
    Documents' feature."""
    per_doc_k = max(2, min(6, 36 // max(len(doc_names), 1)))   # keeps the prompt bounded with up to 10 PDFs
    chunks: list[Document] = []
    for name in doc_names:
        try:
            docs = vector_store.similarity_search(name, k=per_doc_k, filter={"source": name})
        except Exception:
            docs = []
        if not docs:
            try:
                docs = vector_store.similarity_search(" ", k=per_doc_k, filter={"source": name})
            except Exception:
                docs = []
        chunks.extend(docs)

    context = format_chunks_for_prompt(chunks, multi_doc=True)
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)
    structured_model = model.with_structured_output(DocumentComparison)
    chain = COMPARE_PROMPT | structured_model
    return chain.invoke({"context": context, "num_docs": len(doc_names), "doc_names": ", ".join(doc_names)})


def render_comparison(comparison: DocumentComparison):
    st.markdown(f"**Overview:** {comparison.overview}")
    st.write("")

    # One tab per document: readable for up to 10 PDFs (a 10-column layout is not).
    tabs = st.tabs([d.document for d in comparison.document_summaries]) if comparison.document_summaries else []
    for col, doc in zip(tabs, comparison.document_summaries):
        with col:
            st.markdown(
                f"""<div class="compare-doc-card">
                <div class="doc-name">📄 {doc.document}</div>
                <div style="font-size:0.88rem;color:#374151;line-height:1.5;">{doc.summary}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        similar_html = "".join(f"<li>{s}</li>" for s in comparison.similarities)
        st.markdown(
            f"""<div class="diff-card similar"><div class="diff-title">🤝 Similarities</div>
            <ul style="margin:0;padding-left:18px;font-size:0.88rem;color:#374151;">{similar_html}</ul></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        diff_html = "".join(f"<li>{d}</li>" for d in comparison.differences)
        st.markdown(
            f"""<div class="diff-card differ"><div class="diff-title">⚡ Key differences</div>
            <ul style="margin:0;padding-left:18px;font-size:0.88rem;color:#374151;">{diff_html}</ul></div>""",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Suggested questions
# --------------------------------------------------------------------------- #

def get_suggested_questions(doc_names: list[str]) -> list[str]:
    """A small starter set of one-click questions for the chat box. The
    explicit "compare" ask lives in its own dedicated feature above this,
    so these stay focused on regular Q&A."""
    single_doc_questions = [
        "Summarize this document in a few bullet points",
        "What are the key takeaways?",
        "Are there any important dates, numbers, or figures mentioned?",
    ]
    if len(doc_names) >= 2:
        multi_doc_questions = [
            "Which document covers which topics?",
            "Summarize each document separately",
            "What questions would someone need both documents to answer?",
        ]
        return multi_doc_questions + single_doc_questions
    return single_doc_questions


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #

def _score_tier(score: int) -> tuple[str, str]:
    if score >= 4:
        return "#1A7F37", "#DAFBE1"
    if score == 3:
        return "#B54708", "#FFF1E0"
    return "#B42318", "#FEE4E2"


def _score_card(label: str, score: int, reasoning: str) -> str:
    text_color, bg_color = _score_tier(score)
    pct = score / 5 * 100
    return f"""
    <div style="background:{bg_color}0D;border:1px solid {bg_color};border-radius:12px;
    padding:14px 16px;height:100%;">
        <div style="font-size:0.78rem;font-weight:600;color:#57606A;text-transform:uppercase;
        letter-spacing:0.03em;margin-bottom:2px;">{label}</div>
        <div style="font-size:1.9rem;font-weight:700;color:{text_color};line-height:1.2;">
        {score}<span style="font-size:1rem;font-weight:500;color:#8B949E;">/5</span></div>
        <div style="background:#E7E9EC;border-radius:999px;height:6px;margin:8px 0 10px 0;overflow:hidden;">
            <div style="background:{text_color};width:{pct}%;height:100%;border-radius:999px;"></div>
        </div>
        <div style="font-size:0.83rem;color:#57606A;line-height:1.4;">{reasoning}</div>
    </div>
    """


def render_evaluation(evaluation: "AnswerEvaluation | None"):
    if evaluation is None:
        return

    is_pass = evaluation.verdict.strip().lower() == "pass"
    badge_color = "#1A7F37" if is_pass else "#B54708"
    badge_bg = "#DAFBE1" if is_pass else "#FFF1E0"
    icon = "✅" if is_pass else "⚠️"

    st.markdown(
        f"""<div style="display:inline-block;padding:5px 16px;border-radius:999px;
        background:{badge_bg};color:{badge_color};font-weight:600;font-size:0.85rem;
        margin:6px 0;box-shadow:0 1px 2px rgba(0,0,0,0.06);">
        {icon}&nbsp;LLM Evaluation&nbsp;·&nbsp;{evaluation.verdict}</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("🧠 View evaluation details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                _score_card("Groundedness", evaluation.groundedness_score, evaluation.groundedness_reasoning),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                _score_card("Relevance", evaluation.relevance_score, evaluation.relevance_reasoning),
                unsafe_allow_html=True,
            )
        st.caption(
            "Scored by a second, independent Gemini call that grades the answer "
            "strictly against the retrieved context (the LLM-as-a-judge pattern)."
        )


def render_doc_chips(documents: "OrderedDict[str, list[Document]]"):
    if not documents:
        return
    chips = "".join(
        f'<span class="doc-chip">📄 {name} <span class="count">{len(chunks)}</span></span>'
        for name, chunks in documents.items()
    )
    st.markdown(f'<div class="doc-chip-row">{chips}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state + document management (ChromaDB is the source of truth)
# --------------------------------------------------------------------------- #

def ensure_state():
    defaults = {
        "documents": OrderedDict(),   # source name -> list[Document] (chunks, with full metadata)
        "vector_store": None,         # langchain Chroma, persistent collection "rag_documents"
        "embeddings": None,
        "restored": False,            # documents already re-read from ChromaDB after a restart?
        "chat_history": [],           # Traditional RAG turns
        "pending_question": None,
        "comparison_result": None,
        # knowledge graph + communities (built on demand, reused for every question)
        "knowledge_graph": None,
        "graph_signature": None,      # fingerprint of the chunks the graph was built from
        "communities": [],
        "community_reports": [],
        "entity_store": None,         # Chroma collection of entity embeddings (Local Search)
        "extraction_cache": None,     # chunk_id -> ChunkGraphData (loaded from disk once)
        "report_cache": None,         # context hash -> CommunityReport
        "search_history": [],         # AnswerResult list for Local / Global / Hybrid
        "graph_pending": None,
        "last_build_summary": None,
        "last_upload_results": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state["extraction_cache"] is None:
        st.session_state["extraction_cache"] = load_cache(GRAPH_CACHE_PATH, EXTRACTION_VERSION, ChunkGraphData)
    if st.session_state["report_cache"] is None:
        st.session_state["report_cache"] = load_cache(REPORT_CACHE_PATH, REPORT_VERSION, CommunityReport)


def get_store(api_key: str):
    """The persistent ChromaDB collection (created lazily: it needs the API key
    for the embedding model)."""
    if st.session_state["vector_store"] is None:
        embeddings = vs.make_embeddings(api_key)
        st.session_state["embeddings"] = embeddings
        st.session_state["vector_store"] = vs.open_store(embeddings)
    return st.session_state["vector_store"]


def restore_documents(api_key: Optional[str] = None):
    """After an app restart the chunks are still in ChromaDB: read them back so
    the PDFs show up again without re-uploading or re-embedding. (Reading stored
    chunks does not load the embedding model.)"""
    st.session_state["restored"] = True
    try:
        store = get_store(api_key)
        st.session_state["documents"] = vs.load_documents(store)
    except Exception as e:
        st.warning(f"Could not read the existing ChromaDB index ({type(e).__name__}: {e}). Starting empty.")


def all_chunks() -> list:
    return [c for chunks in st.session_state["documents"].values() for c in chunks]


def known_hashes() -> dict:
    """content_hash -> source of every indexed PDF (duplicate detection by content)."""
    return {chunks[0].metadata["content_hash"]: name
            for name, chunks in st.session_state["documents"].items() if chunks}


def reset_conversations():
    st.session_state["chat_history"] = []
    st.session_state["comparison_result"] = None
    st.session_state["search_history"] = []


def reset_graph_state():
    st.session_state["knowledge_graph"] = None
    st.session_state["graph_signature"] = None
    st.session_state["communities"] = []
    st.session_state["community_reports"] = []
    st.session_state["entity_store"] = None
    st.session_state["search_history"] = []
    st.session_state["last_build_summary"] = None


def clear_everything():
    store = st.session_state["vector_store"]
    for name in list(st.session_state["documents"]):
        try:
            if store is not None:
                vs.delete_source(store, name)
        except Exception:
            pass
    st.session_state["documents"] = OrderedDict()
    st.session_state["pending_question"] = None
    st.session_state["last_upload_results"] = []
    reset_conversations()
    reset_graph_state()


def handle_index_pdfs(uploaded_files, api_key, chunk_size, chunk_overlap):
    """Parse every selected PDF independently, then add ONLY the new chunks to
    ChromaDB. One bad PDF never stops the others."""
    err = selection_error(len(uploaded_files))
    if err:
        st.error(err)
        return
    files = [(f.name, f.getvalue()) for f in uploaded_files]
    docs = st.session_state["documents"]
    results = prepare_uploads(files, known_hashes(), set(docs), len(docs), chunk_size, chunk_overlap)

    store = None
    for r in results:
        if r.status != "indexed":
            continue
        try:
            store = get_store(api_key)
            with st.spinner(f"Embedding {r.filename} ({len(r.chunks)} chunks)..."):
                vs.add_chunks(store, r.chunks)
            docs[r.source] = r.chunks
        except Exception as e:      # embedding / rate-limit / ChromaDB failure for THIS file only
            try:
                if store is not None:
                    vs.delete_source(store, r.source)     # roll back a half-indexed PDF
            except Exception:
                pass
            r.status = "error"
            r.message = f"embedding/indexing failed ({type(e).__name__}: {e})"
    st.session_state["last_upload_results"] = results
    if any(r.status == "indexed" for r in results):
        reset_conversations()       # the document set changed; an old graph is detected as stale automatically


def handle_remove_doc(name, api_key):
    try:
        vs.delete_source(get_store(api_key), name)       # only this PDF's chunks
    except Exception as e:
        st.error(f"Could not remove '{name}' from ChromaDB: {e}")
        return
    st.session_state["documents"].pop(name, None)
    reset_conversations()


def render_upload_results():
    icons = {"indexed": "✅", "duplicate": "♻️", "empty": "⚠️", "error": "❌", "rejected": "🚫"}
    for r in st.session_state["last_upload_results"]:
        line = f"{icons.get(r.status, '•')} **{r.filename}** — {r.message}"
        (st.error if r.status in ("error", "rejected") else st.warning if r.status in ("empty", "duplicate") else st.success)(line)


# --------------------------------------------------------------------------- #
# Knowledge graph indexing UI (extraction -> graph -> communities -> reports)
# --------------------------------------------------------------------------- #

def graph_is_stale() -> bool:
    return (st.session_state["knowledge_graph"] is not None
            and st.session_state["graph_signature"] != chunks_signature(all_chunks()))

def _community_cache_key(chunks: list) -> str:
    """
    Stable cache key for the current document set.

    Communities depend on the extracted graph, which in turn depends
    on the current chunks. Include the Leiden parameters so changing
    them automatically invalidates the cache.
    """
    payload = {
        "chunks_signature": chunks_signature(chunks),
        "extraction_version": EXTRACTION_VERSION,
        "max_cluster_size": 10,
        "leiden_seed": 42,
        "cache_version": 1,
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def load_community_cache(chunks: list):
    """
    Load communities only when the cache belongs to the current
    document set and current Leiden configuration.
    """
    key = _community_cache_key(chunks)

    if not os.path.exists(COMMUNITY_CACHE_PATH):
        return None

    try:
        with open(COMMUNITY_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("cache_key") != key:
            return None

        communities_data = data.get("communities", [])

        return [
            Community.model_validate(item)
            for item in communities_data
        ]

    except Exception:
        return None


def save_community_cache(chunks: list, communities: list):
    """
    Persist the detected communities so hierarchical Leiden does not
    need to run again for the same document set.
    """
    key = _community_cache_key(chunks)

    payload = {
        "cache_key": key,
        "communities": [
            c.model_dump(mode="json")
            for c in communities
        ],
    }

    tmp_path = COMMUNITY_CACHE_PATH + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    os.replace(tmp_path, COMMUNITY_CACHE_PATH)


def build_graph_index(model_name: str, api_key: str, delay: float = 0.0, workers: int = 3, max_reports: int = 20):
    """The Microsoft-GraphRAG-style indexing pipeline, with a cache at every
    expensive step:

      chunks ─► Gemini extraction (cached per chunk_id) ─► NetworkX graph
             ─► hierarchical Leiden communities ─► Gemini community reports (cached per community content)
             ─► entity embeddings in ChromaDB (for Local Search)
    """
    chunks = all_chunks()
    cache = st.session_state["extraction_cache"]
    cached, new = split_cached(chunks, cache)
    summary: dict = {"chunks": len(chunks), "reused": cached}

    # ---- 1. entity & relationship extraction (only chunks not in the cache) ----
    bar = st.progress(0.0, text=f"Extracting entities and relationships... ({cached} chunks reused from cache, {new} to extract)")
    counts = {"entities": 0, "relationships": 0}

    def on_extract(done, total, res):
        counts["entities"] += len(res.entities)
        counts["relationships"] += len(res.triples)
        bar.progress(done / total, text=f"Extracting — chunk {done}/{total} · entities {counts['entities']} · "
                                         f"relationships {counts['relationships']}")

    try:
        results = extract_graph_data(
            chunks, model_name, api_key, on_progress=on_extract, cache=cache,
            delay_between_calls=delay, max_workers=workers,
            on_checkpoint=lambda: save_cache(cache, GRAPH_CACHE_PATH, EXTRACTION_VERSION),   # progress survives Stop
        )
    except Exception as e:
        bar.empty()
        st.error(f"Graph extraction failed ({type(e).__name__}: {e}). Chunks finished so far are cached.")
        save_cache(cache, GRAPH_CACHE_PATH, EXTRACTION_VERSION)
        return
    save_cache(cache, GRAPH_CACHE_PATH, EXTRACTION_VERSION)
    ex = summarize_extraction(results)
    summary.update(extracted=new, failed=ex["error"])

    # ---- 2. graph ----
    kg = build_knowledge_graph(results)
    gs = graph_stats(kg)
    summary.update(entities=gs["nodes"], relationships=gs["edges"])

    # ---- 3. communities ----

    cached_communities = load_community_cache(chunks)

    if cached_communities is not None:
        communities = cached_communities

        bar.progress(
            1.0,
            text=f"Loaded {len(communities)} communities from cache."
        )

        summary["communities_reused"] = True

    else:
        bar.progress(
            1.0,
            text="Detecting communities (hierarchical Leiden)..."
        )

        communities = []

        try:
            communities = detect_communities(kg)

            save_community_cache(chunks, communities)

            summary["communities_reused"] = False

        except Exception as e:
            st.warning(
                f"Community detection failed: {e}. "
                "Local Search and Hybrid RAG still work; "
                "Global Search needs communities."
            )

    summary["communities"] = len(communities)

    # ---- 4. community reports (cached by content) ----
    reports, failures = [], []
    if communities:
        rbar_state = {"reused": 0}

        def on_report(i, total, rep, reused):
            rbar_state["reused"] += int(reused)
            bar.progress(i / total, text=f"Community reports — {i}/{total} ({rbar_state['reused']} reused from cache)")

        try:
            report_cache = st.session_state["report_cache"]
            reports, failures = generate_community_reports(
                kg, communities, model_name, api_key, cache=report_cache, on_progress=on_report,
                max_reports=max_reports, max_workers=workers, delay_between_calls=delay,     # same speed/quota controls as extraction
                on_checkpoint=lambda: save_cache(report_cache, REPORT_CACHE_PATH, REPORT_VERSION),
            )
        except Exception as e:
            st.warning(f"Community report generation failed ({type(e).__name__}: {e}).")
        save_cache(st.session_state["report_cache"], REPORT_CACHE_PATH, REPORT_VERSION)
    summary.update(reports=len(reports), report_failures=len(failures))
    bar.empty()

    # ---- 5. entity embeddings in ChromaDB (Local Search: find entities by meaning) ----

    entity_store = None

    try:
        get_store(api_key)
        embeddings = st.session_state["embeddings"]

        current_graph_signature = chunks_signature(chunks)

        reuse_entity_index = False

        if os.path.exists(ENTITY_INDEX_META_PATH):
            try:
                with open(
                    ENTITY_INDEX_META_PATH,
                    "r",
                    encoding="utf-8",
                ) as f:
                    entity_meta = json.load(f)

                reuse_entity_index = (
                    entity_meta.get("graph_signature")
                    == current_graph_signature
                    and entity_meta.get("embedding_model")
                    == vs.EMBEDDING_MODEL
                    and entity_meta.get("extraction_version")
                    == EXTRACTION_VERSION
                )

            except Exception:
                reuse_entity_index = False

        if reuse_entity_index:
            entity_store = vs.open_entity_store(embeddings)

            st.info(
                "♻️ Reused existing entity embedding index."
            )

        else:
            entity_store = vs.rebuild_entity_index(
                embeddings,
                kg,
            )

            with open(
                ENTITY_INDEX_META_PATH,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "graph_signature": current_graph_signature,
                        "embedding_model": vs.EMBEDDING_MODEL,
                        "extraction_version": EXTRACTION_VERSION,
                    },
                    f,
                    indent=2,
                )

            st.success(
                "✅ Entity embedding index created."
            )

    except Exception as e:
        st.warning(
            f"Could not build the entity embedding index "
            f"({type(e).__name__}: {e}). "
            "Local Search will match entity names by text only."
        )

    st.session_state.update(
        knowledge_graph=kg, graph_signature=chunks_signature(chunks), communities=communities,
        community_reports=reports, entity_store=entity_store, search_history=[], last_build_summary=summary,
    )
    for cid, err in failures:
        st.warning(f"Community report {cid} failed: {err}")
    if ex["error"]:
        st.warning(f"{ex['error']} of {len(chunks)} chunk(s) failed extraction (e.g. API rate limit). "
                   "Click Build / Update again to retry only those chunks.")


def render_graph_index(api_key: str, model_name: str):
    """Build controls + graph / community statistics + visualisation. Returns True
    when a current (non-stale) graph exists."""
    chunks = all_chunks()
    cached, new = split_cached(chunks, st.session_state["extraction_cache"])
    stale = graph_is_stale()
    if stale:
        st.warning("The documents changed since the knowledge graph was built. Click **Build / Update Knowledge Graph** "
                   "(only chunks that were never extracted before cost Gemini calls).")

    delay = st.slider("Pause between Gemini extraction calls (seconds)", 0.0, 8.0, 0.0, 0.5, key="extract_delay",
                      help="Minimum gap between the START of two Gemini calls, shared by ALL workers. The free tier "
                           "allows roughly 15 requests per minute: 4 seconds = at most 15 per minute (slower, but no "
                           "429 errors). 0 = no limit.")
    workers = st.slider("Parallel extraction workers", 1, 10, 3, key="extract_workers",
                        help="Chunks processed at the same time. Faster when you have quota to spare (0 pause); with a "
                             "pause above 0 the pause caps the speed, so extra workers add nothing. Too many workers "
                             "with no pause cause 429 rate-limit failures (failed chunks are retried on the next build).")
    max_reports = st.slider("Community reports to write (largest communities first)", 5, 30, 20, key="max_reports",
                            help="Each report is one Gemini call. Global Search reads at most 20 reports and Local "
                                 "Search at most 2 per question, so more than 20 rarely helps. Reports are cached: "
                                 "raising this later only writes the new ones.")
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    clicked = st.button("🕸️ Build / Update Knowledge Graph", key="build_graph")
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"{len(chunks)} chunks · {cached} already extracted (cached) · {new} need extraction "
               f"(≈ {new} Gemini calls) · extraction is reused between questions and restarts.")
    if clicked:
        if not api_key:
            st.error("Please provide a Gemini API key in the sidebar first.")
            st.stop()
        build_graph_index(model_name, api_key, delay, workers, max_reports)
        stale = graph_is_stale()

    kg = st.session_state["knowledge_graph"]
    if kg is None:
        st.info("Click **Build / Update Knowledge Graph** to extract entities and relationships from your documents.")
        return False

    b = st.session_state["last_build_summary"]
    if b:
        st.caption(f"Last build: {b['chunks']} chunks ({b['reused']} reused, {b['extracted']} extracted, {b['failed']} failed) · "
                   f"{b['entities']} entities · {b['relationships']} relationships · {b['communities']} communities · "
                   f"{b['reports']} reports")

    st.markdown("**Knowledge Graph**")
    stats = graph_stats(kg)
    cstats = community_stats(st.session_state["communities"], kg)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entities", stats["nodes"])
    c2.metric("Relationships", stats["edges"])
    c3.metric("Communities", cstats["communities"])
    c4.metric("Largest community", cstats["largest_community"])

    with st.expander("Graph Statistics"):
        st.write(f"Isolated entities (no relationship): {stats['isolated_nodes']} · connected components: "
                 f"{stats['connected_components']} (largest: {stats['largest_component_size']})")
        st.write(f"Community levels: {cstats['levels']} · top-level communities: {cstats['top_level_communities']} · "
                 f"entities in a community: {cstats['entities_in_communities']} ({cstats['coverage']:.0%})")
        if stats["top_entities_by_degree"]:
            st.write("Top entities by degree: " + ", ".join(f"{n} ({d})" for n, d in stats["top_entities_by_degree"]))
        for u, v, data in sorted(kg.graph.edges(data=True), key=lambda e: (e[0], e[1], e[2]["predicate"])):
            srcs = sorted({p["source"] for p in data["provenance"]})
            st.text(f"{kg.graph.nodes[u]['name']} -[{data['predicate']}]-> {kg.graph.nodes[v]['name']}   ({', '.join(srcs)})")

    reports = st.session_state["community_reports"]
    with st.expander(f"Community Reports ({len(reports)})"):
        if not reports:
            st.write("No community reports (communities need entities connected by relationships).")
        for r in reports:
            st.markdown(f"**{r.title}**  ·  `{r.community_id}` · level {r.level} · {r.size} entities · rating {r.rating:.0f}/10")
            st.write(r.summary)
            for f in r.findings:
                st.markdown(f"- **{f['summary']}** — {f['explanation']}")
            locs = "; ".join(f"{l['source']} p.{l['page_number']}" for l in r.locations[:12])
            st.caption(f"Built from: {locs}" + (" …" if len(r.locations) > 12 else ""))
            st.markdown("---")

    with st.expander("🕸️ Knowledge Graph Visualization", expanded=False):
        show_isolated = st.checkbox("Show isolated nodes (entities with no relationships)", value=False,
                                    key="graph_show_isolated")
        graph_html, graph_h = render_graph_svg(kg, include_isolated=show_isolated)
        if graph_html:
            show_html(graph_html, graph_h)
            st.caption("Node colour = entity type · node size = number of connections · "
                       "hover a node for details · arrows follow the fact's direction.")
        else:
            st.info("No relationships to draw yet. Tick 'Show isolated nodes' to see the extracted entities.")
    return not stale


# --------------------------------------------------------------------------- #
# Shared Answer / Sources / Evidence renderer (used by EVERY retrieval mode)
# --------------------------------------------------------------------------- #
#
# Everything shown here comes from RetrievedEvidence objects built from real
# metadata. The model only contributed the [E#] markers, which were validated.

def _evidence_header(e) -> str:
    mark = "✓" if e.citation_status == "cited" else "○"
    if e.is_community:
        return f"{mark} **[{e.evidence_id}]** Community report `{e.community_id}` — {e.title}"
    page = e.page_number if e.page_number is not None else "?"
    return f"{mark} **[{e.evidence_id}]** PDF: **{e.source}** · Page: **{page}** · Chunk: `{e.chunk_id}`"


def _evidence_meta(e) -> str:
    bits = []
    if e.distance is not None:
        bits.append(f"vector distance {e.distance:.2f} (lower = closer)")
    if e.graph_hop is not None:
        bits.append("graph: mentions a matched entity" if e.graph_hop == 0 else f"graph: {e.graph_hop} hop(s) from a matched entity")
    if e.is_community:
        bits.append("community-level summary")
    return " · ".join(bits)


def _other_locations(e) -> str:
    if not e.other_locations:
        return ""
    return "Also appears in: " + "; ".join(
        f"{l['source']} p.{l['page_number']}" if l.get("page_number") is not None else str(l["source"])
        for l in e.other_locations)


def render_answer_result(res, kg=None):
    with st.container(border=True):
        st.markdown(f"**Answer** · _{res.mode}_")
        st.write(res.answer)
        if res.unknown_ids:
            st.warning("The model cited evidence IDs that do not exist (" + ", ".join(res.unknown_ids) +
                       "). They were removed from the answer.")
    if res.insufficient and res.evidence:
        st.info("The retrieved evidence was judged insufficient to answer this question (it is listed below "
                "only so you can see what was searched).")
    for note in res.notes:
        st.caption(note)

    if res.evidence:
        st.markdown("**Sources**  \n<span style='font-size:0.8rem;color:#6B7280'>✓ cited in the answer · ○ retrieved but not cited</span>",
                    unsafe_allow_html=True)
        for e in sorted(res.evidence, key=lambda x: x.citation_status != "cited"):
            st.markdown(_evidence_header(e))
            meta = _evidence_meta(e)
            if meta:
                st.caption(meta)
            if _other_locations(e):
                st.caption("↳ " + _other_locations(e))

        with st.expander(f"Source Evidence ({len(res.evidence)} item(s))"):
            for e in res.evidence:
                st.markdown(_evidence_header(e))
                if e.is_community:
                    st.text(e.text)
                    locs = "; ".join(f"{l['source']} p.{l['page_number']}" for l in e.provenance[:15])
                    st.caption(f"Built from: {locs}" + (" …" if len(e.provenance) > 15 else ""))
                else:
                    st.text(e.text[:1200] + ("..." if len(e.text) > 1200 else ""))
                    if e.graph_facts:
                        st.caption("Graph facts from this text: " + " | ".join(e.graph_facts))
                st.markdown("---")

    if res.matched:
        with st.expander(f"Matched Graph Entities ({len(res.matched)})"):
            for m in res.matched:
                extra = f" ({m.matched_text})" if m.match_type == "semantic" else ""
                st.markdown(f"• **{m.display_name}** — {m.match_type}{extra}")
        if all(m.match_type in ("fuzzy", "semantic") for m in res.matched):
            st.info("These entities were matched by approximate similarity (fuzzy / embedding) only, "
                    "so double-check they are what you meant.")
    if res.facts:
        with st.expander(f"Retrieved Graph Facts ({len(res.facts)})"):
            for f in res.facts:
                st.markdown(f"**[hop {f.hop}]** {f.subject} -[{f.predicate}]-> {f.object}")
                for p in f.provenance[:search.MAX_PROVENANCE_PER_FACT]:
                    page = p["page"] if p["page"] is not None else "?"
                    desc = f"*{p['description']}*  \n" if p.get("description") else ""
                    st.markdown(f"{desc}*Evidence:* \"{p['evidence']}\"  \n*Source:* {p['source']} · *Page:* {page} · "
                                f"*Chunk:* `{p['chunk_id']}`")
                if len(f.provenance) > search.MAX_PROVENANCE_PER_FACT:
                    st.caption(f"Showing {search.MAX_PROVENANCE_PER_FACT} of {len(f.provenance)} supporting passages.")
                st.markdown("---")
        if kg is not None:
            with st.expander("Retrieved Subgraph (visual)"):
                sub_html, sub_h = render_graph_svg(kg, res.matched, res.facts, height=380)
                if sub_html:
                    show_html(sub_html, sub_h)
                st.caption("Red outline = matched entity · indigo = hop 1 · orange = hop 2 · arrows follow the fact's direction.")
    if res.findings:
        with st.expander(f"Intermediate findings from the map stage ({len(res.findings)})"):
            for p in res.findings:
                st.markdown(f"**{p['score']}/100** — {p['description']}  \n"
                            f"<span style='font-size:0.8rem;color:#6B7280'>from {', '.join('[' + i + ']' for i in p['evidence_ids'])}</span>",
                            unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Question flows
# --------------------------------------------------------------------------- #

def show_error(mode: str, e: Exception):
    low = str(e).lower()
    hint = ""
    if "huggingface" in low or "sentence" in low or "winerror 10060" in low or "connection" in low and "hub" in low:
        hint = (" The local embedding model could not be loaded. If it was never downloaded, connect to the internet once "
                "(huggingface.co) or point LOCAL_EMBEDDING_MODEL to a downloaded model folder; if it is already "
                "downloaded, set HF_HUB_OFFLINE=1.")
    elif "429" in low or "quota" in low or "rate" in low or "resource" in low:
        hint = " This looks like a Gemini rate limit / quota problem — wait a moment and try again."
    elif "api key" in low or "api_key" in low or "permission" in low or "401" in low or "403" in low:
        hint = " This looks like an API key problem — check GOOGLE_API_KEY."
    st.error(f"{mode} failed: {type(e).__name__}: {e}.{hint}")


def process_question(question, api_key, model_name, top_k, max_per_pdf, sources, enable_judge):
    """Traditional RAG: ChromaDB similarity search -> evidence -> Gemini."""
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner(f"Retrieving relevant chunks and generating an answer with {model_name}..."):
                res = search.basic_search(get_store(api_key), question, search.chat_model(model_name, api_key),
                                          top_k=top_k, max_per_pdf=max_per_pdf, sources=sources)
        except Exception as e:
            traceback.print_exc()
            show_error("Traditional RAG", e)
            return
        evaluation = None
        if enable_judge and res.evidence and not res.insufficient:
            try:
                with st.spinner("Running LLM Evaluation..."):
                    evaluation = judge_answer(question, res.context, res.answer, model_name, api_key)
            except Exception as e:
                st.warning(f"LLM Evaluation failed ({type(e).__name__}); the answer itself is unaffected.")
        render_answer_result(res)
        render_evaluation(evaluation)
    st.session_state["chat_history"].append({"question": question, "result": res, "evaluation": evaluation})


def process_graph_question(question, mode, api_key, model_name, top_k, max_per_pdf, sources):
    kg = st.session_state["knowledge_graph"]
    reports = st.session_state["community_reports"]
    entity_store = st.session_state["entity_store"]
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner(f"{mode}: retrieving evidence and generating an answer with {model_name}..."):
                model = search.chat_model(model_name, api_key)
                if mode == "Local Search":
                    res = search.local_search(kg, question, model, entity_store=entity_store, reports=reports, sources=sources)
                elif mode == "Hybrid RAG":
                    res = search.hybrid_search(get_store(api_key), kg, question, model, top_k=top_k, max_per_pdf=max_per_pdf,
                                               sources=sources, entity_store=entity_store, reports=reports)
                else:
                    res = search.global_search(reports, question, model)
        except Exception as e:
            show_error(mode, e)
            return
        render_answer_result(res, kg)
    st.session_state["search_history"].append(res)


def graph_suggestions(kg, limit: int = 4) -> list:
    """Starter questions built from the graph itself, so each one names entities that exist in it."""
    g = kg.graph
    out: list = []
    for name, _deg in graph_stats(kg)["top_entities_by_degree"][:2]:
        out.append(f"What is {name}?")
    for u, v, _data in sorted(g.edges(data=True), key=lambda e: (e[0], e[1], e[2]["predicate"])):
        q = f"How is {g.nodes[u]['name']} related to {g.nodes[v]['name']}?"
        if q not in out:
            out.append(q)
        if len(out) >= limit:
            break
    return out[:limit]


MODE_CAPTIONS = {
    "Local Search": "Microsoft GraphRAG-inspired **Local Search**: entities in your question → their graph neighbourhood → "
                    "the text chunks behind those facts (+ community reports). Best for specific questions about named things.",
    "Global Search": "Microsoft GraphRAG-inspired **Global Search**: a map-reduce over community reports. "
                     "Best for questions about the corpus as a whole (themes, common approaches).",
    "Hybrid RAG": "**Hybrid RAG**: ChromaDB vector evidence + knowledge-graph evidence, merged and de-duplicated.",
}
GLOBAL_SUGGESTIONS = [
    "What are the major themes across all uploaded documents?",
    "What are the common approaches discussed across the documents?",
]


def render_graph_mode_qa(mode, kg, api_key, model_name, top_k, max_per_pdf, sources):
    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-block"><h3>💬 Ask questions ({mode})</h3></div>', unsafe_allow_html=True)
    st.caption(MODE_CAPTIONS[mode])
    if mode == "Global Search" and sources:
        st.caption("The PDF filter does not apply to Global Search (it reads community reports of the whole corpus).")

    suggestions = GLOBAL_SUGGESTIONS if mode == "Global Search" else graph_suggestions(kg)
    if suggestions:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"graph_suggest_{mode}_{i}", use_container_width=True):
                st.session_state["graph_pending"] = s
    if st.session_state["search_history"] and st.button("🧹 Clear conversation", key="graph_clear_chat"):
        st.session_state["search_history"] = []
        st.rerun()

    st.write("")
    for res in [r for r in st.session_state["search_history"] if r.mode == mode]:
        with st.chat_message("user"):
            st.write(res.question)
        with st.chat_message("assistant"):
            render_answer_result(res, kg)

    typed = st.chat_input(f"Ask a question ({mode})...")
    question = st.session_state.pop("graph_pending", None) or typed
    if question:
        if not api_key:
            st.error("Please provide a Gemini API key in the sidebar first.")
            st.stop()
        process_graph_question(question.strip(), mode, api_key, model_name, top_k, max_per_pdf, sources)


def main():
    inject_css()
    ensure_state()

    # --- Header with title + a clear, always-visible Clear button ---
    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.markdown(
            """<div class="app-header-left"><span class="icon">📚</span>
            <div><h1>RAG Document Q&A</h1>
            <div class="app-subtitle">Upload up to 10 PDFs, ask questions, and see exactly which PDF, page and chunk each answer came from.</div>
            </div></div>""",
            unsafe_allow_html=True,
        )
    with header_right:
        st.write("")
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear all", use_container_width=True, help="Remove all indexed documents (and their ChromaDB chunks) and chat history"):
            clear_everything()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            api_key = st.text_input("Gemini API key", type="password")
        if not st.session_state["restored"]:
            restore_documents(api_key)          # re-read chunks already in ChromaDB (local embeddings: no key needed)

        model_name = st.selectbox("Gemini model", GEMINI_MODEL_OPTIONS, index=0)

        st.markdown("---")
        st.subheader("Chunking")
        chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, step=100)
        st.caption("Larger chunks = fewer chunks = fewer Gemini extraction calls. But the local embedding model "
                   "(MiniLM) reads only ~1000 characters per chunk, so retrieval ignores the rest of a longer chunk.")
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 100, step=50)

        st.markdown("---")
        st.subheader("Retrieval")
        top_k = st.slider("Chunks to retrieve per question (k)", 2, 16, 6)
        max_per_pdf = st.slider("Max chunks from one PDF", 1, 6, 2)
        st.session_state["source_filter"] = [s for s in st.session_state.get("source_filter", [])
                                             if s in st.session_state["documents"]]
        picked = st.multiselect("Only search these PDFs (optional)", list(st.session_state["documents"]), key="source_filter")
        sources = picked or None
        st.caption("Search covers ALL indexed PDFs unless you pick some here. k and the per-PDF cap apply to "
                   "Traditional RAG and Hybrid RAG.")

        st.markdown("---")
        st.subheader("LLM Evaluation")
        enable_judge = st.checkbox("Enable LLM Evaluation", value=False)
        st.caption("A second, independent Gemini call scores each answer for groundedness and relevance "
                   "(Traditional RAG mode). Off by default: it doubles the Gemini calls per question.")

        if st.session_state["documents"]:
            st.markdown("---")
            st.subheader("📄 Indexed documents")
            for name in list(st.session_state["documents"].keys()):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{name}**  \n{len(st.session_state['documents'][name])} chunks")
                if c2.button("✕", key=f"remove_{name}", help=f"Remove {name} (only its chunks are deleted from ChromaDB)"):
                    handle_remove_doc(name, api_key)
                    st.rerun()

    # --- Upload / index area (PDF only) ---
    st.markdown('<div class="section-block"><h3>1. Add your documents</h3></div>', unsafe_allow_html=True)
    uploaded_pdfs = st.file_uploader(
        f"Upload up to {MAX_PDFS} PDFs — try related documents to see cross-document retrieval",
        type=["pdf"],
        accept_multiple_files=True,
    )
    selection_problem = selection_error(len(uploaded_pdfs)) if uploaded_pdfs else None
    if selection_problem:
        st.error(selection_problem)
    st.write("")
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    index_clicked = st.button("📥 Index PDF(s)", key="index_pdf")
    st.markdown('</div>', unsafe_allow_html=True)

    if index_clicked:
        if not uploaded_pdfs:
            st.warning("Select at least one PDF first.")
        elif not selection_problem:         # indexing uses local embeddings: no Gemini key needed
            handle_index_pdfs(uploaded_pdfs, api_key, chunk_size, chunk_overlap)

    render_upload_results()

    docs = st.session_state["documents"]
    if docs:
        render_doc_chips(docs)
        stored = vs.count(st.session_state["vector_store"]) if st.session_state["vector_store"] is not None else 0
        st.caption(f"{len(docs)} / {MAX_PDFS} PDFs · {sum(len(c) for c in docs.values())} chunks · "
                   f"ChromaDB (persistent, ./.chroma_store) holds {stored} chunks")
    else:
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        st.info("👆 Upload one or more PDFs, then click Index to get started.")
        return

    doc_names = list(docs.keys())
    total_chunks = sum(len(c) for c in docs.values())

    # --- Retrieval Mode ---
    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    mode = st.radio("Retrieval Mode", list(MODES), horizontal=True, key="retrieval_mode")

    if mode in GRAPH_MODES:
        graph_ready = render_graph_index(api_key, model_name)
        if not graph_ready:
            if mode == "Hybrid RAG":
                st.warning("**Hybrid RAG requires the Knowledge Graph.** Build (or update) it with the button above — "
                           "this mode does not silently fall back to Traditional RAG.")
            return
        if mode == "Global Search" and not st.session_state["community_reports"]:
            st.warning("Global Search needs community reports, but none exist yet (communities need entities "
                       "connected by relationships). Add documents with more relationships, or use another mode.")
            return
        render_graph_mode_qa(mode, st.session_state["knowledge_graph"], api_key, model_name, top_k, max_per_pdf, sources)
        return

    # ----------------------- Traditional RAG ----------------------- #

    # --- Compare Documents: dedicated feature, only shown for 2+ documents ---
    if len(doc_names) >= 2:
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="compare-banner">
            <div class="title">🆚 Compare Documents</div>
            <div class="desc">{len(doc_names)} documents indexed — get a structured side-by-side comparison
            instead of asking one question at a time.</div></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        run_compare = st.button("🆚 Compare These Documents", use_container_width=True, key="run_compare")
        st.markdown('</div>', unsafe_allow_html=True)

        if run_compare:
            if not api_key:
                st.error("Please provide a Gemini API key in the sidebar first.")
                st.stop()
            try:
                with st.spinner(f"Comparing {len(doc_names)} documents with {model_name}..."):
                    st.session_state["comparison_result"] = generate_comparison(
                        get_store(api_key), doc_names, model_name, api_key
                    )
            except Exception as e:
                show_error("Compare Documents", e)

        if st.session_state["comparison_result"] is not None:
            st.write("")
            with st.container(border=True):
                render_comparison(st.session_state["comparison_result"])

    # --- Chat / Q&A ---
    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-block"><h3>💬 Ask a question</h3></div>', unsafe_allow_html=True)
    scope = f"{len(sources)} selected PDF(s)" if sources else f"all {len(doc_names)} PDF(s)"
    st.caption(f"Searching {scope}, {total_chunks} chunks indexed, retrieving top {top_k} per question "
               f"(max {max_per_pdf} per PDF).")
    st.write("")

    suggestions = get_suggested_questions(doc_names)
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"suggest_{i}", use_container_width=True):
            st.session_state["pending_question"] = s

    st.write("")
    st.write("")

    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_answer_result(turn["result"])
            render_evaluation(turn.get("evaluation"))

    # --------------------------------------------------------------------------- #
    # LLM Cache Status
    # --------------------------------------------------------------------------- #

    with st.expander("LLM Cache Status", expanded=True):
        cache_info = llm_cache.get_last_cache_status()
        cache_status = cache_info.get("status", "NOT_CHECKED")

        status_messages = {
            "HIT": ("success", "CACHE HIT — Cached response reused"),
            "MISS": ("warning", "CACHE MISS — No cached response found"),
            "SAVED": ("info", "CACHE SAVED — Response stored successfully"),
            "CLEARED": ("info", "Cache cleared"),
            "ERROR": ("error", "Cache error — Check the terminal"),
        }

        message = status_messages.get(cache_status)

        if message:
            display_type, display_message = message
            getattr(st, display_type)(display_message)
        else:
            st.caption("Cache status: Not checked yet")

        st.write(
            "Cached entries:",
            llm_cache.get_cache_stats()["cached_responses"],
        )

        if cache_info.get("timestamp"):
            st.caption(f"Last cache event: {cache_info['timestamp']}")

    typed_question = st.chat_input("Ask a question about the document(s)...")
    question = st.session_state.pop("pending_question", None) or typed_question

    if question:
        if not api_key:
            st.error("Please provide a Gemini API key in the sidebar first.")
            st.stop()
        process_question(question, api_key, model_name, top_k, max_per_pdf, sources, enable_judge)


if __name__ == "__main__":
    main()