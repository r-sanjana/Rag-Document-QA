# RAG Document Q&A — multi-PDF Traditional RAG + Microsoft-GraphRAG-inspired search

A Streamlit app that answers questions over **up to 10 PDFs** and always shows **which PDF, which page and which chunk** an answer came from.

> **Terminology.** This project is *inspired by the concepts* of [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
> (text units → entities/relationships → knowledge graph → hierarchical Leiden communities → community reports → Local / Global search).
> It is **not** the `graphrag` package and **not** a drop-in equivalent. The tables below say exactly which parts follow
> Microsoft's design, which are simplified, and which are missing. Where this README says
> *"Microsoft GraphRAG-inspired implementation"* it means a re-implementation of the concept on this project's stack.

## 1. Objective

Take a corpus of PDFs (three of which may overlap in content), index it three ways (vector chunks, knowledge graph, community
reports) and answer questions with four retrieval modes, with every claim traceable to real source metadata.

Stack: Python · Streamlit · LangChain · Google Gemini (extraction, reports, answers) · **local sentence-transformers embeddings** · **ChromaDB** · **NetworkX** · Pydantic · pypdf.
Extra dependencies: `graspologic-native` (Leiden, see §8) and `sentence-transformers` (local embeddings, see §5a).

## 2. Architecture

```
                           up to 10 PDFs
                                 │
                     ingestion.py  (validate · hash · parse · chunk)
                                 │   chunk = text + {source, page, page_number, chunk_id, content_hash}
              ┌──────────────────┴───────────────────┐
              ▼                                      ▼
   vector_store.py → ChromaDB                graph_rag.py → Gemini extraction (Pydantic)
   (id = chunk_id, text, local embedding,         │  validated entities / relationships / evidence
    metadata)  ── persistent, incremental          ▼
              │                              NetworkX MultiDiGraph (provenance on every node & edge)
              │                                    │  hierarchical Leiden (graspologic-native)
              │                                    ▼
              │                              communities ─► Gemini community reports
              │                                    │
              └──────────────┬─────────────────────┘
                             ▼
                        user question
        ┌──────────────┬───────────┴───┬────────────────┐
        ▼              ▼               ▼                ▼
  Traditional RAG   Local Search   Global Search    Hybrid RAG        (search.py)
   ChromaDB         graph →text     community       vector + graph
   chunks           units+reports   reports map/    evidence merged
        │              │            reduce               │
        └──────────────┴───────┬────────┴────────────────┘
                               ▼
         evidence.py: merge · near-duplicate collapse · rank · limit · [E1],[E2]…
                               ▼
                             Gemini
                               ▼
         answer with [E#] ─► evidence.map_citations() validates every ID
                               ▼
              Answer · Sources (PDF, page, chunk) · Evidence · ✓ cited / ○ not cited
```

| File | Role |
|---|---|
| `app.py` | Streamlit UI only: upload, mode switch, graph build, shared result renderer, Compare Documents, LLM judge |
| `ingestion.py` | up-to-10 rule, content hashing / duplicates, per-PDF parsing, chunk metadata |
| `vector_store.py` | **all ChromaDB code**: incremental add/delete, global retrieval with distances, entity-embedding index |
| `graph_rag.py` | extraction + validation (Pydantic), extraction cache, NetworkX graph, communities, community reports, local graph retrieval, SVG visualisation |
| `search.py` | the four retrieval modes + grounded answer generation |
| `evidence.py` | shared `RetrievedEvidence` model, dedup, ranking, `[E#]` context, citation checking (no Streamlit) |
| `tests/` | 50 pytest tests (see §14) |

## 3. Multi-PDF ingestion

* **Maximum 10 PDFs** (`MAX_PDFS` in `ingestion.py`). Selecting 11 shows an error and indexes **nothing**; the first 10 are never processed silently. The cap also applies to the total indexed (a 9-PDF index accepts only 1 more).
* Every PDF is processed **independently**: a corrupt PDF → ❌ for that file only; a PDF with no extractable text (empty / scanned / image-only) → ⚠️ "not indexed" (it is not counted as indexed); a failure while embedding one PDF rolls back that PDF's partial chunks and the others stay.
* **Duplicates are detected by SHA-256 of the file content, never by filename:**
  same name + same content → duplicate · different name + same content → duplicate · same name + different content → indexed separately as `name [ab12cd].pdf` (nothing is overwritten).

## 4. Chunks, metadata and IDs

Every chunk carries:

| field | meaning |
|---|---|
| `source` | PDF name shown in the UI |
| `page` | **0-based** page index (kept because earlier code used it) |
| `page_number` | **1-based** page = `page + 1`. **This is what the UI shows.** |
| `chunk_id` | deterministic: `graph_rag.make_chunk_id()` = source + page + hash(text) |
| `content_hash` | SHA-256 of the whole PDF file |

`chunk_id` is the **same value** as the ChromaDB id and the knowledge-graph's chunk key (a test asserts the two ID sets are equal), so any graph fact can be traced back to the exact chunk.
Page numbers are the physical PDF page (what a viewer shows), which can differ from a printed page label inside the document.

## 5. ChromaDB (the vector database)

* Persistent folder `./.chroma_store`, collection `rag_documents_local` (cosine distance).
* Stored per chunk: **id** (`chunk_id`), **document** (chunk text), **embedding** (local model, §5a), **metadata** (above).
* **Incremental indexing:** adding a PDF embeds only chunk IDs that are not already in Chroma; removing a PDF deletes only its chunks; nothing else is re-embedded (tested with an embedding counter).
* **Persistence:** after a restart the app re-reads the chunks from Chroma, so the PDFs reappear without re-uploading or re-embedding.
* A second collection `graph_entities_local` (same folder) holds one embedding per knowledge-graph entity, used by Local Search (§10). It is rebuilt when the graph is built.
* NetworkX is **not** a database; it only holds the graph in memory (the extraction and report caches are small JSON files, see §7).
* The index is shared by everyone using the same working directory (single-user demo design).

### 5a. Local embeddings (no Gemini quota) and the Hugging Face download

Embeddings are computed on your computer with `sentence-transformers/all-MiniLM-L6-v2` (`vector_store.LocalEmbeddings`). Indexing a PDF and every vector search therefore cost **zero Gemini calls**. Gemini is only used for graph extraction (one call per new chunk), community reports (one per new community) and answers.

* **Loading is lazy and offline-first.** The model is loaded the first time text is embedded (not when the app only re-reads stored chunks), and if it is already in the Hugging Face cache it loads with `local_files_only=True` — *no network requests*. A plain load pings `huggingface.co` on every start; on a network that blocks it that shows up as `WinError 10060 … Retrying in 1s [Retry 1/5]` and makes start-up very slow. The model is shared by all browser sessions of the process.
* **First run needs one download** (~90 MB) from huggingface.co. If your network blocks it: download the model files on another connection (or machine) and set `LOCAL_EMBEDDING_MODEL` in `.env` to that folder, e.g. `LOCAL_EMBEDDING_MODEL=C:\models\all-MiniLM-L6-v2`. Once it is cached you can also set `HF_HUB_OFFLINE=1`.
* **Collections are per model.** Vectors from different models have different sizes, so the local index lives in `rag_documents_local` / `graph_entities_local`. An older index built with Gemini embeddings is left untouched and is simply not used; re-upload your PDFs to index them locally.
* **MiniLM reads only ~256 tokens (~1000 characters) per chunk.** Keep chunk size ≈ 1000 or less, otherwise the tail of a long chunk does not influence retrieval (the full text is still sent to Gemini). It is an English-centred model of modest quality; retrieval quality will differ from Gemini embeddings. `ENTITY_MATCH_MAX_DISTANCE` (§10) was never calibrated and distances differ per model — re-tune it.

### 5b. Reducing Gemini usage

| Gemini call | When | How to reduce |
|---|---|---|
| graph extraction | 1 per **new** chunk (cached by `chunk_id`, also across restarts) | bigger chunks (fewer chunks); use **Pause between extraction calls = 4 s** on the free tier (≈15 requests/minute) to avoid 429 errors; extraction is only repeated for chunks that failed |
| community reports | 1 per new/changed community (cached) | nothing needed after the first build |
| answers | 1 per question (Local/Hybrid/Traditional); Global = 1 per 5 reports + 1 | choose a mode that fits the question |
| LLM Evaluation | 1 extra per Traditional-RAG answer | **off by default** |
| embeddings | none (local) | — |

The Build button shows how many chunks still need extraction (≈ number of Gemini calls) before you click it.

## 6. Traditional RAG

`ChromaDB similarity search over ALL PDFs → evidence → Gemini`.
Global top-k (no forcing chunks from every PDF) with a cap of `max_per_pdf` chunks per PDF so one long document can't crowd out the others; optional "only search these PDFs" filter. Chroma's **distance** is displayed as *"distance 0.32 (lower = closer)"* — it is never called a confidence score.

## 7. Knowledge graph (NetworkX `MultiDiGraph`)

Gemini extracts per chunk (Pydantic structured output): entities (name, type, **description**) and relationships (subject, predicate, object, **description**, **evidence quote**). Validation (unchanged guards): entities/relationships must appear in the chunk, the evidence must be a quote from the chunk that mentions both ends, no negative or pronoun relationships, etc.

* **Provenance:** every node keeps its mentions `(source, page, chunk_id)`; every edge keeps a list of `{source, page, chunk_id, evidence, description}` — one record per supporting chunk. A relationship stated in three PDFs is **one edge with three provenance records**.
* **Entity merging is deliberately conservative:** only the same *normalized name* (case/punctuation-insensitive) merges across PDFs. "MLOps", "MLOps framework" and "Machine Learning Operations" stay **separate** entities (no aggressive alias merging). Local Search's entity-embedding step (§10) bridges some of those gaps at query time instead.
* Descriptions are **LLM-written summaries** of a chunk (length-capped, not verbatim); the **evidence quotes** are the verified part.
* **Extraction cache:** results are cached by `chunk_id` (in memory, and in `.graph_extraction_cache.json`). 500 cached + 40 new chunks → only 40 Gemini calls. Failed chunks are never cached, so **Build / Update** again retries exactly those.

## 8. Community detection  *(Microsoft concept: hierarchical Leiden)*

| | Microsoft GraphRAG | this project |
|---|---|---|
| algorithm | hierarchical Leiden (`graspologic`, `max_cluster_size`≈10) | **same algorithm**, via `graspologic-native` (`hierarchical_leiden`, `max_cluster_size=10`, same default seed) |
| input graph | entity graph | undirected weighted projection of the MultiDiGraph (weight = supporting records) |
| components | default `use_lcc=true` keeps only the largest connected component | **all** components are clustered (keeping only the largest would drop small documents' entities) |
| hierarchy | communities split recursively | same: communities larger than 10 are re-clustered (levels L0, L1, …) |

Why a new dependency: NetworkX's `leiden_communities` is only a dispatch stub (it needs an external backend), and no Leiden code ships in NetworkX itself. `graspologic-native` is the small Rust package Microsoft's implementation relies on — far lighter than installing all of `graspologic`. If it is missing, the app says so; it does **not** silently substitute another clustering algorithm.
Isolated entities (no relationships) belong to no community.
A graph whose structure is a star can legitimately yield a single level — Leiden only splits a community when that improves modularity.

## 9. Community reports  *(Microsoft concept: community reports)*

For every community with at least one relationship, Gemini writes a report with Microsoft's fields: **title, summary, rating (0–10) + explanation, findings**. It only sees that community's entities/relationships (with descriptions and one evidence quote each; capped at 40 entities / 60 relationships).
**Provenance is computed by code, not by Gemini:** each report stores its entities, relationships, the `chunk_id`s and the `(PDF, page)` list it rests on.
Reports are cached by a hash of their exact input context, so unchanged communities are never re-summarised (`.graph_report_cache.json`).
*Simplification:* each community is summarised directly from its own members; Microsoft substitutes sub-community reports when a parent's context is too large.

## 10. Local Search  *(Microsoft concept: Local Search)*

```
question → entities (name match; if none, ChromaDB entity-embedding match) → bounded 1–2 hop bidirectional
         neighbourhood → facts ranked & capped → linked text chunks → + community reports of the matched entities
         → evidence ([E#]) → Gemini
```
* Reuses the existing matcher (exact → substring → fuzzy) and traversal (bidirectional, bounded to 1–2 hops).
* **Semantic fallback:** if no entity name literally matches, entities whose embedding is close to the question are used. The distance threshold (`ENTITY_MATCH_MAX_DISTANCE = 0.35`, cosine) is an **uncalibrated starting value — tune it on your documents**.
* **Hub protection:** facts are ranked (fewest hops, most supporting chunks) and capped at 25; at most 3 supporting chunks per fact; at most 8 graph chunks; 2 community reports. Truncation is reported: *"Showing 1 of 12 retrieved graph facts."*
* Text chunks are ranked by how many matched entities/facts point to them. Hop `0` means "this chunk mentions a matched entity", `1/2` = chunk behind a fact that many hops away.
* Matches that are *only* fuzzy/semantic trigger a visible "double-check" notice. With **no** match, Gemini is **not called**: *"No relevant entities were found in the knowledge graph for this question."*

## 11. Global Search  *(Microsoft concept: Global Search)*

Not top-k retrieval: a **map-reduce over community reports**.
**MAP:** reports (level 0, largest first, max 20) are batched (5 per call); Gemini returns scored points (0–100) each citing report IDs. IDs outside the batch are dropped, score-0 points are discarded.
**REDUCE:** the highest-scoring points (max 15) go to Gemini, which writes the final answer from the *points only*. The UI shows the intermediate findings and the community reports used, each with the PDFs/pages it was built from.
*Simplifications:* reports aren't shuffled between batches, there is no dynamic community selection, the level is fixed to level 0 (coarsest), and report importance is the LLM's own rating.
Best for: *"What are the major themes across all documents?"* Poor for narrow factual questions (use Local/Hybrid/Traditional).

## 12. DRIFT Search — **not implemented** (on purpose)

Microsoft's DRIFT starts from the top community reports, produces an initial answer plus follow-up questions, then runs Local-Search-style searches for each follow-up and aggregates them. A faithful version needs follow-up generation, a priority queue of questions, iterative local runs and its own aggregation — many LLM-dependent steps I cannot validate without live Gemini access, and a "simplified DRIFT" would be a different feature under that name. There is no placeholder in the UI. Local + Global (+ Hybrid) cover the practical use cases here.

## 13. Hybrid RAG

`ChromaDB vector evidence + Local-Search graph evidence (+ community reports) → merge → de-duplicate → rank → limit (10) → Gemini`.
* A chunk found by **both** channels becomes one item (`vector+graph`) keeping its distance *and* its graph facts.
* Vector distance and graph hop are **never combined into one score**. Order: ① found by both (closest distance first) ② vector-only and graph-only chunks interleaved ③ community reports last.
* **Requires the knowledge graph.** If it isn't built (or is stale) the app says so and shows the Build button; it does **not** fall back to Traditional RAG. If the graph is built but no entity matches the question, the answer is vector-only and says so.

**Near-duplicate handling:** only chunks that are ≥ 90 % identical (word 3-shingle Jaccard) are collapsed for Gemini, and **every location is kept** ("Near-identical text also appears in: B.pdf p.7; C.pdf p.3" is both sent to Gemini and shown). Chunks that merely say similar things in different words — your three overlapping PDFs — are intentionally **not** merged.

## 14. Provenance and citations

1. Evidence items are built **from metadata** (`RetrievedEvidence`: text, source, page, page_number, chunk_id, retrieval type, distance *or* graph hop, graph facts, community id, other locations).
2. Before prompting, items get IDs `[E1]`, `[E2]`… and Gemini sees `Source / Page / Chunk ID / Text` blocks.
3. Gemini must cite with `[E#]` only and must not write PDF names or pages.
4. After the answer, `map_citations()` checks every `[E#]`: **unknown IDs are removed from the text and reported**; valid ones mark the evidence ✓ *cited*; the rest are ○ *retrieved but not cited*.
5. The UI shows PDF, page, chunk ID, evidence text, distance / graph hop — all from the metadata, never from the model.
6. If the evidence doesn't answer the question, the model is told to reply exactly: *"The available documents do not provide sufficient evidence to answer this question."* The UI then labels the retrieved items as "judged insufficient". Vector search always returns *something*, so this refusal is the model's job under the prompt — it is the part that needs the manual test below.

## 15. Install, configure, run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
(`sentence-transformers` pulls in PyTorch — a large download.) Create a `.env` file next to `app.py`:
```
GOOGLE_API_KEY=your_gemini_api_key
```
(or paste the key into the sidebar). Run:
```bash
streamlit run app.py
```
Indexing and restoring need **no** Gemini key; the key is used for graph extraction, community reports and answers. Requires Python 3.10+. Generated files: `.chroma_store/` (ChromaDB), `.graph_extraction_cache.json`, `.graph_report_cache.json` (delete to force re-extraction).
After a restart, PDFs reappear from ChromaDB; click **Build / Update Knowledge Graph** once — with warm caches this costs ≈ no Gemini calls (only the cheap entity-embedding step).

## 16. Testing

```bash
pip install pytest reportlab         # test-only dependencies (reportlab makes the synthetic PDFs)
python -m pytest tests -q
```
**58 tests, all *mechanical*:** synthetic PDFs (3 overlapping MLOps PDFs + 7 unrelated), a real ChromaDB in a temp folder, a hashed-bag-of-words **fake embedding model** (and a stand-in `sentence_transformers` module for the loader tests), a **fake extraction chain** and a **fake chat model**. They prove the plumbing: ingestion rules, metadata/page numbers (checked against a `PAGEMARK-n` token printed on each PDF page), incremental Chroma indexing, cross-PDF retrieval, graph merging and provenance, extraction/report caches, Leiden hierarchy, local/global/hybrid flows, hub limits, near-duplicate collapse, citation validation, refusal paths, and the real `app.py` driven through Streamlit's `AppTest` (every mode, stale graph, upload handler, Clear all).
They do **not** prove that real Gemini extracts good graphs, writes good reports, answers well, or refuses unrelated questions — see §17.

| Scenario | Test |
|---|---|
| 1 one PDF · 2 three PDFs · 3 ten PDFs · 4 eleven PDFs · 5 same content/other name · 6 same name/other content | `test_ingestion.py` (+ `test_upload_flow.py` through the real handler) |
| 7 cross-document retrieval | `test_vector.py`, `test_search.py::test_7_*` |
| 8 global search · 9 graph provenance/hybrid · 10 community detection · 11 reports | `test_search.py`, `test_graph.py` |
| 12 hybrid · 13 citation mapping · 14 unrelated question · 15 page verification | `test_search.py`, `test_evidence.py`, `test_ingestion.py::test_1_*` |

## 17. Manual tests to run with your real PDFs and Gemini key

1. Upload your 10 PDFs; compare each page number in *Sources* with the PDF viewer (physical page).
2. *"What is MLOps?"* in Traditional RAG: do the 3 overlapping PDFs appear (raise "Max chunks from one PDF" if one dominates)?
3. An **unrelated** question (e.g. "What is the capital of Mars?") in all four modes: expect the insufficient-evidence sentence / the no-entity message, no invented sources.
4. Build the graph; open *Community Reports* — are titles/summaries sensible and do "Built from" PDFs/pages make sense?
5. Local Search *"How is X related to Y?"*: open *Retrieved Graph Facts*, check evidence quotes and pages against the PDFs.
6. Global Search *"What are the major themes across all uploaded documents?"* and *"What are the common approaches discussed across the documents?"*.
7. Add an 11th PDF / a renamed copy / a corrupted file; remove a PDF and click Build / Update (expect 0 new extraction calls).
8. Restart the app: PDFs should reappear; Build / Update should be fast.
9. Tune `ENTITY_MATCH_MAX_DISTANCE` in `vector_store.py` if Local Search matches too many / too few entities by meaning.

## 18. Known limitations

* **Extraction quality** is the ceiling of everything graph-based. Sparse graphs (few relationships) mean few communities and thin Global Search. Predicate wording differs between PDFs ("supports" vs "enables") and is not unified.
* **No alias merging** ("MLOps" vs "Machine Learning Operations" stay separate nodes).
* **Cost/time:** one Gemini call per new chunk + one per new community (see §5b); free-tier rate limits can fail chunks (they are retried on the next Build / Update).
* **Entity names that are very common** can match many questions; fuzzy/semantic matches are flagged but can still be wrong.
* **Distance** values are cosine distances of the local embedding model and are only comparable within one query; no relevance cutoff is applied (none could be justified without calibration), so refusing irrelevant questions relies on the prompt.
* **Single shared index** per working directory; not multi-user safe.
* Page numbers are physical PDF pages; scanned PDFs are not OCR'd.
* Graph and reports are in memory (rebuilt from caches after restart); the graph visualisation is static (hover only).
* **Microsoft source check:** the architecture was mapped from Microsoft's published documentation (indexing flow, Local/Global/DRIFT search, Leiden settings) and write-ups of its implementation; I did not audit the repository code line by line, so behaviours described as "Microsoft's" here are as documented, not independently verified.