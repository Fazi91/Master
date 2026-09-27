# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

`llm-hallu-pipeline` builds an evidence-grounded QA system over a PDF (thesis/paper),
with hallucination detection. Pipeline: PDF -> extract text & images -> chunk ->
build a semantic knowledge graph -> load into Neo4j -> serve retrieval + rerank +
LLM answer generation + NLI-based verification through a small FastAPI webapp.

Rough flow:
1. Extraction: `scripts/extract_clean.py`, `scripts/extract_images.py`,
   `scripts/audit_pdf_images.py`, `scripts/audit_raster_candidates.py`,
   `scripts/classify_raster_images.py`, `scripts/review_raster_images.py`
2. Chunking: `scripts/build_page_dataset.py`, `scripts/build_chunks_dataset.py`
   (produces `data/graph_v2/chunks.csv`, the chunk set the live app reads)
3. Graph building: `scripts/build_core_relations.py`,
   `scripts/build_semantic_graph.py`, `scripts/build_table_exclusion_mask.py`,
   `scripts/build_raster_inventory.py`,
   `scripts/extract_semantic_relations.py` (typed relation extraction over the
   whole corpus, added after the initial graph build)
4. Loading into Neo4j: `scripts/load_graph_v2_to_neo4j.py`,
   `scripts/graph_client.py` + `scripts/clear_graph.py` (wipe/reset the graph)
5. Serving: `webapp/pdf_direct_qa.py` (retrieval + verify_unit + relevant_window +
   rephrase engine) and `webapp/evaluation_app.py` (FastAPI app, EvaluationService,
   GraphVerifier, benchmark, and the inline HTML/JS frontend it serves at `/`).
   These two files are the live, active app. (`webapp/main.py` and
   `webapp/index.html` were an earlier, unrelated webapp and have been removed;
   several other scripts/ files and root-level `config.json` that only fed a
   since-removed FAISS-based retrieval path have also been removed.)

## Tech stack

- Python 3, FastAPI + Uvicorn for the web API
- Neo4j (via the `neo4j` driver) for the knowledge graph
- sentence-transformers (bi-encoder + cross-encoder reranker) and a local
  Hugging Face causal LM (`transformers`) for retrieval + generation
- Brute-force numpy cosine similarity for semantic search (no FAISS - the
  corpus is small enough that an exact search is both simpler and fast
  enough; `faiss-cpu` was removed from `requirements.txt`), scikit-learn
  (TF-IDF) for lexical retrieval
- PyMuPDF / pdfminer.six / pypdf for PDF parsing

## Project structure

- `scripts/` - pipeline stages (extraction -> chunking -> graph -> Neo4j);
  none of these are imported by the live app - they're one-off CLI scripts
  run manually to (re)generate the `data/` files
- `webapp/` - `pdf_direct_qa.py` (extraction/retrieval engine) and
  `evaluation_app.py` (FastAPI app + inline frontend); this is the live app
- `data/` - `raw/`, `extracted/`, `processed/`, `graph_v2/` (the active
  graph dataset the live app and `load_graph_v2_to_neo4j.py` read)
- `outputs/` - generated/intermediate files; currently empty (its former
  contents fed a since-removed FAISS-based retrieval path) - don't assume a
  new file written here is read by anything without checking
- `config.neo4j.json` - local Neo4j connection info (gitignored - contains a
  plaintext password, never commit this file or put its contents elsewhere)
- `.env` - runtime settings (model names, retrieval thresholds; see the
  `os.getenv(...)` calls at the top of `webapp/pdf_direct_qa.py` and
  `webapp/evaluation_app.py` for the full list)

## Conventions & known issues to respect during cleanup

- Never hardcode corpus-specific content (questions, answers, PDF phrases) in
  `webapp/pdf_direct_qa.py` or `webapp/evaluation_app.py` - they're meant to stay
  a generic plan/retrieve/rerank/compose/verify pipeline.
- Secrets (Neo4j password, API keys) belong in `.env` or `config.neo4j.json`,
  both gitignored - never move them into tracked files.
- `requirements.txt` is currently saved as UTF-16 with CRLF line endings,
  which is unusual for a Python project and can break some tools (pip is
  tolerant, but linters/diff tools may not be) - worth re-saving as UTF-8 if
  you touch it.
- `webapp/pdf_direct_qa.py`, `webapp/evaluation_app.py`, and
  `scripts/build_semantic_graph.py` are large single files (60KB+); when
  cleaning up, prefer extracting cohesive pieces (e.g. retrieval, reranking,
  verification, prompt-building) into separate modules under `scripts/` or a
  new `webapp/` submodule rather than rewriting them wholesale in one pass.
- `*.log`, `__pycache__/`, `.venv/`, and most `outputs/` contents are
  gitignored - don't propose "cleanup" that just re-adds generated/log files
  to version control.

## Working with this repo

- Prefer exploring one script or module at a time and proposing a plan before
  editing, especially for `webapp/pdf_direct_qa.py`, `webapp/evaluation_app.py`,
  and `build_semantic_graph.py`.
- After any refactor, sanity-check by running the affected pipeline stage
  script directly, and/or starting the webapp
  (`uvicorn webapp.evaluation_app:app`) and hitting it, and/or running
  `tests/test_evaluation_30.py`.
- Ask before deleting or overwriting anything under `data/` or `outputs/` -
  these can be expensive to regenerate.
