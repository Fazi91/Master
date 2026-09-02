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
2. Chunking: `scripts/chunk_text.py`, `scripts/build_chunks_dataset.py`,
   `scripts/build_page_dataset.py`, `scripts/compare_chunks.py`
3. Graph building: `scripts/build_core_relations.py`, `scripts/build_rel_text_text.py`,
   `scripts/build_semantic_graph.py`, `scripts/build_table_exclusion_mask.py`,
   `scripts/build_raster_inventory.py`, `scripts/build_semantic_to_neo4j_mapping.py`
4. Loading into Neo4j: `scripts/load_graph_to_neo4j.py`,
   `scripts/load_graph_v2_to_neo4j.py`, `scripts/clear_graph.py`,
   `scripts/export_graph_csvs.py`, `load_text_image_rels.py`
5. Serving: `scripts/graph_client.py`, `scripts/retriever.py`,
   `scripts/text_index.py`, `scripts/set_text_sim_weights.py`,
   `scripts/rebuild_text_index_from_nodes_csv.py`, `scripts/answer_synth.py`,
   `webapp/main.py` (FastAPI backend), `webapp/index.html` (frontend)

## Tech stack

- Python 3, FastAPI + Uvicorn for the web API
- Neo4j (via the `neo4j` driver) for the knowledge graph
- sentence-transformers (bi-encoder + cross-encoder reranker) and a local
  Hugging Face causal LM (`transformers`) for retrieval + generation
- FAISS for dense vector search, scikit-learn (TF-IDF) for lexical retrieval
- PyMuPDF / pdfminer.six / pypdf for PDF parsing

## Project structure

- `scripts/` - all pipeline stages (extraction -> chunking -> graph -> Neo4j -> retrieval)
- `webapp/` - FastAPI app (`main.py`) and static frontend (`index.html`)
- `data/` - `raw/`, `extracted/`, `processed/`, `graph/`, `graph_v2/`, `audit/`
- `outputs/` - generated embeddings, FAISS indices, jsonl chunk files (gitignored)
- `config.json` - embedding/index config
- `config.neo4j.json` - local Neo4j connection info (gitignored - contains a
  plaintext password, never commit this file or put its contents elsewhere)
- `.env` - runtime settings (model names, retrieval thresholds; see the
  `os.getenv(...)` calls at the top of `webapp/main.py` for the full list)
- `load_text_image_rels.py` - standalone helper at repo root (candidate to
  move into `scripts/` for consistency)

## Conventions & known issues to respect during cleanup

- Never hardcode corpus-specific content (questions, answers, PDF phrases) in
  `webapp/main.py` - it's meant to stay a generic plan/retrieve/rerank/compose/
  verify pipeline (see the module docstring).
- Secrets (Neo4j password, API keys) belong in `.env` or `config.neo4j.json`,
  both gitignored - never move them into tracked files.
- `requirements.txt` is currently saved as UTF-16 with CRLF line endings,
  which is unusual for a Python project and can break some tools (pip is
  tolerant, but linters/diff tools may not be) - worth re-saving as UTF-8 if
  you touch it.
- There is a stray empty folder named `CLAUD.md` (missing the "E") at the repo
  root, left over from an earlier mistake - safe to delete once this file
  exists.
- `webapp/main.py` and `scripts/build_semantic_graph.py` are large single
  files (60KB+); when cleaning up, prefer extracting cohesive pieces (e.g.
  retrieval, reranking, verification, prompt-building) into separate modules
  under `scripts/` or a new `webapp/` submodule rather than rewriting them
  wholesale in one pass.
- `*.log`, `__pycache__/`, `.venv/`, and most `outputs/` contents are
  gitignored - don't propose "cleanup" that just re-adds generated/log files
  to version control.

## Working with this repo

- Prefer exploring one script or module at a time and proposing a plan before
  editing, especially for `webapp/main.py` and `build_semantic_graph.py`.
- After any refactor, sanity-check by running the affected pipeline stage
  script directly, and/or starting the webapp (`uvicorn webapp.main:app`) and
  hitting it, since there is no automated test suite yet.
- Ask before deleting or overwriting anything under `data/` or `outputs/` -
  these can be expensive to regenerate.
