# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A framework-free RAG service (FastAPI) that answers questions about uploaded PDF/DOCX/TXT documents with numbered citations. Every stage (chunking, hybrid retrieval, reranking, generation, citation validation, evals) is hand-written on purpose: do not introduce LangChain/LlamaIndex. The `langchain-*` packages in `requirements.txt` are transitive deps of the Voyage SDK, not used by this code.

The README (`readme.md`), prompts, user-facing strings and log messages are in Spanish (Rioplatense, "voseo"). Keep new user-facing text consistent with that.

## Commands

All commands must run from the repo root: the evals open `evals/dataset.json` relatively, and modules are run with `-m` because imports are package-relative to the root.

```bash
source venv/bin/activate
docker compose up -d                # Postgres 17 + pgvector on host port 5433 (needed by everything below except the pure unit tests)
python3 -m scripts.reindex          # (re)index samples/banfield*.txt into Postgres
uvicorn main:app --reload           # API: POST /documents (multipart file), POST /question
python3 -m pytest -v                # unit + integration tests (integration tests skip if Postgres is down)
python3 -m pytest tests/test_ranking.py::test_rrf_both_lists   # single test
python3 -m evals.run_eval           # retrieval eval: recall@4 + MRR (calls Voyage)
python3 -m evals.e2e_evals          # end-to-end answer/refuse eval (calls Claude, ~US$0.10/run, ~100 s)
```

`.env` needs `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`, plus `API_KEY` for the HTTP endpoints (clients send it in the `X-API-Key` header; if it is unset, `/documents` and `/question` answer 503 instead of staying open). Optional: `QUESTION_RATE_PER_MIN` (20), `DOCUMENT_RATE_PER_MIN` (5), `MAX_UPLOAD_MB` (10), `DB_POOL_MAX` (10), `LOG_LEVEL` (INFO). `DATABASE_URL` is optional and defaults to `postgresql://rag:rag@localhost:5433/rag` (the `docker-compose.yml` credentials). `repository/repository.py` builds the Voyage client **at import time**, so even the unit tests that import it need `VOYAGE_API_KEY` set. It does not connect to Postgres at import: connections come from a `psycopg_pool.ConnectionPool` created lazily by `get_pool()` (size `DB_POOL_MAX`, validated with `check_connection`) and closed at exit by `close_pool()`. Use `with get_pool().connection() as conn:` for new queries, never `psycopg.connect`.

`db/schema.sql` is mounted into `docker-entrypoint-initdb.d`, so it only runs when the `pgdata` volume is created. To re-apply it to an existing database (it is idempotent): `docker compose exec -T db psql -U rag -d rag < db/schema.sql`. There are no migrations.

`run_eval` measures whatever is currently indexed: after changing chunk size/overlap, run `scripts.reindex` first.

There is no linter or formatter configured.

## Architecture

Layers: `main.py` (HTTP only) → `service/` (domain logic, no HTTP knowledge) → `repository/repository.py` (Postgres/pgvector, Voyage embeddings, RRF, Voyage reranker). Services raise domain exceptions from `exceptions/exceptions.py`; `main.py` maps them to status codes (`UnsupportedFileType` → 415, `EmptyDocumentError` → 422). Sync service functions are called via `run_in_threadpool`.

**Access control** (`security.py`, wired in `main.py`): both endpoints depend on `rate_limit(limiter)`, which first checks the API key (`hmac.compare_digest`) and then applies an in-memory sliding-window `RateLimiter` keyed by that key (429 with `Retry-After`). A middleware answers 413 for bodies whose declared `Content-Length` exceeds `MAX_UPLOAD_MB` (+64 KiB of multipart overhead), and `/documents` also reads at most `MAX_UPLOAD_BYTES + 1` bytes. The limiter lives in one process's memory, so it does not work across replicas. `GET /health` is public (no key, no limit): it runs `SELECT 1` through the pool (`repository.ping`, via `service/health.py`) and answers 503 if the database is down; the Dockerfile `HEALTHCHECK` calls it. Logging is configured in `main.py` (`LOG_LEVEL`); use `logging.getLogger(__name__)` with Spanish messages instead of `print`, and never log question text. HTTP tests replace `main.answer` / `main.process_document` with fakes (`tests/test_security.py`).

**Storage** (`db/schema.sql`): `documents(id, filename UNIQUE, content_type, created_at)` and `chunks(id, document_id → documents ON DELETE CASCADE, chunk, content, n_tokens, metadata jsonb, embedding vector(1024), tsv)`. `tsv` is a generated `to_tsvector('spanish', content)` column. Indexes: HNSW (cosine) on `embedding`, GIN on `tsv`.

**Ingest** (`service/ingest.py`): parse by content type → fixed-size token chunks with `tiktoken` cl100k (500 tokens, overlap 100) → `add_chunks(filename, chunks, metadata, ids, content_type)`. `add_chunks` validates that chunks/metadata/ids have the same length, embeds the chunks with `embed_texts` (Voyage `voyage-4-lite`, batches of 128, **outside** the transaction), then in one transaction deletes the document with the same `filename` (its chunks cascade) and inserts the document and its chunks. Re-uploading therefore replaces the document, and a failed insert leaves the previous version intact. Chunk ids are `{filename}_chunk_{i}`; metadata is `{source, chunk}`.

**Query** (`service/query.py`, `answer()`):
1. Router: a short Claude Haiku call classifies `SMALLTALK` vs `SEARCH`. Off-topic questions deliberately go to `SEARCH` so they get refused rather than answered from model knowledge.
2. Retrieval (`default_retrieve`): `rerank_search` = `candidate_rows`, the deduplicated, unweighted union of `vector_rows` top-N (pgvector `<=>` cosine distance) and `lexical_rows` top-N (full-text), with `n_each=20` (up to 40 candidates) → Voyage `rerank-2.5` keeps top k. `candidate_rows` itself defaults to 10, but `rerank_search` passes 20. On any exception it falls back to `hybrid_search` (weighted RRF over the two branches, `ALPHA = 0.75`, k=60).
3. Generation: context is numbered `[1]..[n]`; Claude Haiku 4.5 is told to cite and to reply with exactly `NOT_FOUND` if the answer isn't in context.
4. Validation in code (don't rely on the prompt alone): if the reply starts with `NOT_FOUND` or contains no `[n]` citation, return the refusal with no sources; otherwise return only the cited chunks as `sources`.

The lexical branch (`lexical_rows`) builds its query with `websearch_to_tsquery('spanish', q)` and then replaces `&` with `|`. Do not remove that replacement: with the default AND semantics, question words such as "quién" or "cuántos" are not stop words and make almost every natural-language question match nothing. It ranks with `ts_rank_cd` (not BM25: no IDF, coarse scores) and breaks ties by `id` so evals are reproducible. Always filter with `tsv @@ query` against the stored `tsv` column; computing `to_tsvector(...)` inline would skip the GIN index.

`answer()` takes a `retrieve=` parameter so evals/tests can swap the retrieval strategy. `search`, `hybrid_search` and `rerank_search` return Chroma-shaped dicts (`{"documents": [[...]], "metadatas": [[...]], ...}`, one outer list per query; the shape was kept from the Chroma era); keep that shape when adding strategies, since `answer()` and `run_eval` index `[0]`. The lower-level `vector_rows` / `lexical_rows` return plain tuples `(id, content, metadata, score)`.

Design rationale worth knowing before changing retrieval (details in readme.md "Hallazgos"): weighted RRF can bury a chunk that only the lexical branch finds, which is why the reranker gets the unweighted union; a cosine-distance threshold does not separate answerable from unanswerable questions on this corpus, so refusal is handled by the prompt + citation rule; Voyage embeddings differ slightly between calls, so ranking differences below ~1e-3 are noise.

## Evals

`evals/dataset.json`: 36 questions over the Banfield corpus in `samples/`; 25 answerable (with a literal `must_contain` phrase that must appear in a retrieved chunk, compared accent- and case-insensitively) and 11 with `must_contain: null` (should be refused). With only 25 positives, one question = 0.04 recall, so small differences are noise.

Current numbers on Postgres: `run_eval` gives semantic 0.88/0.73, hybrid 0.75 0.92/0.79 and rerank 1.00/1.00 (recall@4 / MRR); `e2e_evals` gives 11/11 negatives refused and 0/25 positives wrongly refused. The 1.00 is likely generous: up to 40 candidates over 38 chunks means the reranker sees almost the whole corpus.

## Status

The Chroma → Postgres migration lives on branch `feat/pgvector-migration` and is not merged. `main` still has the Chroma + in-memory BM25 version (commit `ee88a27`), useful as a baseline. The local `chromadb/` directory is a leftover and no code reads it. Not done yet: a paraphrased validation set for the evals, measuring reranker latency/cost, trimming `requirements.txt` (still carries Chroma's transitive dependencies, ~700 MB image), and choosing a hosting platform and managed Postgres.
