# Module 08 — Session Corpus Upload and Organizer Benchmark

**STATUS: BUILD.** No code is shipped. Build to this spec.

## 1. Purpose

Two session-scoped features on top of the runtime:

1. **Corpus upload** (`prototype/upload.py`): an organizer replaces the bundled
   corpus for the current WebSocket session only with `.md`, `.txt`, or corpus
   `.json` files.
2. **Organizer benchmark** (`prototype/benchmark.py`): a labeled question set
   is run against the uploaded session corpus and scored.

Both are isolated: uploads live only in the session, disappear on disconnect,
and never affect the bundled corpus or other sessions.

## 2. `parse_upload(files) -> tuple[CorpusChunk]`

Input: list of `{"name": str, "text": str}`. Output: `CorpusChunk` records, or
raise `ValueError` with a human-readable reason. Enforce all of:

- 1–25 files; every file needs `name` + `text`; basename via
  `PurePosixPath(name.replace("\\","/")).name`; strip a UTF-8 BOM and
  normalize CRLF → LF; per-document size ≤ 250 000 chars.
- **`.json`** — a list of chunk records; each has non-empty string
  `doc_id`, `section`, `text` and optional string-only `metadata`. Reject any
  other shape (single object, training records with `tokens`/`labels`). Derive
  `chunk_id = "<doc_id> §<section>"` (never trust a supplied `chunk_id`).
  Offsets stamped `offset_basis: "uploaded JSON chunk text"`.
- **`.md` / `.txt`** — `doc_id` = filename stem; every `## heading` is a
  section (no headings → one `Content.1`); body becomes the chunk text with
  `start`/`end` offsets computed like Module 02 ingest; `entity` =
  `doc_id.replace("_"," ")`.
- **Anything else** → error listing supported formats (PDF requires prior
  conversion).
- Totals: 1–250 chunks, unique `chunk_id`s, section ≤ 16 000 chars,
  `doc_id`/`section` ≤ 150 chars.

Every chunk records `metadata.source` (file name) and `metadata.source_hash`
(SHA-256 of the whole normalized file).

## 3. `benchmark.py`

### `validate_cases(raw)`
Must be a list of 1–50 objects, each with a non-blank `question` (≤ 4000) and
optional `expected_citations` (list of strings), `unsupported` (bool),
`intent_count` (int ≥ 1), `id`. Contract:
`schemas/organizer-benchmark.schema.json`.

### `async run_benchmark(runtime, raw_cases)`
For each case, open a fresh `LiveConnection` against the session runtime, send
the question as a final `TranscriptMessage` (via `make_transcript_message`),
then evaluate the update:

- `citation_match` = `set(citations) == set(expected_citations)`.
- `abstention_match` = for `unsupported` cases: no citations and a non-empty
  `uncertainty`.
- `intent_match` = `len(sub_queries) == intent_count` when declared.
- Aggregates `supported` (claim text equals an evidence text AND every cited id
  is a real corpus id in the claim's `candidate_ids`) and `fabricated`
  (citation ids not in the corpus).
- Returns
  `{type:"benchmark_result", scope:"uploaded session corpus", passed, total,
  citation_support:{supported, emitted}, fabricated_ids, trace_events,
  elapsed_ms, cases:[{id, question, pass, expected_citations,
  actual_citations, intent_count, uncertainty}], disclosure:
  "Organizer-supplied labels evaluated locally against this session corpus;
  not a certified benchmark."}`.

## 4. Interfaces

- Consumed by Module 07 inside `/ws/stream` (`corpus_upload`,
  `benchmark_run`). The new runtime reuses the shared corpus/decomposer build
  path with `parser="spacy"`.
- Provides `corpus_loaded`, `benchmark_result`, `error` to Module 09.
- Depends on `citefrontier.models.CorpusChunk`, `prototype.server_types`
  (shared constructor), Modules 02/06.

## 5. Limits

1 MB wire payload; 16 000 chars per message; 250 sections; 25 files.
Frontend-side: combined size ≤ 900 KB; `JSON.stringify` ≤ 1 MB.

## 6. Acceptance criteria

- `.md` provenance: `Orion_X1.md` with `## Warranty.1` →
  chunk `Orion_X1 §Warranty.1`, `entity: "Orion X1"`, and
  `text == source[start:end]` (offsets exact).
- Reject training data (`[{"tokens":[...],"labels":[...]}]`), duplicate
  document IDs, empty uploads, and `.pdf` files.
- A guild forced `chunk_id: "FAKE"` in uploaded corpus JSON is ignored
  (`chunk_id` derived as `D §S`).
- Uploads replace only that session; another simultaneous session keeps the
  bundled corpus; invalid uploads preserve the active corpus; uploaded corpus
  access requires the new session token; disconnect → 404.
- A labeled benchmark reports per-case PASS/FAIL, supported==emitted for
  valid data, and `fabricated_ids == 0` in the acceptance set.