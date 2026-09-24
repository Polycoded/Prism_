# CiteFrontier — Interfaces and Data Formats

This is the authoritative cross-module contract reference for the system to be
built. It defines every data structure that crosses a module boundary, the
exact JSON shapes, and the data flow between module pairs. Where a JSON Schema
file exists under `schemas/`, it is the authoritative contract and the text
here summarizes it.

## 1. Central data types

### 1.1 `CorpusChunk` (shipped — Module 02)

```json
{
  "chunk_id": "LumaPad_S1 §Warranty.1",
  "doc_id": "LumaPad_S1",
  "section": "Warranty.1",
  "text": "The LumaPad S1 has a 24-month limited warranty starting on the retail purchase date.",
  "metadata": {
    "entity": "LumaPad S1",
    "title": "LumaPad S1 support",
    "source": "LumaPad_S1.md",
    "source_hash": "20c11fef…",
    "start": "62",
    "end": "151"
  }
}
```

Defined in `citefrontier/models.py` (frozen dataclass). `chunk_id` is the ONLY
legal citation ID; format `<doc_id> §<section>`. Consumers: Module 02 indexes,
Module 05 claim layer, Module 09 source inspector, Module 08 upload.

### 1.2 `Intent` (produced by Module 01, consumed by 02/04/05/06)

Public record shape (`Intent.public()` = `asdict`):

```json
{
  "key": "ab12cd34…", "query": "what is the warranty period",
  "subject": "LumaPad S1", "entity_ids": ["LumaPad_S1"],
  "topic": ["period","warranty"], "constraints": [],
  "source_spans": [[62,86]], "method": "bert_intent_boundary"
}
```

Module 05 keeps an internal richer `(key, query)` minimal form for its own
bookkeeping; the full `Intent` shape above is the cross-module contract.

### 1.3 `TranscriptMessage` (inbound, Module 07 parses → Module 06 consumes)

Schema: `schemas/transcript-input.schema.json`. Required `event_id`, `turn_id`,
`text`, `timestamp_ms`; optional `is_final`, `revision_of`,
`target_intent_key`, `sequence`. `additionalProperties: false`.

### 1.4 `update` envelope (Module 06 → 07/09)

Schema: `schemas/live-output.schema.json`; full field map in
`module-10-schemas-telemetry.md`. The key contract points:

- `decision` ∈ {`wait`, `provisional_retrieve`, `commit_retrieve`, `suppress`}.
- `claims` — current committed claims (shape in `module-05`).
- `answer` — rendered answer string or `null`; bullet-grouped on presentation
  suppresses via `presentation_items`.
- `citations` — all cited `chunk_id`s in the current claims.
- `uncertainty` — claim texts without citations.
- `clarification` — a question to the user, or `null`.
- `retrieval_events` — telemetry for this event (Module 10).
- `stats` — `{searches, cache_reuses}`; `history` — last 10 version patches;
  `corpus_hash` — active corpus SHA-256.

### 1.5 Telemetry event

`{event_type, timestamp_s, server_ms, session_id, event_id, turn_id,
token_cost, payload}` — contract in `schemas/telemetry-event.schema.json` and
`module-10`. Produced by Module 06; served by Module 07.

## 2. Module-to-module data flow (target)

```
client (Module 09 / script)
   │  JSON TranscriptMessage  (schema 1.3)
   ▼
Module 07 server.py — validate, enforce caps, dispatch tasks
   │  TranscriptMessage
   ▼
Module 06 runtime — orchestration
   │  1 Module 03 transcript  → cleaned text + change records
   │  2 Module 04 controller  → decision
   │  3 Module 01 decomposer  → tuple[Intent] (+ diagnostic)
   │  4 Module 02 SearchIndex → candidate ids → finalize → Candidates
   │  5 Module 05 session     → claims, versioning, answer
   │
   ├─ update envelope ────────────────▶ Module 09 render
   └─ telemetry events ───────────────▶ GET /session/{id}/telemetry (07)

side paths:
   Module 08 upload  → chunks → new runtime for the session (06/02)
   Module 08 benchmark_run → cases → run over session runtime
```

## 3. Fixed interfaces of the shipped components

These are non-negotiable because Modules 01–02 are frozen:

### Module 02 → Module 06 (retriever contract)

- `SearchIndex(chunks, decomposer, backend, rerank)` with
  - `candidates(intent) -> tuple[chunk_id]` (recall; top 20; RRF constant 60);
  - `finalize(intent, candidate_ids) -> tuple[Candidate(evidence_id, score,
    relevance)]` (rerank + gates; top 5);
  - `.chunks[chunk_id]` returns the `CorpusChunk`.
- `HybridRetriever.search(query, retrieval_event_id, limit=5)` and
  `DenseHybridRerankRetriever.search(...)` in `citefrontier/retrieval.py`.

### Module 01 → Module 06 (decomposer contract)

- `decomposer.split(text) -> tuple[Intent]`;
- `decomposer.entities(text)`, `decomposer.topic(text, entities=())`;
- `decomposer.aliases` (dict `doc_id -> display name`);
- `decomposer.consume_diagnostic()` (BERT-shadow diagnostics);
- `decomposer.method` (string naming the active splitter).

## 4. File / wire formats

### 4.1 `prototype/corpus.json`
JSON array of `CorpusChunk`. Built by `python -m prototype.ingest` from
`prototype/corpus/sources/*.md`. Hashed by Module 06 to form `corpus_hash`.

### 4.2 `prototype/corpus-manifest.json`
`{schema_version, format, corpus_sha256, documents:[{file, sha256,
characters}], chunks, chunking}`.

### 4.3 `prototype/models-manifest.json`
`{models:[{repository, revision}], parser:"en_core_web_sm==3.8.0"}`. Written by
`python -m prototype.prepare_models`.

### 4.4 `prototype/demo/scenarios.json` (optional)
Replay fixtures `{key: {text, prefixes[], followup?, revision?}}`. Served only
when `CITEFRONTIER_DEMO=1`; retrieval never reads it.

### 4.5 `prototype/reports/scorecard.json` (optional)
If present, `GET /scorecard` returns it; the frontend renders it neutrally.
Absent → `{"status":"pending","official_validation":"pending"}`.

### 4.6 Organizer benchmark input
JSON array per `schemas/organizer-benchmark.schema.json`, sent in-band as
`{"type":"benchmark_run","cases":[…]}`.

## 5. Guarantees across every interface

1. A citation string is always a `chunk_id` of the active corpus.
2. A cited claim's `text` is always a normalized-substring exact copy of one
   committed chunk's `.text` (exact-extractive verification).
3. `source_spans` are valid offsets into the original utterance.
4. Every emitted telemetry event satisfies the full contract (Module 10 §5).
5. Modules 01–02 public contracts (`Intent` fields; `candidates`/`finalize`;
   RRF constant; top-k windows; vocabulary-gate allowlist) are frozen.