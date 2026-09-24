# Module 06 — Live Runtime (revision-safe orchestration)

**STATUS: BUILD.** No code is shipped. Build to this spec.

## 1. Purpose

The live runtime is the per-connection state machine that coordinates
Modules 01–05: it validates events, deduplicates, applies normalization,
asks the Module 04 controller for a decision, dispatches Module 02 retrieval,
uses Module 05 synthesis to build and version claims, and emits the Module 10
`update` envelope. Its defining guarantees: **no tentative candidate becomes a
claim**, and **no stale asynchronous result mutates an answer**.

## 2. Files to create

- `prototype/runtime.py` — `LiveRuntime` + `LiveConnection`.
- One `TranscriptMessage` Pydantic model (Module 07 owns the byte-level
  definition; this module consumes it).

## 3. `LiveRuntime.__init__(chunks=None, backend=None, parser=None,
reuse=True, rerank=True)`

1. Corpus: if `chunks` is None, read `CITEFRONTIER_CORPUS` (default
   `prototype/corpus.json`) and build `CorpusChunk`(s). Reject: empty corpus,
   duplicate `chunk_id`, missing `doc_id`/`section`/`text`, or mismatched
   `metadata.end - metadata.start != len(text)`.
2. `corpus_hash` = SHA-256 of `json.dumps(asdict(chunk), sort_keys=True)` over
   all chunks.
3. `backend` = arg or `CITEFRONTIER_BACKEND` (default `lightweight`).
4. Decomposer: for parser modes `bert|shadow|auto` build the Module 01
   `HybridDecomposer` (respecting `CITEFRONTIER_FALLBACK_PARSER`,
   `CITEFRONTIER_BERT_MODEL`, `CITEFRONTIER_BERT_THRESHOLD`); preload the
   detector when `CITEFRONTIER_BERT_PRELOAD != "0"` (swallow load errors).
   Otherwise `Decomposer(chunks, parser_mode)`.
5. `SearchIndex(chunks, decomposer, backend, rerank)` (Module 02).
6. `work_slots = asyncio.Semaphore(4)`.

## 4. Per-session state (`LiveConnection`)

- `session_id` (uuid4), `token` (`secrets.token_urlsafe(32)`).
- `lock` (asyncio.Lock) — serializes `handle`.
- `generation` — incremented on every event; monotonic staleness guard.
- `turn_id`; `events` (event_id → fingerprint/turn/sequence/owner);
  `receipts` (event_id → last result); `inflight` (event_id → Future).
- `cache` — tentative candidate sets keyed by
  `(corpus_hash, backend, intent.key)`.
- `claims` (list of claim dicts), `history` (version patches),
  `telemetry`, `intents` (key → Intent), `version` (answer version counter).
- `previous` / `previous_text` — last parsed intents/text for stability.
- `search_count`, `reuse_count`.
- `verifier = ExtractiveEntailmentVerifier()` (Module 05).

`record(kind, event, **payload)` builds a telemetry item
(`schemas/telemetry-event.schema.json`) with an explicit zero
`token_cost = {input_tokens:0, output_tokens:0, estimated_usd:0.0,
basis:"non_generative_pipeline"}`.

## 5. Event handling (`async handle(event)`)

### 5.1 Pre-lock normalization
`fingerprint = event.model_dump_json()`; run Module 03 `normalize_disfluencies`
then `normalize_correction(fluent, decomposer.aliases)`; if changed, re-copy
the event with `text=normalized`. Keep the raw text for telemetry.

### 5.2 Idempotency (under lock)
- Same `event_id` + different fingerprint → `ValueError("event_id reused with
  different content")`.
- Stored receipt → return it with `duplicate=True`, `retrieval_events=[]`.
- In-flight future → `await asyncio.shield(future)`; same `duplicate` marker.

### 5.3 Validation (under lock)
- Session closed → error; `len(events) >= 500` → error.
- Turn already finalized → error ("use a new turn_id").
- `revision_of` must reference an earlier event of the SAME turn.
- `sequence` must strictly increase.
- `target_intent_key` only on final events.

### 5.4 Decision preparation
- `is_format = presentation(text)`; `is_social = social(text)` (Module 01).
- For final non-format events: `updates, clarification = _refinement(text,
  target)`; force `clarification` when the subject correction was ambiguous;
  `translation_request` → clarification "Translation is not supported …".
- `additions = _late_detail_additions(text)` for final declarative follow-ups.
- Register the inflight future + event record; on turn change clear cache and
  previous state; `generation += 1`; `revision = generation`; mark the turn
  finalized.
- Record `transcript_normalized` if text changed.
- **Cache invalidation:** if `revision_of` is set, or the new text does not
  start with the previous text (lowercased), record `provisional_invalidated`
  and clear the cache.
- Parse: `tuple(updates.values())` if refining; else `additions`; else
  `decomposer.split(text)` unless format/social.
- Multi-product guard: if text mentions >1 entity and any intent has no entity
  but contains `it|its|they|their|them`, set the "more than one product"
  clarifier.
- Ask Module 04 for the decision (its inputs are computed here), record
  `controller_decision` + `intents_decomposed` (with
  `decomposer.consume_diagnostic()`).
- `wait`/`suppress` → build the snapshot immediately (no retrieval).
  Suppress with existing claims + the bullets grammar keeps the answer with
  bullet grouping (see §7).

### 5.5 Retrieval
`todo = (intents not already cached)` for non-final; all intents for final.
Run `asyncio.wait_for(asyncio.gather(*[_retrieve(...)]), timeout=30)`; on
timeout record `retrieval_timeout` and continue with what finished.

`_retrieve(event, intent, final)`:
- `retrieval_id = f"r-{event.event_id}-{intent.key[:8]}"`.
- Final + reuse enabled + cache hit → record `candidate_cache_reused`, bump
  `reuse_count`, use the cached ids.
- Else record `retrieval_started`; `ids = await asyncio.to_thread(
  retriever.candidates, intent)` under `work_slots`; record
  `retrieval_finished` (ids + phase).
- Final phase: `evidence = await asyncio.to_thread(retriever.finalize, intent,
  ids)`; record `final_evidence_ranked`.

### 5.6 Commit / stale handling (under lock again)
- Extend the trace from results.
- If `closed or revision != generation` → record `stale_result_discarded` and
  return a `wait` snapshot (late corrections reject old results).
- Non-final → store candidate ids in cache; snapshot
  `provisional_retrieve` (claims empty).
- Final → build claims:
  - `updates`: keep non-target claims; replace targets through the Module 05
    claim builder sharing the same `claim_id` with `claim_revision+1`.
  - `additions`: keep claims, append new.
  - else: new claims for every parsed intent.
- Update `intents`, `claims`, `version += 1`, append a `history` patch
  `{version, parent_version, added, removed, replaced, preserved}`; record
  `claim_verified` per changed claim + one `answer_version` event; clear the
  cache; return the answer snapshot.

## 6. Session lifecycle

`async close()`: set `closed`, bump generation, cancel inflight futures,
clear all state. The server (Module 07) calls close on disconnect.

## 7. Snapshot (`update` envelope)

`_snapshot(event, decision, reason, trace, intents=(), changed=(), answer=False,
clarification=None)` returns the envelope per `schemas/live-output.schema.json`
(`module-10`). Special rules:

- `answer` is `None` unless `answer=True`; bullet rendering uses
  `presentation_items` (see below).
- `uncertainty` = texts of claims with no citations.
- `stats = {searches, cache_reuses}`; `history = list(history[-10:])`.
- **Bullets:** only when `decision == "suppress"`, claims exist, and the text
  matches `\bbullets\b` — parse the requested count from
  `\b(one|two|three|four|five|\d+)\s+bullets\b` (default: all claims), group
  claims into that many contiguous groups, and emit
  `presentation_items = [{claim_ids, text, citations}]`. No new retrieval.

`_finish(result, event, started)`: fill `latency_ms`, append
`response_emitted`, store the receipt, resolve the inflight future.

## 8. Interfaces

- Called by Module 07 (`connection.handle(message)`).
- Uses Modules 01 (decomposer/classifiers), 02 (SearchIndex), 03
  (normalization), 04 (decision), 05 (claim builder, refinement, verifier).
- Emits the `update` envelope + telemetry events (Module 10).
- `bert_intent_tagger/scripts/06_compare_to_rule_based.py` also constructs
  `LiveRuntime(parser="rules")` once this module exists.

## 9. Acceptance criteria

Reproduce the behaviors in `docs/testing.md` exactly, including: provisional
cache revalidation before commit (`stats = {searches:1, cache_reuses:1}` for a
single-intent final after one tentative); duplicate idempotency; no re-emission
of old claims across turns; stale-result discard; unknown-topic abstention;
format suppression preserving claims, stats, and citations; targeted delta
(one extra search, unaffected claim unchanged); conflicting-quantity
withholding; parallel intent scheduling (all `retrieval_started` before the
first `retrieval_finished`); and single-commit for overlapping duplicates.