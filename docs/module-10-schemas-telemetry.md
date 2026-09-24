# Module 10 — Schemas and Telemetry

**STATUS: CONTRACT.** The JSON Schema files under `schemas/` are shipped and
are the authoritative contracts. Build the modules against them.

## 1. Files

| File | Contract for |
|---|---|
| `schemas/transcript-input.schema.json` | Inbound `TranscriptMessage` (Modules 07/06). |
| `schemas/live-output.schema.json` | The outbound `update` envelope (Module 06). |
| `schemas/telemetry-event.schema.json` | `retrieval_events[]` entries and the session telemetry log. |
| `schemas/organizer-benchmark.schema.json` | The `benchmark_run` case array (Module 08). |
| `schemas/replay-case.schema.json` | A local replay case shape used by `tests/` to define acceptance scenarios. |

## 2. `transcript-input.schema.json`

Required: `event_id`, `turn_id`, `text`, `timestamp_ms`. Optional:
`is_final` (false), `revision_of` (null), `target_intent_key` (null),
`sequence` (null). `additionalProperties: false`. Constraint summary: `text`
1–4000; ids ≤ 100/500; `timestamp_ms ≥ 0`; `sequence ≥ 0`.

## 3. `live-output.schema.json`

Required fields: `type`, `session_id`, `event_id`, `turn_id`, `decision`,
`reason`, `sub_queries`, `claims`, `answer`, `answer_version`,
`changed_claim_ids`, `citations`, `uncertainty`, `clarification`,
`tentative_count`, `retrieval_events`, `latency_ms`, `stats`, `history`,
`corpus_hash`, `presentation_items`. Meaning of each field:

| Field | Type | Meaning |
|---|---|---|
| `type` | string | `"update"` |
| `decision` | string | `wait` \| `provisional_retrieve` \| `commit_retrieve` \| `suppress` |
| `reason` | string | controller rationale |
| `sub_queries` | array | `Intent.public()` records (Module 01) |
| `claims` | array | current committed claims (Module 05 §7) |
| `answer` | string\|null | rendered answer; bullets-grouped on suppresses |
| `answer_version` | int | monotonically increasing per commit |
| `changed_claim_ids` | array | claim ids replaced/added this turn |
| `citations` | array | all cited chunk ids |
| `uncertainty` | array | texts of claims with no citation |
| `clarification` | string\|null | question back to the user |
| `tentative_count` | int | cached provisional candidates |
| `retrieval_events` | array | trace for this event |
| `latency_ms` | number | wall time of `handle()` |
| `stats` | object | `{searches, cache_reuses}` |
| `history` | array | last 10 version patches |
| `corpus_hash` | string | active corpus SHA-256 |
| `presentation_items` | array | bullet groups `{claim_ids, text, citations}` |

## 4. `telemetry-event.schema.json`

Required: `event_type`, `timestamp_s`, `server_ms`, `session_id`, `event_id`,
`turn_id`, `token_cost`, `payload`
(top-level `additionalProperties: false`).

`token_cost` = `{input_tokens: 0, output_tokens: 0, estimated_usd: 0.0,
basis: "non_generative_pipeline"}` on every event.

Event types and their key payload fields:

| `event_type` | Key payload fields |
|---|---|
| `transcript_normalized` | `raw_text`, `normalized_text`, `correction`, `disfluencies` |
| `controller_decision` | `decision`, `reason`, `revision` |
| `intents_decomposed` | `intents` (public), `method`, `decomposer_diagnostic` |
| `provisional_invalidated` | `revision_of`, `discarded_candidates` |
| `retrieval_started` | `retrieval_event_id`, `query`, `intent_id`, `trigger` |
| `retrieval_finished` | `retrieval_event_id`, `intent_id`, `candidate_chunk_ids`, `phase` |
| `candidate_cache_reused` | `retrieval_event_id`, `query`, `candidate_chunk_ids`, `phase` |
| `final_evidence_ranked` | `retrieval_event_id`, `intent_id`, `candidate_chunk_ids`, `reranked` |
| `retrieval_timeout` | `reason` |
| `stale_result_discarded` | `revision`, `current_revision` |
| `claim_verified` | `claim_id`, `verification`, `citations`, `candidate_ids`, `intent_id` |
| `answer_version` | `version`, `parent_version`, `added`, `removed`, `replaced`, `preserved` |
| `response_emitted` | `elapsed_ms`, `backend` |

Core-library engine telemetry additionally uses `claim_evidence_selected`
(verification score/label, selected_chunk_id, rejected_citation_ids,
uncertainty_required).

## 5. Recording rules

- `server_ms` measured from session origin (`perf_counter`).
- `timestamp_s` = client `timestamp_ms / 1000`.
- Each event appears both inline in the matching `update.retrieval_events`
  and in the session log served by `GET /session/{id}/telemetry`.
- Every event carries the full contract; the type set is closed under the
  enum above.

## 6. Interfaces

- `transcript-input` ↔ Module 07 Pydantic model ↔ Module 06.
- `live-output` ↔ Module 06 snapshot builder ↔ Module 09 renderer.
- `telemetry-event` ↔ Modules 05/06 recorders ↔ Module 07 endpoint.
- `organizer-benchmark` ↔ Module 08 ↔ Module 09 upload dialog.
- `replay-case` ↔ `tests/` acceptance scenarios ↔ `docs/testing.md`.
- Schemas are contract-only; no runtime code imports the JSON files.