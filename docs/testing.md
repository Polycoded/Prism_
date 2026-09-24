# CiteFrontier — Acceptance Criteria and Testing

This document defines the verification contract for a rebuilt system. It lists
what must hold for each module and supplies the behavior families that the
acceptance tests must cover. After building, the developer authors unit
tests under `prototype/tests/` (product) and `tests/` (core library) that
encode these criteria.

## 1. Per-module acceptance

| Module | Must satisfy |
|---|---|
| 01 Decomposer | `bert_intent_tagger` reproduction (train/5/6/7 outcomes in `module-01`); boundary/scope/hard-negative regressions; guarded `auto` acceptance with a fake detector; offsets valid; no similarity merging; protected coordination not split. |
| 02 Retrieval/Fusion | `python -m prototype.ingest` reproduces `corpus.json` hashes equal to the shipped manifest; `prepare_models` then offline dense inference works; eligibility/vocabulary/overlap gates abstain correctly; RRF constant and top-k windows per `module-02`. |
| 03 Normalization | The fixed behavior table in `module-03` §7; corrections ambiguous path never silently rewrites a product. |
| 04 Controller | Decision/reason pairs per `module-04` §6, including the 60-scenario expectations. |
| 05 Session | Provisional never commits; correction invalidates provisionals; delta refinement preserves unaffected claims; conflict/unknown/instruction abstentions; context only from user intents; citations ⊆ committed candidates. |
| 06 Runtime | Idempotency, staleness, cache reuse, parallel scheduling, suppress-with-bullets, envelope shape (`schemas/live-output`). |
| 07 Server | HTTP endpoints, WS lifecycle, token-gated telemetry/corpus, error handling without disconnect. |
| 08 Upload/Benchmark | Upload validation matrix; session isolation; benchmark pass/fail + support + fabricated-ID counting. |
| 09 Frontend | Rendering, inspector, diff, bullets suppression, responsive, reduced-motion, upload/benchmark UI, zero JS errors in tested flows. |
| 10 Schemas | Every emitted telemetry event satisfies `telemetry-event.schema.json`; every update satisfies `live-output.schema.json`. |

## 2. Acceptance scenario families (replay set)

A replay set of **60 scenarios** across families must pass. Each scenario is a
list of transcript steps; the final step is committed (`is_final`); optional
`prefixes` exercise partial/provisional behavior. Families:

| Family | Count | Expectation |
|---|---|---|
| single | 10 | exact citation equals the gold chunk; one claim. |
| compound | 15 | two intents, two independently cited extracts (gold = 2 chunk ids). |
| correction | 10 | prefix mentioning product X, final committing product Y with `revision_of`; cites Y only. |
| refinement | 10 | initial answer, then a delta turns `targeted_delta`; unaffected claim preserved; one extra search; claim ID stable. |
| presentation | 5 | after an answer, a formatting turn → `suppress`, claims/citations/stats/version unchanged; mixed factual turns are NOT suppressed. |
| unsupported | 10 | no citations, `uncertainty` present, zero fabricated IDs. |

Replay checks per step:
- `set(citations) == set(gold)` (or, for `unsupported`, no citations and
  non-empty uncertainty);
- decision/reason (e.g. `targeted_delta` on refinement, `suppress` on
  presentation);
- `len(sub_queries) == intent_count` when declared;
- no new searches on suppressed turns;
- unchanged claims byte-identical to the previous step;
- every cited claim: `text == evidence[0].text` and a citation ⊆
  `candidate_ids`.

Additional metrics the replay must produce: attribution
`supported/emitted == 1.0`; intent coverage `correct/expected == 1.0`; early
retrieval triggers on eligible prefixes; trace coverage 100% (every event has
all `telemetry-event` fields); fabricated IDs = 0.

## 3. Adversarial behaviors that must keep passing

1. **Staleness** — a delayed retrieval that finishes after a revision emits
   `stale_result_discarded` and never reaches the answer.
2. **Provisional isolation** — tentative ids may be reused but a claim may
   only cite ids revalidated against the final intent.
3. **Exact extraction** — each cited claim text equals its evidence verbatim.
4. **Abstention** — telepathy/teleportation/unknown sibling product/
   conflicting quantities (`12-month` vs `24-month` for the same entity &
   property)/instruction-like text → uncertainty, no citation.
5. **Refinement continuity** — a late detail replaces only the affected claim
   (`claim_id` stable, `claim_revision+1`).
6. **Presentation suppression** — formatting-only turns leave search count,
   claims, citations, and version unchanged.
7. **Idempotency** — resending an `event_id` returns the stored result with
   `duplicate=True` and no new work.
8. **Authorization** — session telemetry/corpus endpoints require the bearer
   token and vanish on disconnect.
9. **Unknown sibling guard** — "LumaPad X99 warranty" must not borrow the
   LumaPad S1 warranty.

## 4. Reproducing the fixed modules

### Module 01 (decomposer)
```powershell
python scripts/01_download_data.py
python scripts/02_convert_to_bio.py        # intentionally exits
python scripts/02_recover_source_boundaries.py --split train   # + dev, test
python scripts/03_split_dataset.py
python scripts/04_train.py
python scripts/05_evaluate.py
python scripts/06_compare_to_rule_based.py  # requires the rebuilt runtime
python scripts/07_venue_domain_evaluate.py
```
Expected: dev BIO F1 0.9995, test BIO F1 0.9987, test exact span accuracy
0.9980; venue review 200 cases; `promotion_eligible=false`.

### Module 02 (corpus + retrieval)
```powershell
python -m prototype.ingest            # must match shipped manifest hashes
python -m prototype.prepare_models    # once, with network
```
Spot checks: every chunk text is an exact source slice; `metadata.source_hash`
equals the SHA-256 of the source file; dense inference loads from the local
cache only.

## 5. Suggested test layout

- `prototype/tests/` — `test_transcript.py` (03), `test_controller.py` (04),
  `test_session.py` (05), `test_live.py` (06 + API), `test_upload.py` (08),
  `test_benchmark.py` (08), `test_bert_decomposition.py` (01 integration with
  a fake detector).
- `tests/` — `test_validation.py` (05 invariants), `test_grounding.py` (05),
  `test_ingest.py` (02), `test_evaluation.py` (04/05 replay proxies).
- Run: `python -m unittest discover -s prototype/tests -v` and
  `python -m unittest discover -s tests -v`.