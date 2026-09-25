# CiteFrontier — Rebuild Documentation

This directory is the **build blueprint** for the CiteFrontier system. Two
modules are shipped in this repository as reference implementations and must
be reproduced/used **exactly as they are**; every other module is **not
shipped as code** and must be built from the instructions in its blueprint.

The intended workflow: a developer starts with an empty workspace, takes the
two shipped reference modules, and builds the remaining modules by following
their blueprint specs, in the order below. The result must reproduce the exact
behavior described in `architecture.md`.

## Status table

| Doc | Module | Status | Shipped code |
|---|---|---|---|
| `module-01-multi-intent-decomposer.md` | Multi-Intent Decomposer (BERT tagger + rules/spaCy splitter + guarded hybrid) | **FIXED — reproduce exactly** | `bert_intent_tagger/`, `prototype/decomposition.py`, `prototype/bert_decomposition.py` |
| `module-02-corpus-retrieval-fusion.md` | Corpus constructor + ingestion, retrieval, fusion, reranking | **FIXED — reproduce exactly** | `prototype/search.py`, `prototype/ingest.py`, `prototype/prepare_models.py`, `prototype/corpus*`, `citefrontier/{models,text,retrieval}.py` |
| `module-03-transcript-normalization.md` | Disfluency + correction normalization of inbound text | BUILD | none |
| `module-04-retrieval-controller.md` | Retrieval timing policy (WAIT/PROVISIONAL/COMMIT/SUPPRESS) | **SHIPPED — reference added** | `citefrontier/controller.py`, `prototype/controller.py` |
| `module-05-session-synthesis.md` | Session state, claims, answer versioning, delta refinement, dialogue context, abstention | BUILD | none |
| `module-06-live-runtime.md` | Revision-safe orchestration: event validation, cache, staleness, claim commit | BUILD | none |
| `module-07-websocket-server.md` | FastAPI HTTP API + WebSocket `/ws/stream` delivery | BUILD | none |
| `module-08-corpus-upload-benchmark.md` | Session corpus upload + organizer benchmark runner | BUILD | none |
| `module-09-frontend.md` | Static web client | BUILD | none |
| `module-10-schemas-telemetry.md` | Wire/schema + telemetry contracts (reference) | CONTRACT | `schemas/` (JSON Schema) |

Every BUILD doc is self-contained: purpose, inputs/outputs, data formats,
required file layout, a step-by-step build procedure with exact class/function
signatures, the exact constants and algorithms, interface contracts, and
acceptance criteria. `architecture.md` and `interfaces.md` tie the modules
together. As build modules are implemented, their reference code is added to
the repository and the status column above flips from BUILD to
**SHIPPED — reference added** (the blueprint remains the specification of
record).

## Build order

1. Module 01 (Multi-Intent Decomposer) — reproduce/train per blueprint.
2. Module 02 (Corpus Retrieval and Fusion) — reproduce corpus + retrievers.
3. Module 04 (Retrieval Controller) — builds only on Module 01 delegates.
4. Module 05 (Session Synthesis) — builds on Modules 01, 02, 03.
5. Module 06 (Live Runtime) — orchestrates 01–05.
6. Module 07 (WebSocket Server) — exposes 06.
7. Module 03 (Transcript Normalization) — can be built any time before 05.
8. Module 08 (Upload/Benchmark) — on top of 06/07.
9. Module 09 (Frontend) — consumes 07.
10. Module 10 (Schemas) — authored alongside 03–07; the JSON files under
    `schemas/` are the authoritative contracts.

## Fixed, non-negotiable components

Modules 01 and 02 are frozen design contracts and are shipped as reference
implementations. Their methodology must not be redesigned, simplified,
replaced, or modified. All other modules are rebuilt strictly from their
blueprints.