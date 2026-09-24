# CiteFrontier

A **local, non-generative streaming retrieval system**. It receives timestamped
transcript hypotheses over a WebSocket, decomposes messy multi-intent
questions, starts retrieval while the question is still forming, rejects stale
work when the speaker corrects themselves, and commits only **source-verifiable
answers** — exact extracts from a user-supplied corpus with citations.

This repository is a **reconstruction package**. It ships the two fixed,
heavyweight modules as reference implementations, and specifies every other
module as a build blueprint in [`docs/`](docs/README.md):

| Module | Status | Reference code |
|---|---|---|
| **01 — Multi-Intent Decomposer** (BERT intent-boundary tagger: data, training, model, inference, decomposition logic) | Fixed — reproduce exactly | `bert_intent_tagger/`, `prototype/decomposition.py`, `prototype/bert_decomposition.py` |
| **02 — Corpus Retrieval and Fusion** (corpus construction, retrieval, fusion/reranking) | Fixed — reproduce exactly | `prototype/search.py`, `prototype/ingest.py`, `prototype/prepare_models.py`, `prototype/corpus*`, `citefrontier/{models,text,retrieval}.py` |
| **03–09** — normalization, retrieval controller, session-aware synthesis, live runtime, WebSocket server, upload/benchmark, frontend | Build from blueprint | none |
| **10 — schemas + telemetry contracts** | Shipped contracts | `schemas/` |

System guarantees (see [`docs/architecture.md`](docs/architecture.md)):

- **No generative model** — answers are exact corpus extracts; every event
  records an explicit zero token cost.
- **No unverified claims** — a citation is always a real corpus chunk ID in the
  final committed candidate set, passing exact-extractive verification.
- **Revision safe** — corrections invalidate earlier speculative work; a late
  result can never overwrite a newer answer.
- **Fully local and offline** — a deterministic `lightweight` retrieval
  backend and a `dense` backend (BM25 + MiniLM + cross-encoder rerank) whose
  pinned assets are provisioned in advance and loaded only from the local
  cache.

## Read first

- [`docs/README.md`](docs/README.md) — status table and build order.
- [`docs/architecture.md`](docs/architecture.md) — target system, module map,
  execution flow, environment contract.
- [`docs/interfaces.md`](docs/interfaces.md) — every data structure that
  crosses a module boundary.

## Rebuild the system

1. **Reproduce Module 01** (Multi-Intent Decomposer) — follow
   [`docs/module-01-multi-intent-decomposer.md`](docs/module-01-multi-intent-decomposer.md)
   and `bert_intent_tagger/README.md` (train or reuse the shipped checkpoint).
2. **Reproduce Module 02** (Corpus Retrieval and Fusion) — follow
   [`docs/module-02-corpus-retrieval-fusion.md`](docs/module-02-corpus-retrieval-fusion.md)
   (rebuild the corpus with `python -m prototype.ingest`, provision models with
   `python -m prototype.prepare_models`).
3. **Build Modules 03–09** from their blueprints, in the order in
   `docs/README.md`.
4. **Verify** against [`docs/testing.md`](docs/testing.md):
   ```powershell
   python -m unittest discover -s prototype/tests -v
   python -m unittest discover -s tests -v
   ```

## Repository contents

- `bert_intent_tagger/` — Module 01: training scripts, promoted checkpoint,
  results, technical report.
- `prototype/` — Modules 01–02 reference code (decomposer, BERT boundary
  inference, retrieval/fusion, corpus builder, corpus, model manifest).
- `citefrontier/` — Module 02 reference subset (corpus record types, text
  helpers, retrievers).
- `schemas/` — wire and telemetry contracts for the build-to-spec modules.
- `docs/` — the reconstruction blueprints.

## Dependencies

- Module 01: `bert_intent_tagger/requirements.txt` (torch 2.5.1,
  transformers 4.46.3, datasets, accelerate, seqeval, numpy); spaCy optional.
- Module 02 dense backend: `rank-bm25`, `sentence-transformers` (see
  `prototype/requirements.txt`).
- Modules 03–09 dependency pins are listed in each blueprint.

## Boundaries

- The shipped corpus and replay fixtures are a **fictional synthetic control**;
  replace them through the corpus ingest or the session corpus upload feature.
- The finished product performs text processing only: no microphone, no voice
  pipeline, no web search or external-knowledge path, no generative answer
  layer.
- Everything is in-memory and per-session; session state and telemetry are
  cleared on disconnect.