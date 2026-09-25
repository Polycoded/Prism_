# CiteFrontier — System Architecture

This document specifies the **target system** that the blueprints in this
directory are used to build. Two components are shipped as reference
implementations (Modules 01–02); the rest are built from their blueprint specs.

## 1. What the system is

CiteFrontier is a **local, non-generative streaming retrieval prototype**. It
receives timestamped transcript hypotheses over a WebSocket while the speaker
is still forming a complete question, does useful work on the partial text, and
commits answers only when the transcript is final.

Three hard constraints shape every module:

1. **No generative model.** Answers are exact extracts copied verbatim from a
   real corpus chunk. Every event records an explicit zero token cost with the
   basis `non_generative_pipeline`.
2. **No unverified claims.** A claim may carry a citation only if the citation
   is a real corpus chunk ID that belongs to the final committed intent's
   candidate set and passes exact-extractive verification.
3. **Revision safety.** Corrections invalidate earlier speculative work, and a
   late-arriving asynchronous result can never overwrite the answer produced by
   a newer transcript generation.

The corpus is the only knowledge source: no web search, no external answer
API, no voice pipeline.

## 2. What enters the system

A client opens a WebSocket to `/ws/stream`, receives a `ready` message with a
session ID and per-session bearer token, then sends a sequence of JSON
**transcript hypothesis messages** matching `schemas/transcript-input.schema.json`:

```json
{
  "event_id": "event-3",
  "turn_id": "turn-1",
  "text": "For LumaPad S1, what is the warranty period and what liquid damage is excluded",
  "timestamp_ms": 1500,
  "is_final": false,
  "revision_of": null,
  "target_intent_key": null,
  "sequence": 3
}
```

Input contract:
- `text` is the **cumulative** hypothesis for the turn, not the appended words.
- `is_final` is `true` exactly once per turn, when the transcript is committed.
- `revision_of` references an earlier `event_id` of the same turn when the
  speaker corrected the phrasing.
- `target_intent_key` selects one claim for targeted refinement and is valid
  only on a final event.
- Events arrive in strictly increasing `sequence`; a turn cannot be finalized
  twice; `event_id` cannot be reused with different content; at most 500 events
  per session.

## 3. How input is processed

Each event passes through five stages (details in the module blueprints):

1. **Idempotency** (Module 06). Handled `event_id`s re-serve their stored
   receipt marked `duplicate`; duplicates never re-run retrieval.
2. **Normalization** (Module 03). Fillers (`um`, `uh`, `erm`, `er`, `ah`),
   immediate word repetitions, and explicit local corrections ("6 — sorry, 7",
   "I meant S2") are resolved deterministically. Raw, cleaned, and removed text
   are kept for telemetry.
3. **Controller decision** (Module 04). One of:
   - `wait` — prefix not yet a stable searchable intent;
   - `provisional_retrieve` — stable searchable prefix; candidates are cached
     but must never become claims;
   - `commit_retrieve` — final transcript; candidates are reused/re-run then
     revalidated, and claims may be committed;
   - `suppress` — presentation-only or social turn ("put that in bullets",
     "hi"); no retrieval.
4. **Decomposition** (Module 01). The Multi-Intent Decomposer splits the text
   into scoped `Intent` records (entity scope, topic, constraints, source
   spans, method). In the default configuration the decomposer runs the
   trained DistilBERT boundary tagger in guarded `auto` mode with a
   spaCy/structural fallback.
5. **Synthesis** (Module 05) and **retrieval** (Module 02): each intent is
   searched concurrently; on a final event candidates are reranked and gated,
   and an exact extract or an abstention is chosen per intent. Claims are
   versioned and returned.

## 4. Module map (target layout after rebuild)

```
repo root/
  bert_intent_tagger/         Module 01 reference (shipped)
  prototype/
    decomposition.py          Module 01 reference (shipped)
    bert_decomposition.py     Module 01 reference (shipped)
    search.py                 Module 02 reference (shipped)
    ingest.py                 Module 02 reference (shipped)
    prepare_models.py         Module 02 reference (shipped)
    corpus.json / corpus/sources / corpus-manifest / models-manifest  Module 02
    transcript.py             Module 03 (BUILD)
    controller.py             Module 04 reference (shipped)
    session.py                Module 05 (BUILD)
    runtime.py                Module 06 (BUILD)
    server.py                 Module 07 (BUILD)
    server_types.py           shared TranscriptMessage constructor (BUILD, 07)
    upload.py                 Module 08 (BUILD)
    benchmark.py              Module 08 (BUILD)
    __main__.py               CLI entry point (BUILD, 07)
    static/                   Module 09 (BUILD)
    demo/scenarios.json       optional replay fixtures (BUILD, 09)
    tests/                    Module acceptance tests (BUILD)
  citefrontier/
    models.py, text.py, retrieval.py   Module 02 reference (shipped)
    controller.py                       Module 04 reference (shipped)
  schemas/                    Contracts (shipped, JSON Schema)
  tests/                      acceptance tests for modules 04–07 (BUILD)
  docs/                       this blueprint set
```

This layout is what the blueprints instruct the developer to construct; only
the marked reference files exist before building.

## 5. Execution flow (target)

```
Client                                      Server (to build: modules 06+07)
────────────────────                        ─────────────────────────────────
ws://host/ws/stream              ─────────▶ accept; create a session
                              ◀─────────── {"type":"ready",
                                             session_id, token, backend,
                                             parser, corpus_hash}
partial hypothesis (is_final=false) ──────▶ 1 normalize (03)
                                             2 dedupe by event_id
                                             3 controller: wait|provisional...
                                             if provisional_retrieve:
                                               decompose (01) -> intents
                                               SearchIndex.candidates (02)
                                               cache candidate ids only
                              ◀─────────── update(provisional_retrieve,
                                             sub_queries, claims=[])
final hypothesis (revision_of set
  on correction)                 ─────────▶  invalidate cache if corrected
                                             decompose (01)
                                             per intent: reuse cache OR search
                                             finalize/rerank/gate (02)
                                             build claims (05) - extract or
                                             abstain; version answer
                              ◀─────────── update(commit_retrieve, claims,
                                             citations, answer_version,
                                             history, retrieval_events)
session state is in-memory per connection;
cleared on disconnect. Telemetry requires
the session bearer token.
```

Latency: provisional work may be reused, but the final event always
revalidates candidates against the committed intent before a claim is emitted.

## 6. Environment contract

The runtime reads only environment variables (no config file):

| Variable | Default | Meaning |
|---|---|---|
| `CITEFRONTIER_PORT` | `8000` | HTTP/WS port for `python -m prototype` |
| `CITEFRONTIER_BACKEND` | `lightweight` | `lightweight` or `dense` retrieval backend |
| `CITEFRONTIER_PARSER` | `spacy` | `rules`, `spacy`, `bert`, `shadow`, `auto` |
| `CITEFRONTIER_FALLBACK_PARSER` | `spacy` | splitter used when BERT is not served |
| `CITEFRONTIER_BERT_MODEL` | `bert_intent_tagger/model/checkpoints/best` | local BERT checkpoint directory |
| `CITEFRONTIER_BERT_THRESHOLD` | `0.90` | minimum per-span confidence to serve BERT |
| `CITEFRONTIER_BERT_DISAGREE_THRESHOLD` | `0.97` | minimum confidence to prefer BERT over fallback on disagreement |
| `CITEFRONTIER_BERT_PRELOAD` | `1` | preload the BERT checkpoint at server start |
| `CITEFRONTIER_CORPUS` | `prototype/corpus.json` | runtime corpus JSON path |
| `CITEFRONTIER_DEMO` | `0` | set to `1` to enable `GET /demo-scenarios` fixtures |
| `HF_HUB_DISABLE_XET` | (unset until set by Module 05 grounding) | disables the Xet transfer client for reproducible model downloads |

Retrieval model revisions are pinned in Module 02's `EMBED_MODEL`,
`EMBED_REVISION`, `RERANK_MODEL`, `RERANK_REVISION` and mirrored by
`prototype/models-manifest.json`. The spaCy model is `en_core_web_sm==3.8.0`.

## 7. Dependencies

- Module 01: `bert_intent_tagger/requirements.txt` (torch 2.5.1, transformers
  4.46.3, datasets, accelerate, seqeval, numpy). spaCy 3.8.7 + en_core_web_sm
  3.8.0 optional for `parser="spacy"`.
- Module 02 dense backend: `rank-bm25==0.2.2`, `sentence-transformers==5.7.0`
  (installed separately from the official torch index). Lightweight backend:
  Python standard library only.
- Modules 03–08: `fastapi==0.115.12`, `uvicorn==0.34.2`,
  `websockets==15.0.1`, `httpx==0.28.1`, plus the Module 01/02 sets. Exact
  pins are listed in each module blueprint.
- Module 09: no build step (static assets); vendored GSAP 3.13.0 optional.

## 8. Reproduction entry points (after building)

```powershell
# lightweight product
python -m prototype --backend lightweight

# dense semantic profile
python -m prototype.prepare_models
python -m prototype --backend dense

# guarded BERT boundary detection
python -m prototype --backend lightweight --parser auto

# verification
python -m unittest discover -s prototype/tests -v
python -m unittest discover -s tests -v
```

Decomposer training/evaluation reproduction is in
`bert_intent_tagger/README.md` and `module-01`.