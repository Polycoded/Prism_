# `prototype/` — Modules 01–02 reference code

This directory holds the reference implementations of the two **fixed**
CiteFrontier components:

- **Module 01 — Multi-Intent Decomposer:**
  - `decomposition.py` — vocabulary, the rules/spaCy structural splitter, the
    presentation/social/translation classifiers, the `Intent` record.
  - `bert_decomposition.py` — `DistilBertBoundaryDetector`,
    `BertIntentAdapter`, and the guarded `HybridDecomposer` facade.
- **Module 02 — Corpus Retrieval and Fusion:**
  - `search.py` — `SearchIndex` with the pinned embedding/reranker revisions,
    reciprocal-rank fusion, and the eligibility/vocabulary/overlap gates.
  - `ingest.py` — Markdown → `corpus.json` builder.
  - `prepare_models.py` — dense model provisioning (offline cache).
  - `corpus.json`, `corpus-manifest.json`, `models-manifest.json`,
    `corpus/sources/*.md` — the bundled corpus and its provenance.
  - `prototype/requirements.txt` — dependencies for the shipped code (spaCy
    optional for `parser="spacy"`; rank-bm25 + sentence-transformers for the
    dense backend; torch/transformers are in `bert_intent_tagger/requirements.txt`).

- **Module 04 — Retrieval Controller (shipped reference):**
  - `controller.py` — the full `decide(...)` decision function
    (`prototype.controller.decide`) used by the live runtime; the
    dependency-light `citefrontier.controller.RetrievalController` lives in
    the `citefrontier` package.

Every other module of the system (session-aware synthesis, live runtime,
WebSocket server, upload/benchmark, frontend, normalization) is **not**
shipped here; it is built from the blueprints in
[`docs/`](../docs/README.md). The target layout the blueprints produce is
documented in [`docs/architecture.md`](../docs/architecture.md).

## Using the reference code

```powershell
# Reproduce the bundled corpus (must match the shipped manifest hashes)
python -m prototype.ingest

# Provision dense retrieval model assets once, then run offline
python -m prototype.prepare_models
```

Standalone decomposer usage (rules mode needs no spaCy):

```python
from prototype.decomposition import Decomposer
import json

chunks = tuple(CorpusChunk(**item) for item in json.load(open("prototype/corpus.json", encoding="utf-8")))
decomposer = Decomposer(chunks, parser="rules")
for intent in decomposer.split("For LumaPad S1, what is the warranty period and what receipt opens a repair?"):
    print(intent.public())
```

Full reproduction instructions and the simpler/harder behavioral contracts are
in [`docs/module-01-multi-intent-decomposer.md`](../docs/module-01-multi-intent-decomposer.md)
and [`docs/module-02-corpus-retrieval-fusion.md`](../docs/module-02-corpus-retrieval-fusion.md).