# Module 02 — Corpus Retrieval and Fusion

**STATUS: FIXED — reproduce exactly.** This module is shipped as reference
code. Do not redesign, simplify, replace, or modify its methodology.

This module owns the corpus (format, construction, provenance), retrieval
(sparse + dense), fusion, and reranking. Two implementations of the same
retrieval contract exist and both are part of the fixed component:

- `prototype/search.py` — the live runtime's `SearchIndex`, in `lightweight`
  and `dense` modes.
- `citefrontier/retrieval.py` — the dependency-light `HybridRetriever` and
  `DenseHybridRerankRetriever`.

## 1. Reference files (shipped)

| File | Role |
|---|---|
| `citefrontier/models.py` | `CorpusChunk` (and `Evidence`) frozen dataclasses. |
| `citefrontier/text.py` | `content_tokens` (stopword-filtered tokenizer) used by retrieval. |
| `citefrontier/retrieval.py` | `HybridRetriever`, `InMemoryCorpusGraph`, `DenseHybridRerankRetriever`. |
| `prototype/search.py` | `SearchIndex` with pinned embedding/reranker constants. |
| `prototype/ingest.py` | Markdown → `corpus.json` + `corpus-manifest.json` builder. |
| `prototype/prepare_models.py` | Provision dense model assets (offline cache). |
| `prototype/corpus.json`, `prototype/corpus-manifest.json`, `prototype/models-manifest.json`, `prototype/corpus/sources/*.md` | Bundled corpus and provenance. |

Dependencies: `rank-bm25==0.2.2`, `sentence-transformers==5.7.0` (dense only;
installed from the official torch index separately), no deps for lightweight.

## 2. Purpose

Given a decomposed `Intent`, retrieve corpus passages that can support an
answer, fuse two independent ranking signals, rerank the survivors, and —
through eligibility + vocabulary + overlap gates — return either a short
trustworthy candidate list or an empty set so the runtime can abstain.
Retrieval must be reproducible and offline at inference time.

## 3. Corpus: format and construction

### 3.1 `CorpusChunk` (`citefrontier/models.py`)

```python
@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str    # "<doc_id> §<section>"; the ONLY legal citation id
    doc_id: str
    section: str
    text: str
    metadata: dict[str, str]
```

Runtime invariants (enforced by the Module 06 runtime when it loads a corpus):
non-empty, unique `chunk_id`s; every chunk has `doc_id`, `section`, non-empty
`text`; if `metadata.start/end` present, `end - start == len(text)`.

### 3.2 Markdown ingestion (`prototype/ingest.py`)

`python -m prototype.ingest` reads every `*.md` under `prototype/corpus/
sources/` and writes `prototype/corpus.json` + `prototype/corpus-manifest.json`.

Rules: `doc_id` = filename stem; `entity` = `doc_id.replace("_"," ")`; every
`## heading` is a section; complete body between headings is a chunk (no
truncation); `metadata.title` = first `#` line; `metadata.source` = filename;
`metadata.source_hash` = SHA-256 of the whole normalized file (LF);
`metadata.start/end` = char offsets of the body; `chunk_id = "<doc_id> §
<section>"`. The manifest records `corpus_sha256` and per-document
`{file, sha256, characters}`.

Bundled corpus: 10 documents / 30 sections; manifest `corpus_sha256 =
693aeffa…`. Re-implementers must reproduce the corpus byte-for-byte (hashes in
the shipped manifest are the target).

### 3.3 `Evidence` (`citefrontier/models.py`)

`Evidence(chunk, score, coverage, origin, retrieval_event_id, path)` — the
core-library retrieval return type.

## 4. The live runtime retriever (`prototype/search.py`)

`SearchIndex(chunks, decomposer, backend="lightweight", rerank=True)`.

Indexes built at construction:
- `tokens = {chunk_id: terms(text)}` (Module 01 vocabulary);
- `topic_tokens = {chunk_id: decomposer.topic(text, (doc_id,))}`;
- `idf[t] = log(1 + N / (1 + df_t))`;
- dense mode: `BM25Okapi([words(text)…])`, the `all-MiniLM-L6-v2` embedding
  matrix (normalized), and (if `rerank`) the `ms-marco-MiniLM-L-6-v2`
  cross-encoder. All loads `local_files_only=True`.

### 4.1 Eligibility (`_eligible(intent)`)

- **Unknown-sibling guard:** scan the query for `firstword + modeltoken`; if a
  mentioned model is not a known entity, return `()`;
- empty `entity_ids` → all chunk ids;
- else ids whose `doc_id` is in the entity ids or whose text mentions the
  entity's alias.

### 4.2 Recall (`candidates(intent) -> tuple[chunk_id]`)

- `lightweight`: sparse score `Σ idf / sqrt(len(chunk))` and Jaccard overlap;
  top-20 per ranking by `(-score, chunk_id)`; RRF-merge `1/(60+rank)`; return
  top-20 by `(-fusion, chunk_id)`.
- `dense`: `dense = query_embedding @ matrix`; `sparse = BM25Okapi`; top-20 per
  ranking over the eligible set; RRF-merge; top-20. Model calls hold
  `model_lock` (`threading.RLock`).

### 4.3 Finalize (`finalize(intent, candidate_ids) -> tuple[Candidate]`)

1. Keep ids in the eligible set and chunk map; empty → `()`.
2. **Vocabulary gate:** `topic - vocabulary - allowlist` non-empty → `()`. The
   allowlist is fixed: `{"period","exclude","receipt","replace","local",
   "require","coverage","repair","setup","history","trial","kept","reminder"}`.
3. Score: dense+rerank → cross-encoder; else weighted topic overlap
   `Σ idf(topic∩topic_tokens)/Σ idf(topic) + 0.1·len(topic∩terms(section
   head))`.
4. Filter: drop passages matching `\b(?:ignore|disregard|override)\b.{0,80}
   \b(?:instructions|system|rules)\b`; require `topic ∩ topic_tokens` non-empty
   (`relevance = len(overlap)/len(topic)`); lightweight additionally requires
   `relevance >= 0.30`.
5. Return top-5 by `(-score, -relevance, evidence_id)` as
   `Candidate(evidence_id, score, relevance)`.

### 4.4 Pinned model revisions (do not change)

```python
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION  = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_REVISION= "233902d25c440f23af6f7d6e94d2946bac0bee0a"
```

`python -m prototype.prepare_models` snapshots these revisions into the
Hugging Face cache (`ignore_patterns=["onnx/*","openvino/*","*.h5","*.ot",
"pytorch_model.bin"]`) and writes `prototype/models-manifest.json`. Inference
is `local_files_only`.

## 5. The core-library retrievers (`citefrontier/retrieval.py`)

Common interface: `search(query, retrieval_event_id, limit=5) ->
tuple[Evidence, ...]`.

- `HybridRetriever(chunks, graph=None, mode="hybrid")`: tokenizes with
  `content_tokens`. `hybrid`: fuse frequency-weighted lexical + Jaccard scores
  via `1/(60+rank)`, require coverage ≥ 0.5. `semantic` mode: Jaccard only
  (`origin="token_semantic_proxy"` — disclosed proxy, not embeddings).
  `InMemoryCorpusGraph` adds one-hop neighbors at `anchor_score * 0.97`
  (`origin="graph_one_hop"`).
- `DenseHybridRerankRetriever(chunks, graph=None, candidate_limit=40)`: BM25 +
  `all-MiniLM-L6-v2` + RRF over the top-40, optional one-hop expansion, then a
  final `ms-marco-MiniLM-L-6-v2` rerank to top-5. Fails closed (RuntimeError)
  when optional deps are missing.

## 6. Interfaces

- Consumed by the Module 06 runtime: `SearchIndex.candidates(intent)` and
  `SearchIndex.finalize(intent, ids)`; chunk records expose `.chunk_id`
  (citations), `.doc_id`, `.section`, `.text` (extract), `.metadata`
  (provenance/offsets).
- Consumed by the Module 05 session/evidence layer via the core-library
  retrievers (`citefrontier.retrieval`).
- Depends on Module 01 vocabulary (`terms`, `words`) and `decomposer.topic`/
  `aliases`.
- Emits/feeds telemetry events `retrieval_started`, `retrieval_finished`,
  `candidate_cache_reused`, `final_evidence_ranked` (built modules).

## 7. Configuration

`CITEFRONTIER_BACKEND` (`lightweight`|`dense`), `CITEFRONTIER_CORPUS`. The
shipped `SearchIndex(…, rerank=…)` flag turns the dense reranker on/off.
Behavioral constants: candidate window 20, final top-k 5, RRF constant 60.

## 8. Validation

`python -m prototype.ingest` must regenerate `corpus.json` with hashes equal to
the shipped manifest. `python -m prototype.prepare_models` must succeed once
with network, then offline inference must work from the local cache. Module 05
acceptance tests must include: unknown-sibling rejection, conflicting quantity
withholding, instruction-attack refusal, unknown-topic abstention, and
exact-extract + citation-in-candidate checks.

## 9. Important implementation details

- Citations must resolve to `chunk_id`s in the final committed candidate set
  (enforced by the Module 05 synth layer).
- The vocabulary gate deliberately trades recall for abstention.
- `lightweight` must be fully deterministic and dependency-free.
- Corpus sources, hashes, and offsets drive the Module 09 source inspector;
  keep them consistent when the corpus is rebuilt.