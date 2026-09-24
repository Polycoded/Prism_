# Module 01 — Multi-Intent Decomposer

**STATUS: FIXED — reproduce exactly.** This module is shipped as reference
code. Do not redesign, simplify, replace, or modify its methodology.

This module converts one raw compound utterance into zero or more **searchable
scoped intents**. It has two halves:

1. **The decomposition half** (`prototype/decomposition.py`, shipped): a
   deterministic splitter plus shared vocabulary helpers (`words`, `terms`,
   `SYNONYMS`, `FILLER`).
2. **The BERT boundary tagger half** (`bert_intent_tagger/` for training,
   `prototype/bert_decomposition.py` for inference, both shipped): a
   fine-tuned DistilBERT token classifier guarded by structural + confidence
   validation.

Both halves share the contract `decomposer.split(text) -> tuple[Intent]` via
the `HybridDecomposer` facade; `HybridDecomposer` is the entry point the
remaining modules use.

## 1. Reference files (shipped)

| File | Role |
|---|---|
| `prototype/decomposition.py` | Vocabulary, rule splitter, classifiers (`presentation`, `social`, `translation_request`). |
| `prototype/bert_decomposition.py` | `DistilBertBoundaryDetector`, `BertIntentAdapter`, `HybridDecomposer`. |
| `bert_intent_tagger/` | Training scripts `01`–`07`, `model/checkpoints/best/`, `results/`, `README.md`, `TECHNICAL_REPORT.md`, `requirements.txt`. |

Required dependencies: `bert_intent_tagger/requirements.txt` (torch 2.5.1,
transformers 4.46.3, datasets, accelerate, seqeval, numpy); spaCy 3.8.7 +
en_core_web_sm 3.8.0 optional for `parser="spacy"`.

## 2. Purpose

Detect boundaries between distinct requests inside a single spoken utterance,
keep the rest of the pipeline independent of which splitter produced the
boundaries, and preserve enough per-intent metadata that retrieval can be
scoped correctly. Implicit property lists split only when they name distinct
corpus section concepts; coordination inside a single request (numeric range,
product name, "terms and conditions") must not split.

## 3. Inputs

- `text`: cleaned raw utterance. The normalization module (Module 03) runs
  first; the decomposer still tolerates light disfluency.
- `chunks`: `CorpusChunk` records used to build the alias map
  (`doc_id -> metadata["entity"]`) and the domain property vocabulary.

## 4. Outputs — the `Intent` contract

`split(text) -> tuple[Intent]`. `Intent` is a frozen dataclass:

| Field | Type | Meaning |
|---|---|---|
| `key` | `str` | `sha256(repr((entity_ids, topic, constraints)))`[:16] |
| `query` | `str` | retrievable request text; inherited scope is prepended as subject. |
| `subject` | `str` | `" / ".join(alias)` of entity ids, or `""`. |
| `entity_ids` | `tuple[str,...]` | corpus doc ids scoping this request. |
| `topic` | `tuple[str,...]` | sorted synonym-normalized content terms min. stopwords. |
| `constraints` | `tuple[str,...]` | values like `24 months`, `not`, `before`, `only` (regex `\b(?:\d+(?:\.\d+)?(?:\s*(?:people|month[s]?|year[s]?|day[s]?|ghz|percent))?|not|without|before|after|only)\b`). |
| `source_spans` | `tuple[(int,int),...]` | character spans in the original input; a fronted `For X,` scope prepends `(0, scope_end)`. |
| `method` | `str` | `"bert_intent_boundary"`, `"spacy_dependency_rules"`, or `"structural_rules"`. |

Also exposed: `entities(text)`, `topic(text, entities=())`, `aliases`,
`method`, and (on the hybrid) `consume_diagnostic()`.
`Intent.public()` = `asdict`; this is what appears in `update.sub_queries`.

## 5. Vocabulary (`prototype/decomposition.py`)

- `words(text)` = `re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())`.
- `FILLER`: fixed stopword set — reproduce verbatim
  (`a an the and or for to of on in at is are was were be been being it its
  they their them this that these those i we you my our your please tell show
  give summarize repeat me about what when where which who why how do does did
  can could would should must will has have had need needed required want know
  much many long get with from by as s not only also actually instead than
  rather now thanks thank hello hi open opens opening applies apply kept done
  standard used use often`).
- `SYNONYMS`: canonicalization map, e.g. `guarantee→warranty`, `duration→
  period`, `proof→receipt`, `months→month`, `set→setup`, `free→charge`, etc.
  Reproduce the shipped dict exactly.
- `terms(text)` = `{SYNONYMS.get(w,w) for w in words(text) if w not in
  FILLER}`; adds `period` when a number-range or `how long` pattern matches.
- `digest(text)` = `sha256(text.encode()).hexdigest()[:16]`.

## 6. Rule splitter (`Decomposer`)

`Decomposer(chunks, parser)` where `parser` is `"spacy"` (loads
`spacy.load("en_core_web_sm", disable=["ner"])`; method
`spacy_dependency_rules`) or `"rules"` (method `structural_rules`). Other
values raise `ValueError`.

`split(text)` algorithm:

1. **Fronted scope capture.** If `^\s*(?:for|regarding|about)\s+([^,;]+)[,;]\s*`
   matches and the group contains a corpus entity AND contributes no topic
   terms, treat it as scope; splice it out; offset spans afterwards by its
   length.
2. **Cut candidates.** Scan port `;\s* | [?!.]\s+(?=(?:what|when|where|how|
   can|do|does|is|are|which|who)\b) | ,\s*(?=(?:what|when|where|how|can|do|
   does|is)\b) | \b(?:and\s+also|and|also)\s+` (case-insensitive). A match is a
   cut only when:
   - it is not inside a **protected span** (number ranges `between X and Y`,
     `terms and conditions`, `research and development`, or any entity alias);
     and
   - the right side opens with an explicit question word, or the match is a
     semicolon, or the two sides have disjoint non-empty corpus `properties`
     AND either the token at the cut is a noun/conjunction (`dep_ == "conj"`
     or head `conj`, POS in {NOUN, ADJ, DET}) or a fronted scope exists.
3. **Segmentation.** Slice at cut endpoints; strip ` ,;?`; drop segments with
   no `terms`.
4. **Entity resolution.** Per segment: entities literally present; else, if
   segment contains a pronoun (`it|its|they|their|them`) inherit the most
   recent single entity; else inherit the fronted scope (only when it resolves
   to exactly one entity).
5. **Query rewriting.** If the subject is inherited, `query = subject + " " +
   segment-without-pronouns`.
6. **Dedup.** `digest(repr((entity_ids, topic, constraints)))`; keep only the
   first occurrence. Exact canonical equivalence only — similarity merging is
   forbidden.
7. **Spans.** `(base+a, base+b)`; with a scope, prepend `(0, base)`.

Supporting classifiers (used by Module 06 and must be reproduced exactly):

- `presentation(text)`: full-match grammar for presentation-only turns
  (bullets / repeat / shorten / summarize). On match no decomposition is
  attempted. Reproduce the regex from the shipped file.
- `presentation_prefix(text)`: `True` only for a leading possible formatting
  command with no factual words.
- `social(text)`: `" ".join(words(text))` in `{"hi","hello","hey","thanks",
  "thank you"}`.
- `translation_request(text)`: full-match `translate … into|to <lang>`.

## 7. BERT tagger — training data (reproduction contract)

Training boundaries are reconstructed from the public AGIF MixATIS/MixSNIPS
cleaned corpora by **exact token-sequence alignment** with AGIF's original
single-intent splits. Reproduction (exact scripts shipped):

- `01_download_data.py` — downloads `MixATIS_clean_{train,dev,test}.txt`,
  `MixSNIPS_clean_{train,dev,test}.txt` and the original `AGIF_ATIS_/AGIF_
  SNIPS_{split}.txt` from `raw.githubusercontent.com/LooperXX/AGIF/master/data`,
  writing `data/raw/MANIFEST.json`.
- `02_convert_to_bio.py` — parses AGIF records and **fails hard** (AGIF raw
  records lack token-to-intent alignment).
- `02_recover_source_boundaries.py` — finds the **unique ordered exact
  token-sequence match** of each declared intent against the source split.
  None or ambiguous → rejected (`no_unique_exact_source_alignment`). Accepts
  rows into `data/converted/source_corpus_exact_match_v1_{split}.jsonl` with
  `boundary_source="source_corpus_exact_match_v1"`, per-token `bio_tags`, and
  `num_intents`. Writes a stratified proportional 150-row train review sample.
- `03_split_dataset.py` — emits `data/splits/{train,dev,test}.jsonl` and
  `results/split_summary.json`. Expected counts: **46,697 / 2,913 / 2,959**.

## 8. BERT tagger — model and training (reproduction contract)

`04_train.py`:

- Base model `distilbert-base-uncased`, `AutoModelForTokenClassification` with
  labels `["O","B-INTENT","I-INTENT"]`.
- Tokenization: `is_split_into_words=True`, `max_length=128`, first sub-token
  gets the label, others `-100`.
- Training: 3 epochs, lr `3e-5`, warmup ratio `0.1`, train batch `8`, eval
  batch `16`, weight decay `0.01`, seed `20260919`, eval+save each epoch,
  best model by dev `bio_f1` (seqeval), `fp16` on CUDA.
- Output: `model/checkpoints/best/` (+ tokenizer) and
  `results/training_curves.json`.

Fixed reference outcomes recorded in `results/`: dev BIO F1 `0.9994972`; test
BIO F1 `0.9986689`; test exact span-set accuracy `0.9979723`.

## 9. BERT tagger — evaluation scripts

- `05_evaluate.py` — Trainer.predict on test; writes `results/metrics.json`,
  `results/bert_test_predictions.jsonl`.
- `06_compare_to_rule_based.py` — compares the rules splitter and BERT spans
  against the recovered reference, writing `results/rule_based_comparison.json`
  and `results/error_analysis.md`. **Note:** this script subprocesses
  `prototype.runtime.LiveRuntime`, which is a Module 06 build target; it only
  becomes runnable after Modules 03–06 are built. The file is shipped
  untouched.
- `07_venue_domain_evaluate.py` — builds a fixed 200-case synthetic
  venue-domain review (30 single controls, 30 explicit two-intent, 30
  elliptical, 30 three-intent, 40 ASR-style, 40 hard negatives) and writes
  `results/venue_domain_evaluation.json` + `results/venue_domain_review.jsonl`.
  Known outcomes: rules 57.5%, raw accepted BERT 91.0%, guarded `auto` 88.5%,
  `promotion_eligible: false` until independent human review.

## 10. BERT inference (`prototype/bert_decomposition.py`, shipped)

### `DistilBertBoundaryDetector(model_path, max_length=256)`
Lazy `_load()` via `AutoTokenizer/AutoModelForTokenClassification` with
`local_files_only=True`, then `.eval()`.

`predict(text)`:
1. split on `\S+`; tokenize word list
   (`is_split_into_words=True`, `truncation=True`, `max_length=256`);
   take argmax label + softmax confidence per **word** (first sub-token, skip
   `None`/repeated word ids).
2. labels count ≠ words → `valid=False`, `failure_reason="input_truncated"`.
3. Convert BIO to spans; `I-INTENT` without preceding `B-INTENT` →
   `invalid_I_transition_at_word_<i>`. Span confidence = min word confidence.
4. Return `BoundaryPrediction(spans, labels, word_confidences, latency_ms,
   valid, failure_reason)`. Exceptions → `model_error:<Type>:<msg[:160]>`.

### `BertIntentAdapter(rules_decomposer, detector, threshold=0.90, max_spans=8)`
`split(text)` returns `(tuple[Intent], BoundaryPrediction)`; requires `valid`,
`len(spans) <= max_spans`, and every span `>= threshold`, else raises
`ValueError`. Steps: trim ` ,;?.!`; drop glue-only spans; drop a fronted
`for/regarding/about` non-question first span (its entity is inherited); strip
leading connectives (`and based|also|then`) and fillers (`um|uh|erm|okay|ok|
well`); trim a fronted scope inside the first request up to the question word;
resolve entities/topic/constraints/dedup/fingerprint like the rule splitter;
and run the **uncovered-content check** (only connectors, fillers, aliases may
be gaps — `terms(uncovered)` non-empty → `ValueError("meaningful_uncovered_
content")`). `last_prediction()` returns the stored

### `HybridDecomposer(chunks, mode, fallback_parser="spacy", model_path=None,
threshold=0.90, detector=None)`
`mode ∈ {bert, shadow, auto}`. `split(text)`:
1. compute rules intents; try BERT; `agrees = signature(rules) ==
   signature(bert)` (trimmed source spans); if not agreeing AND
   `min(span.confidence) < disagree_threshold` raise
   `"disagreement_confidence_below_threshold"`.
2. serve BERT in `bert`/`auto`, rules on any exception or in `shadow`.
3. store diagnostic in a `ContextVar` (read via `consume_diagnostic()`):
   `mode, served, accepted, fallback_reason, agreement, rule_spans,
   bert_spans, bert_latency_ms, min_confidence`.

## 11. Configuration

`CITEFRONTIER_PARSER`, `CITEFRONTIER_FALLBACK_PARSER`, `CITEFRONTIER_BERT_MODEL`,
`CITEFRONTIER_BERT_THRESHOLD`, `CITEFRONTIER_BERT_DISAGREE_THRESHOLD`,
`CITEFRONTIER_BERT_PRELOAD`. Module 06 constructs `HybridDecomposer` for modes
`bert|shadow|auto`, else `Decomposer(chunks, mode)`.

## 12. Interfaces

- Consumed by: Module 04 (aliases for corrections), Module 05 `session`,
  Module 06 runtime, Module 02 (`decomposer.topic`, `aliases`).
- Emits: `Intent` records and the decomposer diagnostic into telemetry.

## 13. Validation

Ship `prototype/tests/test_bert_decomposition.py`-equivalent coverage: guarded
`auto` acceptance with a fake detector; `shadow` service + agreement logging;
invalid/low-confidence fallback; fronted-scope preservation; boundary +
scope + hard-negative regressions for both splitters; scope-offset integrity;
and the venue review gate via `scripts/07`.

## 14. Important implementation details

- Never merge intents by embedding similarity; exact fingerprints only.
- Never split protected ranges/names.
- Source offsets are against the original pre-trim utterance string.
- BERT inference is `local_files_only=True`; never download at inference.
- Constants (seeds, thresholds, counts) are behavioral; changing them alters
  acceptance behavior.