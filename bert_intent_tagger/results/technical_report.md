# DistilBERT intent-boundary tagger — technical report

## Purpose

This standalone model detects boundaries between requests in a compound utterance. It predicts three token labels: `B-INTENT`, `I-INTENT`, and `O` (connective/glue tokens).

## Training data

The raw cleaned MixATIS and MixSNIPS records do not directly contain intent spans. We recovered boundaries by exact ordered token matching against AGIF's corresponding original single-intent corpus split. A record was accepted only with one unique complete alignment; unmatched or ambiguous records were rejected.

| Partition | Accepted examples |
| --- | ---: |
| Train | 46,697 |
| Dev | 2,913 |
| Test | 2,959 |

Every record is explicitly marked `boundary_source: source_corpus_exact_match_v1`. These are reconstructed benchmark labels, not human-annotated real-user ground truth.

## Model and training

- Base: `distilbert-base-uncased`
- Task: token classification, 3 labels
- Epochs: 3
- Learning rate: 3e-5; linear 10% warmup
- Batch size: 8
- Weight decay: 0.01
- Seed: 20260919
- Runtime: CUDA on NVIDIA RTX 2050
- Wall clock: 1,690 seconds (28.2 minutes)
- Best dev BIO F1: 0.9994972

## Held-out test results

| Metric | Result |
| --- | ---: |
| Token-level BIO F1 | 0.9986689 |
| Exact full-utterance span-set accuracy | 0.9979723 |
| Exact accuracy, one intent | 0.9965986 |
| Exact accuracy, two intents | 0.9982259 |
| Exact accuracy, three intents | 0.9985294 |

Against the same reconstructed test set, the parent pipeline's external `rules` splitter achieved 0.2767827 exact span-set accuracy. This benchmark comparison does not establish real-domain superiority.

## Files

- Checkpoint: `model/checkpoints/best/`
- Test metrics: `results/metrics.json`
- Rule comparison: `results/rule_based_comparison.json`
- Training curves: `results/training_curves.json`
- Evaluation errors: `results/error_analysis.md`

## Limitations

The benchmark is constructed from single-intent ATIS/SNIPS source utterances. Its near-perfect score mainly demonstrates that the model learns this construction pattern. Before production adoption, evaluate at least 100–200 manually boundary-annotated venue-domain compound queries.

## Recommended integration design

The production contract is `runtime.decomposer.split(query)`, which returns `Intent` objects containing `query`, `source_spans`, topic metadata, and a method name. Do not replace it immediately.

1. Add a new BERT-backed decomposer adapter alongside `Decomposer`; do not modify rule behavior.
2. Tokenize the original text with a fast offset-aware tokenizer. Map word-level BIO predictions to character offsets.
3. Convert each predicted intent span to the existing `Intent` shape: `query` is the exact source substring, `source_spans` is the corresponding character range, and `method` is `bert_intent_boundary`.
4. Reuse the existing entity/topic/constraint extraction functions from `Decomposer` only at integration time, so downstream retrieval sees the same fields.
5. Keep the current `spacy_dependency_rules` splitter as fallback. Fall back when BERT produces invalid BIO transitions, no span, excessive/overlapping spans, or confidence below a calibrated threshold.
6. Introduce configuration such as `CITEFRONTIER_PARSER=bert`, retaining `rules` and `spacy`. Shadow-run BERT first: log both outputs while serving the existing splitter.
7. Promote only after a venue-domain labeled evaluation shows a material improvement without regressions.
