# DistilBERT intent-boundary tagger - technical report

## Purpose

This standalone model detects boundaries between requests in a compound utterance. It predicts `B-INTENT`, `I-INTENT`, and `O` for connective or glue tokens.

## Training data

MixATIS and MixSNIPS do not directly publish intent spans. Boundaries were recovered by unique, complete, ordered token matching against AGIF's corresponding original single-intent splits. Ambiguous or unmatched records were rejected.

| Partition | Accepted examples |
|---|---:|
| Train | 46,697 |
| Dev | 2,913 |
| Test | 2,959 |

Every record is marked `source_corpus_exact_match_v1`. These are reconstructed benchmark labels, not human-annotated real-user ground truth.

## Model and reconstructed-benchmark result

- Base: `distilbert-base-uncased`
- Three epochs, learning rate 3e-5, 10% linear warmup
- Batch size 8, weight decay 0.01, seed 20260919
- CUDA training on an NVIDIA RTX 2050 in 1,690 seconds
- Best dev BIO F1: 0.9994972
- Test BIO F1: 0.9986689
- Exact test span-set accuracy: 0.9979723
- Rules exact accuracy on the same 2,959 examples: 0.2767827

The corrected error analysis contains six BERT failures: rules are correct in two and both systems are wrong in four.

## Guarded integration

`prototype/bert_decomposition.py` provides:

- lazy local-only checkpoint loading;
- fast-tokenizer word alignment and character offsets;
- BIO-transition, truncation, overlap, span-count, content, and confidence validation;
- conversion to the existing `Intent` contract;
- shared entity, topic, and constraint enrichment through the existing decomposer;
- `bert`, `shadow`, and `auto` modes;
- automatic fallback to unchanged spaCy/rule behavior;
- per-request diagnostics in `intents_decomposed` telemetry.

The checkpoint is 265,473,092 bytes. On the latest local 200-case venue run, cold load plus first inference was 6.34 seconds; warm CPU inference measured 25.12 ms p50 and 35.43 ms p95. Startup preload keeps this one-time load outside the first request when BERT mode is enabled.

## Venue-domain checkpoint

`results/venue_domain_review.jsonl` contains 200 explicitly labeled synthetic venue cases prepared for review, including 40 punctuation-free ASR-style compounds and 40 single-intent coordination hard negatives. It is not represented as manual or real-user ground truth.

| Splitter | Exact spans |
|---|---:|
| Rules | 115/200 (57.5%) |
| BERT accepted output | 182/200 (91.0%) |
| Guarded auto | 177/200 (88.5%) |

At threshold 0.90, all 102 accepted BERT outputs matched the current synthetic labels; 18 cases fell back, mainly elliptical noun fragments. This is a 22.5 percentage-point gain over rules on the curated set, but `promotion_eligible` remains false until a person independently reviews and corrects the boundaries.

## Remaining limitation

The recovered benchmark and curated venue set are constructed data. Production promotion requires independent annotation of the venue review file and shadow-mode evidence from representative live utterances. Until then, rules remain the served default.
