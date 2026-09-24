# DistilBERT intent-boundary tagger

This directory contains the reproducible training code, compact evaluation evidence,
and promoted checkpoint used by CiteFrontier. It predicts `B-INTENT`, `I-INTENT`,
and `O` tags for compound text utterances. It is a boundary detector, not the live
retrieval corpus and not an answer-generation model.

## Data provenance

Training boundaries were reconstructed from the public AGIF MixATIS/MixSNIPS
corpora by exact token-sequence alignment with their original single-intent ATIS
and SNIPS sources. Only unique, complete ordered alignments were accepted;
ambiguous records were rejected. The boundary source is identified as
`source_corpus_exact_match_v1`, not independently annotated gold data.

Raw, converted, and train/dev/test files are excluded from the repository
because they are external training material. Dataset counts, provenance, and
metrics are retained in `results/`; the training pipeline remains in `scripts/`.

## Training configuration and results

- Base model: `distilbert-base-uncased`
- Three epochs, learning rate 3e-5, batch size 8, weight decay 0.01
- Warmup 10%; seed 20260919
- 46,697 train, 2,913 development, and 2,959 test examples
- Reconstructed-test token BIO F1: `0.9986689`
- Reconstructed-test exact intent-span-set accuracy: `0.9979723`

These scores measure reconstructed ATIS/SNIPS boundaries; they do not establish
venue-domain or live-retrieval accuracy.

The separate 200-case curated synthetic venue review measured raw accepted BERT
spans at 91.0% exact with 100% precision among accepted predictions. Guarded
`auto` mode measured 88.5% exact versus 57.5% for rules and falls back to spaCy
when confidence or structural validation fails. This is locally measured
development evidence on an explicitly synthetic review set.

## Repository contents

- `scripts/01_download_data.py` — fetch source corpora
- `scripts/02_convert_to_bio.py` — convert source format
- `scripts/02_recover_source_boundaries.py` — exact boundary reconstruction
- `scripts/03_split_dataset.py` — deterministic splits
- `scripts/04_train.py` — model training
- `scripts/05_evaluate.py` — held-out evaluation
- `scripts/06_compare_to_rule_based.py` — rule baseline
- `scripts/07_venue_domain_evaluate.py` — venue-domain review
- `results/metrics.json` — reconstructed-test metrics
- `results/training_curves.json` — epoch metrics and environment
- `results/split_summary.json` — split counts
- `results/rule_based_comparison.json` — boundary baseline
- `results/venue_domain_evaluation.json` — guarded venue review
- `model/checkpoints/best/` — promoted local checkpoint

## Reproduce

Create a Python 3.12 environment, install `requirements.txt`, obtain the source
datasets through the download script, then run:

```powershell
python scripts/02_convert_to_bio.py
python scripts/02_recover_source_boundaries.py
python scripts/03_split_dataset.py
python scripts/04_train.py
python scripts/05_evaluate.py
python scripts/06_compare_to_rule_based.py
python scripts/07_venue_domain_evaluate.py
```

The runtime loads `model/checkpoints/best` locally in guarded `auto` mode when
the parser is configured to `bert`, `shadow`, or `auto`. No model or training
dataset is downloaded during inference.
