"""Fine-tune DistilBERT on exact-source-recovered intent-boundary BIO labels."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
CHECKPOINTS = ROOT / "model" / "checkpoints"
RESULTS = ROOT / "results"
LABELS = ["O", "B-INTENT", "I-INTENT"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    seed = 20260919
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_rows, dev_rows = read_jsonl(SPLITS / "train.jsonl"), read_jsonl(SPLITS / "dev.jsonl")

    def tokenize(rows: list[dict]) -> Dataset:
        dataset = Dataset.from_list(rows)
        def encode(batch):
            encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=128)
            labels = []
            for example_index, tags in enumerate(batch["bio_tags"]):
                word_ids = encoded.word_ids(batch_index=example_index); previous = None; aligned = []
                for word_id in word_ids:
                    aligned.append(-100 if word_id is None or word_id == previous else LABEL_TO_ID[tags[word_id]])
                    previous = word_id
                labels.append(aligned)
            encoded["labels"] = labels
            return encoded
        return dataset.map(encode, batched=True, remove_columns=dataset.column_names)

    train_dataset, dev_dataset = tokenize(train_rows), tokenize(dev_rows)
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID)
    def metrics(prediction):
        logits, labels = prediction
        predictions = np.argmax(logits, axis=-1)
        true_sequences, predicted_sequences = [], []
        for predicted, actual in zip(predictions, labels):
            true_sequences.append([LABELS[a] for p, a in zip(predicted, actual) if a != -100])
            predicted_sequences.append([LABELS[p] for p, a in zip(predicted, actual) if a != -100])
        return {"bio_f1": f1_score(true_sequences, predicted_sequences)}
    CHECKPOINTS.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    arguments = TrainingArguments(output_dir=str(CHECKPOINTS), eval_strategy="epoch", save_strategy="epoch", learning_rate=3e-5, per_device_train_batch_size=8, per_device_eval_batch_size=16, num_train_epochs=3, weight_decay=0.01, warmup_ratio=0.1, load_best_model_at_end=True, metric_for_best_model="bio_f1", greater_is_better=True, logging_strategy="epoch", seed=seed, fp16=torch.cuda.is_available(), report_to=[])
    trainer = Trainer(model=model, args=arguments, train_dataset=train_dataset, eval_dataset=dev_dataset, tokenizer=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=metrics)
    trainer.train()
    final_dir = CHECKPOINTS / "best"
    trainer.save_model(str(final_dir)); tokenizer.save_pretrained(str(final_dir))
    report = {"boundary_source": "source_corpus_exact_match_v1", "base_model": "distilbert-base-uncased", "labels": LABELS, "seed": seed, "device": "cuda" if torch.cuda.is_available() else "cpu", "wall_clock_seconds": time.perf_counter() - started, "best_metric": trainer.state.best_metric, "best_checkpoint": trainer.state.best_model_checkpoint, "log_history": trainer.state.log_history}
    (RESULTS / "training_curves.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (final_dir / "README.md").write_text("# Model checkpoint\n\nDistilBERT intent-boundary tagger trained on `source_corpus_exact_match_v1` labels. Consult `results/metrics.json` after evaluation.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
