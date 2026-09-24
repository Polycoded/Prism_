"""Publish exact-source-recovered records in the original train/dev/test partitions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "data" / "converted"
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"


def main() -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"boundary_source": "source_corpus_exact_match_v1"}
    for split in ("train", "dev", "test"):
        source = CONVERTED / f"source_corpus_exact_match_v1_{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
        (SPLITS / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary[split] = {"examples": len(rows), "num_intents": dict(Counter(row["num_intents"] for row in rows))}
        print(f"{split}: {len(rows)}")
    (RESULTS / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
