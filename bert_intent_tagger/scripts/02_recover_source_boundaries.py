"""Recover intent boundaries by exact matching against AGIF's source corpora.

MixATIS and MixSNIPS were built by joining original single-intent utterances.
AGIF ships those source corpora under data/ATIS and data/SNIPS. This script
accepts an example only when every declared intent can be aligned, in order, to
an exact source utterance. It never uses an LLM and never alters raw input.
"""

from __future__ import annotations

import json
import random
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONVERTED = ROOT / "data" / "converted"
RESULTS = ROOT / "results"


def parse_agif(path: Path) -> list[tuple[tuple[str, ...], str]]:
    records: list[tuple[tuple[str, ...], str]] = []
    token_rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) == 2:
            token_rows.append(fields[0])
        elif len(fields) == 1 and token_rows:
            records.append((tuple(token_rows), fields[0]))
            token_rows = []
        else:
            raise ValueError(f"Malformed record in {path}: {line!r}")
    if token_rows:
        raise ValueError(f"Unterminated record in {path}")
    return records


def source_index(source_records: list[tuple[tuple[str, ...], str]]) -> dict[tuple[str, str], list[tuple[str, ...]]]:
    index: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
    for tokens, intent in source_records:
        index[(intent, tokens[0])].append(tokens)
    return index


def recover_spans(tokens: tuple[str, ...], intents: tuple[str, ...], index: dict[tuple[str, str], list[tuple[str, ...]]]) -> tuple[tuple[int, int], ...] | None:
    """Return the sole exact ordered alignment; reject none or ambiguous alignments."""
    @lru_cache(maxsize=None)
    def search(intent_position: int, cursor: int) -> tuple[tuple[int, int], ...] | None | str:
        if intent_position == len(intents):
            return ()
        matches: set[tuple[tuple[int, int], ...]] = set()
        intent = intents[intent_position]
        for start in range(cursor, len(tokens)):
            for candidate in index.get((intent, tokens[start]), []):
                end = start + len(candidate)
                if end <= len(tokens) and tokens[start:end] == candidate:
                    tail = search(intent_position + 1, end)
                    if tail not in (None, "ambiguous"):
                        matches.add(((start, end - 1),) + tail)
                        if len(matches) > 1:
                            return "ambiguous"
        if not matches:
            return None
        return next(iter(matches))

    result = search(0, 0)
    return result if isinstance(result, tuple) else None


def render(tokens: list[str], tags: list[str]) -> str:
    output: list[str] = []
    for index, (token, tag) in enumerate(zip(tokens, tags, strict=True)):
        if tag == "B-INTENT":
            output.append("[" + token)
        else:
            output.append(token)
        if tag in {"B-INTENT", "I-INTENT"} and (index + 1 == len(tags) or tags[index + 1] in {"B-INTENT", "O"}):
            output[-1] += "]"
    return " ".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    args = parser.parse_args()
    split = args.split
    sources = {
        "mixatis_clean": source_index(parse_agif(RAW / f"AGIF_ATIS_{split}.txt")),
        "mixsnips_clean": source_index(parse_agif(RAW / f"AGIF_SNIPS_{split}.txt")),
    }
    composite_files = {
        "mixatis_clean": RAW / f"MixATIS_clean_{split}.txt",
        "mixsnips_clean": RAW / f"MixSNIPS_clean_{split}.txt",
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for dataset, path in composite_files.items():
        for row_number, (token_tuple, compound_intents) in enumerate(parse_agif(path)):
            intents = tuple(compound_intents.split("#"))
            spans = recover_spans(token_tuple, intents, sources[dataset])
            utterance_id = f"{dataset}_{split}_{row_number:05d}"
            if spans is None:
                rejected.append({"utterance_id": utterance_id, "reason": "no_unique_exact_source_alignment"})
                continue
            tags = ["O"] * len(token_tuple)
            for start, end in spans:
                tags[start] = "B-INTENT"
                tags[start + 1:end + 1] = ["I-INTENT"] * (end - start)
            accepted.append({
                "utterance_id": utterance_id, "tokens": list(token_tuple), "bio_tags": tags,
                "num_intents": len(intents), "source_dataset": dataset, "source_split": split,
                "boundary_source": "source_corpus_exact_match_v1",
                "boundary_confidence": "exact_source_match",
            })
    CONVERTED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (CONVERTED / f"source_corpus_exact_match_v1_{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    (CONVERTED / f"source_corpus_exact_match_v1_{split}_rejected.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rejected), encoding="utf-8")
    rng = random.Random(20260919)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        groups[(row["source_dataset"], row["num_intents"])].append(row)
    review: list[dict[str, Any]] = []
    for group in groups.values():
        rng.shuffle(group)
    # Proportional stratified allocation, with enough room for every nonempty group.
    for key, group in groups.items():
        quota = round(150 * len(group) / len(accepted))
        review.extend(group[:quota])
    review = review[:150]
    if split == "train":
        (RESULTS / "human_review_sample_source_match.jsonl").write_text(
            "".join(json.dumps({
                "utterance_id": row["utterance_id"], "source_dataset": row["source_dataset"],
                "num_intents": row["num_intents"], "original_utterance": " ".join(row["tokens"]),
                "rendered_spans": render(row["tokens"], row["bio_tags"]),
                "boundary_source": row["boundary_source"], "boundary_confidence": row["boundary_confidence"],
            }) + "\n" for row in review), encoding="utf-8")
    report = {
        "method": "exact ordered matching against the corresponding AGIF original single-intent split",
        "boundary_source": "source_corpus_exact_match_v1", "total": len(accepted) + len(rejected),
        "accepted": len(accepted), "rejected": len(rejected), "acceptance_rate": len(accepted) / (len(accepted) + len(rejected)),
        "accepted_strata": {f"{dataset}|{count}": value for (dataset, count), value in Counter((row["source_dataset"], row["num_intents"]) for row in accepted).items()},
        "human_review_rows": len(review) if split == "train" else 0,
    }
    (RESULTS / f"source_corpus_recovery_{split}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
