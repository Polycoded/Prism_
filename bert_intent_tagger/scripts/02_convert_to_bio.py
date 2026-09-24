"""Inspect AGIF records before any intent-boundary BIO conversion.

AGIF's cleaned MixATIS/MixSNIPS data provides slot BIO labels and a compound
utterance-level intent list, but it does not provide the original sentence
boundaries or token-to-intent alignment needed for gold intent-boundary labels.
This script intentionally fails rather than silently creating heuristic labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONVERTED_DIR = PROJECT_ROOT / "data" / "converted"


def parse_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    token_rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            if token_rows:
                raise ValueError(f"Missing intent line before blank line in {path}")
            continue
        fields = line.split()
        if len(fields) == 2:
            token_rows.append((fields[0], fields[1]))
            continue
        if len(fields) != 1 or not token_rows:
            raise ValueError(f"Malformed record in {path}: {line!r}")
        records.append(
            {
                "tokens": [token for token, _ in token_rows],
                "slot_tags": [slot for _, slot in token_rows],
                "intents": fields[0].split("#"),
            }
        )
        token_rows = []
    if token_rows:
        raise ValueError(f"Unterminated record in {path}")
    return records


def main() -> None:
    files = sorted(RAW_DIR.glob("Mix*_clean_*.txt"))
    if not files:
        raise SystemExit("No raw files found. Run scripts/01_download_data.py first.")

    preview: list[dict[str, object]] = []
    distribution: Counter[int] = Counter()
    for path in files:
        records = parse_records(path)
        for record in records:
            distribution[len(record["intents"])] += 1
            if len(preview) < 25:
                preview.append({"source_file": path.name, **record})

    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = CONVERTED_DIR / "raw_format_preview.jsonl"
    preview_path.write_text(
        "".join(json.dumps(record) + "\n" for record in preview), encoding="utf-8"
    )
    print(f"Parsed {sum(distribution.values()):,} records; intent-count distribution: {dict(distribution)}")
    print(f"Wrote raw-format preview to {preview_path}")
    raise SystemExit(
        "Cannot create gold intent-boundary BIO tags: AGIF raw records lack "
        "token-to-intent span alignment. Supply gold boundaries or approve a "
        "specific heuristic/annotation policy before conversion."
    )


if __name__ == "__main__":
    main()
