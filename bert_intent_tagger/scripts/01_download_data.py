"""Download untouched cleaned AGIF MixATIS and MixSNIPS files into data/raw."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE_ROOT = "https://raw.githubusercontent.com/LooperXX/AGIF/master/data"
DATASETS = ("MixATIS_clean", "MixSNIPS_clean")
SPLITS = ("train", "dev", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "bert-intent-tagger-reproducibility"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, str]] = []
    for dataset in DATASETS:
        for split in SPLITS:
            filename = f"{dataset}_{split}.txt"
            url = f"{SOURCE_ROOT}/{dataset}/{split}.txt"
            payload = download(url)
            path = RAW_DIR / filename
            path.write_bytes(payload)
            files.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "source_url": url,
                    "filename": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            print(f"Downloaded {filename}: {len(payload):,} bytes")

    manifest = {
        "download_date": date.today().isoformat(),
        "source_repository": "https://github.com/LooperXX/AGIF",
        "files": files,
    }
    (RAW_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RAW_DIR / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
