"""Build corpus.json from UTF-8 Markdown; labels never enter this path."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def ingest(source_dir: Path, output: Path):
    chunks, documents = [], []
    for path in sorted(source_dir.glob("*.md")):
        source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        doc_id = path.stem
        entity = doc_id.replace("_", " ")
        digest = hashlib.sha256(source.encode()).hexdigest()
        headings = list(re.finditer(r"(?m)^## (.+)\n", source))
        for index, match in enumerate(headings):
            end = headings[index+1].start() if index+1 < len(headings) else len(source)
            raw = source[match.end():end]
            body = raw.strip()
            start = match.end() + len(raw) - len(raw.lstrip())
            if not body:
                continue
            chunks.append(dict(chunk_id=f"{doc_id} §{match[1]}", doc_id=doc_id, section=match[1], text=body,
                               metadata={"entity": entity, "title": source.splitlines()[0].lstrip("# "),
                                         "source": path.name, "source_hash": digest, "start": str(start), "end": str(start+len(body))}))
        documents.append({"file": path.name, "sha256": digest, "characters": len(source)})
    if not chunks or len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Require nonempty sources and unique document/section IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"schema_version": 1, "format": "UTF-8 Markdown, LF normalized; character offsets",
                "corpus_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "documents": documents,
                "chunks": len(chunks), "chunking": "one complete section per chunk, no truncation"}
    output.with_name("corpus-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path(__file__).parent / "corpus" / "sources")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "corpus.json")
    args = parser.parse_args()
    print(f"Indexed {len(ingest(args.sources, args.output))} passages")


if __name__ == "__main__":
    main()
