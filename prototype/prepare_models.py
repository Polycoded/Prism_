"""Explicit asset provisioning; inference itself is local-files-only."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from .search import EMBED_MODEL, EMBED_REVISION, RERANK_MODEL, RERANK_REVISION


def main():
    models = []
    for repo, revision in [(EMBED_MODEL, EMBED_REVISION), (RERANK_MODEL, RERANK_REVISION)]:
        path = snapshot_download(repo, revision=revision, ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.ot", "pytorch_model.bin"])
        models.append({"repository": repo, "revision": revision})
        print(f"Provisioned {repo} @ {revision}: {path}")
    Path(__file__).with_name("models-manifest.json").write_text(json.dumps({"models": models, "parser": "en_core_web_sm==3.8.0"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
