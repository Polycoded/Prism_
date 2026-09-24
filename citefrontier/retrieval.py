"""Hybrid retrieval with optional in-memory corpus-graph expansion."""

from __future__ import annotations

from collections import defaultdict
from math import log
from typing import Literal, Protocol

from .models import CorpusChunk, Evidence
from .text import content_tokens


class CorpusGraph(Protocol):
    def neighbors(self, anchor_ids: tuple[str, ...]) -> tuple[str, ...]: ...


class InMemoryCorpusGraph:
    def __init__(self, edges: dict[str, tuple[str, ...]]):
        self._edges = {key: tuple(value) for key, value in edges.items()}

    def neighbors(self, anchor_ids: tuple[str, ...]) -> tuple[str, ...]:
        related: list[str] = []
        for anchor in anchor_ids:
            related.extend(self._edges.get(anchor, ()))
        return tuple(dict.fromkeys(related))


class HybridRetriever:
    """Sparse + token-semantic retrieval fused with reciprocal rank fusion.

    ``mode`` exists for a transparent local ablation.  It is deliberately not
    presented as a production switch: ``semantic`` is a deterministic token
    overlap proxy, not an embedding model.
    """

    def __init__(
        self,
        chunks: tuple[CorpusChunk, ...],
        graph: CorpusGraph | None = None,
        mode: Literal["hybrid", "semantic"] = "hybrid",
    ):
        if mode not in {"hybrid", "semantic"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._graph = graph
        self._mode = mode
        self._tokens = {chunk.chunk_id: content_tokens(chunk.text) for chunk in chunks}
        self._document_frequency: defaultdict[str, int] = defaultdict(int)
        for chunk_id in self._chunks:
            for token in set(self._tokens[chunk_id]):
                self._document_frequency[token] += 1

    def _lexical_score(self, query_tokens: tuple[str, ...], chunk: CorpusChunk) -> float:
        chunk_tokens = self._tokens[chunk.chunk_id]
        if not chunk_tokens:
            return 0.0
        score = 0.0
        for token in query_tokens:
            if token in chunk_tokens:
                score += 1.0 + log((1 + len(self._chunks)) / (1 + self._document_frequency[token]))
        return score / (len(chunk_tokens) ** 0.5)

    def _semantic_score(self, query_tokens: tuple[str, ...], chunk: CorpusChunk) -> float:
        query_set = set(query_tokens)
        chunk_set = set(self._tokens[chunk.chunk_id])
        if not query_set or not chunk_set:
            return 0.0
        return len(query_set & chunk_set) / len(query_set | chunk_set)

    def _query_coverage(self, query_tokens: tuple[str, ...], chunk: CorpusChunk) -> float:
        query_set = set(query_tokens)
        if not query_set:
            return 0.0
        return len(query_set & set(self._tokens[chunk.chunk_id])) / len(query_set)

    @staticmethod
    def _rank(scores: dict[str, float]) -> dict[str, int]:
        return {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(
                sorted(scores.items(), key=lambda item: (-item[1], item[0])), start=1
            )
        }

    def search(self, query: str, retrieval_event_id: str, limit: int = 5) -> tuple[Evidence, ...]:
        query_tokens = content_tokens(query)
        semantic = {chunk_id: self._semantic_score(query_tokens, chunk) for chunk_id, chunk in self._chunks.items()}
        coverage = {chunk_id: self._query_coverage(query_tokens, chunk) for chunk_id, chunk in self._chunks.items()}
        semantic_rank = self._rank(semantic)
        if self._mode == "semantic":
            fused = {
                chunk_id: 1 / (60 + semantic_rank[chunk_id])
                for chunk_id in self._chunks
                if coverage[chunk_id] >= 0.5 and semantic[chunk_id] > 0
            }
            base_origin = "token_semantic_proxy"
        else:
            lexical = {
                chunk_id: self._lexical_score(query_tokens, chunk)
                for chunk_id, chunk in self._chunks.items()
            }
            lexical_rank = self._rank(lexical)
            fused = {
                chunk_id: (1 / (60 + lexical_rank[chunk_id])) + (1 / (60 + semantic_rank[chunk_id]))
                for chunk_id in self._chunks
                if coverage[chunk_id] >= 0.5 and (lexical[chunk_id] > 0 or semantic[chunk_id] > 0)
            }
            base_origin = "hybrid_rrf"
        base_ids = tuple(chunk_id for chunk_id, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit])
        candidates: dict[str, tuple[float, str, tuple[str, ...]]] = {
            chunk_id: (fused[chunk_id], base_origin, (chunk_id,)) for chunk_id in base_ids
        }

        if self._graph and base_ids:
            for related_id in self._graph.neighbors(base_ids[:2]):
                if related_id not in self._chunks:
                    continue
                anchor_score = max(fused.get(anchor_id, 0.0) for anchor_id in base_ids[:2])
                # A linked span may supply qualifying context that does not
                # lexically overlap the current wording.  Keep it below its
                # anchor but high enough to be evaluated in the top-k ablation.
                graph_score = anchor_score * 0.97
                existing = candidates.get(related_id)
                if existing is None or graph_score > existing[0]:
                    candidates[related_id] = (graph_score, "graph_one_hop", (base_ids[0], related_id))

        ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
        return tuple(
            Evidence(
                chunk=self._chunks[chunk_id],
                score=score,
                coverage=coverage[chunk_id],
                origin=origin,
                retrieval_event_id=retrieval_event_id,
                path=path,
            )
            for chunk_id, (score, origin, path) in ranked
        )


class DenseHybridRerankRetriever:
    """Production-oriented BM25 + embeddings + cross-encoder retrieval.

    This backend is intentionally separate from :class:`HybridRetriever`, the
    deterministic dependency-light validation baseline.  It is corpus-only,
    does not own any session state, and fails closed when its optional
    dependencies are unavailable.
    """

    def __init__(
        self,
        chunks: tuple[CorpusChunk, ...],
        graph: CorpusGraph | None = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_limit: int = 40,
    ):
        if candidate_limit < 2:
            raise ValueError("candidate_limit must be at least 2")
        try:
            import numpy as np
            from rank_bm25 import BM25Okapi
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError(
                "Install the retrieval extra: python -m pip install -e '.[retrieval]'"
            ) from exc

        self._np = np
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._ordered_ids = tuple(self._chunks)
        self._tokens = [list(content_tokens(self._chunks[chunk_id].text)) for chunk_id in self._ordered_ids]
        self._graph = graph
        self._candidate_limit = candidate_limit
        self._bm25 = BM25Okapi(self._tokens)
        self._embedder = SentenceTransformer(embedding_model_name)
        self._reranker = CrossEncoder(reranker_model_name)
        self._embeddings = self._embedder.encode(
            [self._chunks[chunk_id].text for chunk_id in self._ordered_ids],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @staticmethod
    def _coverage(query_tokens: tuple[str, ...], chunk_tokens: list[str]) -> float:
        query_set = set(query_tokens)
        if not query_set:
            return 0.0
        return len(query_set & set(chunk_tokens)) / len(query_set)

    def _top_indices(self, scores, limit: int) -> tuple[int, ...]:
        ranked = sorted(range(len(self._ordered_ids)), key=lambda index: (-float(scores[index]), self._ordered_ids[index]))
        return tuple(ranked[:limit])

    def search(self, query: str, retrieval_event_id: str, limit: int = 5) -> tuple[Evidence, ...]:
        query_tokens = content_tokens(query)
        if not query_tokens or not self._ordered_ids:
            return ()

        lexical_scores = self._bm25.get_scores(list(query_tokens))
        query_embedding = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        dense_scores = self._embeddings @ query_embedding
        candidate_count = min(self._candidate_limit, len(self._ordered_ids))
        lexical_indices = self._top_indices(lexical_scores, candidate_count)
        dense_indices = self._top_indices(dense_scores, candidate_count)

        fused: dict[str, float] = {}
        for rank, index in enumerate(lexical_indices, start=1):
            chunk_id = self._ordered_ids[index]
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (60 + rank)
        for rank, index in enumerate(dense_indices, start=1):
            chunk_id = self._ordered_ids[index]
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (60 + rank)

        base_ids = tuple(
            chunk_id for chunk_id, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:candidate_count]
        )
        candidates: dict[str, tuple[str, tuple[str, ...]]] = {
            chunk_id: ("dense_sparse_rrf", (chunk_id,)) for chunk_id in base_ids
        }
        if self._graph and base_ids:
            for anchor_id in base_ids[:2]:
                for related_id in self._graph.neighbors((anchor_id,)):
                    if related_id in self._chunks and related_id not in candidates:
                        candidates[related_id] = ("graph_one_hop_reranked", (anchor_id, related_id))

        candidate_ids = tuple(candidates)
        rerank_scores = self._reranker.predict(
            [(query, self._chunks[chunk_id].text) for chunk_id in candidate_ids],
            show_progress_bar=False,
        )
        reranked = sorted(
            zip(candidate_ids, rerank_scores, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )[:limit]
        token_lookup = {chunk_id: tokens for chunk_id, tokens in zip(self._ordered_ids, self._tokens, strict=True)}
        return tuple(
            Evidence(
                chunk=self._chunks[chunk_id],
                score=float(score),
                coverage=self._coverage(query_tokens, token_lookup[chunk_id]),
                origin=candidates[chunk_id][0],
                retrieval_event_id=retrieval_event_id,
                path=candidates[chunk_id][1],
            )
            for chunk_id, score in reranked
        )
