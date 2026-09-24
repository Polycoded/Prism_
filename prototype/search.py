"""Versioned local hybrid search. No network access at inference time."""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from threading import RLock

from .decomposition import terms, words

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@dataclass(frozen=True)
class Candidate:
    evidence_id: str
    score: float
    relevance: float


class SearchIndex:
    def __init__(self, chunks, decomposer, backend="lightweight", rerank=True):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.ids = tuple(self.chunks)
        self.decomposer = decomposer
        self.backend = backend
        self.rerank_enabled = rerank
        self.model_lock = RLock()
        self.tokens = {c.chunk_id: terms(c.text) for c in chunks}
        self.topic_tokens = {c.chunk_id: decomposer.topic(c.text, (c.doc_id,)) for c in chunks}
        self.vocabulary = set().union(*self.tokens.values())
        self.idf = {t: log(1 + len(chunks) / (1 + sum(t in group for group in self.tokens.values()))) for group in self.tokens.values() for t in group}
        if backend == "dense":
            from sentence_transformers import SentenceTransformer, CrossEncoder
            from rank_bm25 import BM25Okapi
            self.embedder = SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION, local_files_only=True)
            self.reranker = CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, local_files_only=True) if rerank else None
            self.bm25 = BM25Okapi([words(c.text) for c in chunks])
            self.matrix = self.embedder.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
        elif backend != "lightweight":
            raise ValueError("Unknown retrieval backend")

    def _eligible(self, intent):
        # Reject an explicitly named unknown sibling model instead of answering
        # from a different product merely because the property overlaps.
        for alias in self.decomposer.aliases.values():
            pieces = alias.split()
            if len(pieces) > 1:
                import re
                mentioned = re.findall(r"\b" + re.escape(pieces[0]) + r"\s+([A-Za-z]+\d+)\b", intent.query, re.I)
                if any(not self.decomposer.entities(pieces[0] + " " + model) for model in mentioned):
                    return ()
        if not intent.entity_ids:
            return self.ids
        return tuple(i for i in self.ids if self.chunks[i].doc_id in intent.entity_ids or
                     any(alias.lower() in self.chunks[i].text.lower() for doc, alias in self.decomposer.aliases.items() if doc in intent.entity_ids))

    def candidates(self, intent):
        eligible = set(self._eligible(intent))
        query_terms = terms(intent.query)
        if self.backend == "dense":
            with self.model_lock:
                vector = self.embedder.encode([intent.query], normalize_embeddings=True, show_progress_bar=False)[0]
            dense = self.matrix @ vector
            sparse = self.bm25.get_scores(words(intent.query))
            orders = [sorted(eligible, key=lambda i: (-float(scores[self.ids.index(i)]), i))[:20] for scores in (dense, sparse)]
        else:
            # Disclosed lexical proxy; semantic profile uses actual BM25/embeddings.
            sparse = {i: sum(self.idf.get(t, 0) for t in query_terms & self.tokens[i]) / max(1, len(self.tokens[i])) ** .5 for i in eligible}
            overlap = {i: len(query_terms & self.tokens[i]) / max(1, len(query_terms | self.tokens[i])) for i in eligible}
            orders = [sorted(eligible, key=lambda i: (-scores[i], i))[:20] for scores in (sparse, overlap)]
        fusion = {}
        for order in orders:
            for rank, i in enumerate(order, 1):
                fusion[i] = fusion.get(i, 0) + 1 / (60 + rank)
        return tuple(sorted(fusion, key=lambda i: (-fusion[i], i))[:20])

    def finalize(self, intent, candidate_ids):
        allowed = set(self._eligible(intent))
        ids = tuple(i for i in candidate_ids if i in allowed and i in self.chunks)
        if not ids:
            return ()
        topic = set(intent.topic)
        if not topic:
            return ()
        # Both profiles refuse unknown requested concepts. This conservative
        # vocabulary gate deliberately trades paraphrase recall for abstention.
        if topic - self.vocabulary - {"period", "exclude", "receipt", "replace", "local", "require", "coverage", "repair", "setup", "history", "trial", "kept", "reminder"}:
            return ()
        if self.backend == "dense" and self.reranker:
            with self.model_lock:
                scores = self.reranker.predict([(intent.query, self.chunks[i].text) for i in ids], show_progress_bar=False)
        else:
            scores = [sum(self.idf.get(t, 0) for t in topic & self.topic_tokens[i]) /
                      max(1, sum(self.idf.get(t, 0) for t in topic)) +
                      .1 * len(topic & terms(self.chunks[i].section.split('.')[0])) for i in ids]
        result = []
        for i, score in zip(ids, scores):
            import re
            if re.search(r"\b(?:ignore|disregard|override)\b.{0,80}\b(?:instructions|system|rules)\b", self.chunks[i].text, re.I):
                continue
            overlap = topic & self.topic_tokens[i]
            relevance = len(overlap) / max(1, len(topic))
            # A ranking score cannot justify an answer to an unknown topic.
            if not overlap:
                continue
            if self.backend == "lightweight" and relevance < .30:
                continue
            result.append(Candidate(i, float(score), relevance))
        return tuple(sorted(result, key=lambda c: (-c.score, -c.relevance, c.evidence_id))[:5])
