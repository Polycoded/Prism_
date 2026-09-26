"""Entailment verification, claim-slot selection, and citation guarding.

Everything in this module is deterministic and dependency-light.  The only
optional part is :class:`NliEntailmentVerifier`, which needs the NLI extra and
fails closed when it is not installed — it never silently degrades to the
extractive verifier.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from math import exp
from typing import Iterable, Protocol, Sequence, runtime_checkable

# Model downloads must stay reproducible for the optional NLI verifier.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .models import Evidence
from .text import content_tokens, normalized

DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
DEFAULT_NLI_THRESHOLD = 0.70


@dataclass(frozen=True)
class EntailmentResult:
    entailed: bool
    score: float
    label: str


@runtime_checkable
class EntailmentVerifier(Protocol):
    def verify(self, claim: str, evidence_text: str) -> EntailmentResult: ...


class ExtractiveEntailmentVerifier:
    """Entailment iff the normalized claim is a substring of the evidence.

    This is the exact-extractive guarantee every cited claim relies on: the
    claim text is a verbatim copy of part of the cited chunk.
    """

    def verify(self, claim: str, evidence_text: str) -> EntailmentResult:
        claim_normalized = normalized(claim)
        if not claim_normalized:
            return EntailmentResult(entailed=False, score=0.0, label="empty_claim")
        if claim_normalized in normalized(evidence_text):
            return EntailmentResult(entailed=True, score=1.0, label="exact_extractive")
        return EntailmentResult(entailed=False, score=0.0, label="not_entailed")


class NliEntailmentVerifier:
    """Optional cross-encoder NLI verifier.

    Scores the pair ``(evidence_text, claim)`` and requires the entailment
    label with a softmax probability of at least ``threshold``.  Fails closed
    with a clear message when its optional dependencies are missing.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        threshold: float = DEFAULT_NLI_THRESHOLD,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError(
                "NliEntailmentVerifier needs the optional NLI extra: "
                "python -m pip install sentence-transformers torch"
            ) from exc

        self._model = CrossEncoder(model_name)
        self._model_name = model_name
        self._threshold = threshold
        labels = dict(getattr(self._model, "id2label", {}) or {})
        self._entailment_index = next(
            (int(index) for index, label in labels.items() if "entail" in str(label).lower()),
            None,
        )
        if self._entailment_index is None:
            raise RuntimeError(
                f"NLI model {model_name!r} exposes no entailment label "
                f"(labels: {sorted(map(str, labels.values()))}); failing closed"
            )

    @property
    def threshold(self) -> float:
        return self._threshold

    def verify(self, claim: str, evidence_text: str) -> EntailmentResult:
        predicted = self._model.predict([(evidence_text, claim)])
        probabilities = _softmax([float(value) for value in predicted[0]])
        index = self._entailment_index
        score = probabilities[index]
        label = str(dict(self._model.id2label)[index])
        entailed = score >= self._threshold and "entail" in label.lower()
        return EntailmentResult(entailed=entailed, score=score, label=label)


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    largest = max(values)
    exponentials = [exp(value - largest) for value in values]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


@dataclass(frozen=True)
class ClaimEvidenceSelection:
    evidence: Evidence | None
    verification: EntailmentResult | None
    evaluated_candidate_ids: tuple[str, ...]


class ClaimEvidenceSelector:
    """Fill one claim slot from retrieved candidates, or abstain."""

    def __init__(self, verifier: EntailmentVerifier | None = None):
        self._verifier: EntailmentVerifier = (
            verifier if verifier is not None else ExtractiveEntailmentVerifier()
        )

    @property
    def verifier(self) -> EntailmentVerifier:
        return self._verifier

    def select(
        self, claim: str, candidates: tuple[Evidence, ...]
    ) -> ClaimEvidenceSelection:
        evaluated_candidate_ids = tuple(candidate.chunk.chunk_id for candidate in candidates)
        best_key: tuple[float, int] | None = None
        best: tuple[Evidence, EntailmentResult] | None = None
        for index, candidate in enumerate(candidates):
            result = self._verifier.verify(claim, candidate.chunk.text)
            if not result.entailed:
                continue
            # Highest score wins; the earlier candidate wins a score tie.
            key = (result.score, -index)
            if best_key is None or key > best_key:
                best_key, best = key, (candidate, result)
        if best is None:
            return ClaimEvidenceSelection(
                evidence=None, verification=None, evaluated_candidate_ids=evaluated_candidate_ids
            )
        return ClaimEvidenceSelection(
            evidence=best[0], verification=best[1], evaluated_candidate_ids=evaluated_candidate_ids
        )


@dataclass(frozen=True)
class CitationGuardDecision:
    accepted_citation_ids: tuple[str, ...]
    rejected_citation_ids: tuple[str, ...]


class CitationCandidateGuard:
    """Accept only citations that are chunk_ids of the current candidate set.

    This is the structural reason a fabricated citation cannot reach an
    answer: anything outside the candidate tuple passed in here is rejected,
    no matter who proposed it.
    """

    def enforce(
        self,
        proposed_citation_ids: Iterable[str],
        candidates: tuple[Evidence, ...],
    ) -> CitationGuardDecision:
        allowed = {candidate.chunk.chunk_id for candidate in candidates}
        accepted: list[str] = []
        rejected: list[str] = []
        for citation_id in proposed_citation_ids:
            if citation_id in allowed:
                accepted.append(citation_id)
            else:
                rejected.append(citation_id)
        return CitationGuardDecision(tuple(accepted), tuple(rejected))


_QUANTITY = re.compile(r"\b(\d+)[ -](months?|years?|days?|people)\b")
_GENERIC_PROPERTY_TOKENS = frozenset(
    {"period", "month", "months", "year", "years", "day", "days", "people"}
)


def _entity_tokens(evidence: Evidence) -> frozenset[str]:
    entity = evidence.chunk.metadata.get("entity") or evidence.chunk.doc_id
    return frozenset(content_tokens(entity))


def _quantities(evidence: Evidence) -> frozenset[str]:
    return frozenset(match.group(0).lower() for match in _QUANTITY.finditer(evidence.chunk.text))


def conflicting_evidence(
    top: Evidence, candidates: Sequence[Evidence]
) -> tuple[Evidence, ...]:
    """Alternatives that compete with ``top`` on the same numeric quantity.

    Two chunks conflict when they describe the same entity and section root,
    share a non-generic property, and state different quantities.  This is the
    deterministic core of the conflict check that Module 05 §7 describes; the
    live runtime in Module 06 owns the full claim builder around it.
    """
    top_quantities = _quantities(top)
    if not top_quantities:
        return ()
    top_entity = _entity_tokens(top)
    top_tokens = frozenset(content_tokens(top.chunk.text))
    top_root = top.chunk.section.split(".")[0]
    conflicting: list[Evidence] = []
    for candidate in candidates:
        if candidate.chunk.chunk_id == top.chunk.chunk_id:
            continue
        candidate_quantities = _quantities(candidate)
        if not candidate_quantities or candidate_quantities == top_quantities:
            continue
        if _entity_tokens(candidate) != top_entity:
            continue
        if candidate.chunk.section.split(".")[0] != top_root:
            continue
        shared_property = (
            top_tokens & frozenset(content_tokens(candidate.chunk.text)) - _GENERIC_PROPERTY_TOKENS
        ) - top_entity
        if shared_property:
            conflicting.append(candidate)
    return tuple(conflicting)
