"""Typed records used by the deterministic validation prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    WAIT = "wait"
    PROVISIONAL_RETRIEVE = "provisional_retrieve"
    COMMIT_RETRIEVE = "commit_retrieve"
    SUPPRESS = "suppress"


@dataclass(frozen=True)
class CorpusChunk:
    """A citeable corpus unit. `chunk_id` is the only legal citation ID."""

    chunk_id: str
    doc_id: str
    section: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Intent:
    key: str
    query: str


@dataclass(frozen=True)
class TranscriptEvent:
    timestamp_s: float
    text: str
    is_final: bool = False


@dataclass(frozen=True)
class ControllerAction:
    decision: Decision
    intents: tuple[Intent, ...]
    reason: str
    invalidates_provisional: bool = False


@dataclass(frozen=True)
class Evidence:
    """Retrieved corpus evidence with event provenance for auditability."""

    chunk: CorpusChunk
    score: float
    coverage: float
    origin: str
    retrieval_event_id: str
    path: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    claim_id: str
    intent_key: str
    text: str
    citations: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnswerVersion:
    version: int
    parent_version: int | None
    claims: tuple[Claim, ...]
    changed_claim_ids: tuple[str, ...]

    def render(self, bullets: bool = False) -> str:
        prefix = "- " if bullets else ""
        return "\n".join(
            f"{prefix}{claim.text}"
            + (f" [{', '.join(claim.citations)}]" if claim.citations else "")
            for claim in self.claims
        )


@dataclass(frozen=True)
class EngineOutput:
    action: ControllerAction
    answer: AnswerVersion | None
    rendered_answer: str | None
    retrieval_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    timestamp_s: float
    session_id: str
    payload: dict[str, Any]
