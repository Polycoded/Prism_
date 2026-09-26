"""Ephemeral session state and claim/answer synthesis (Module 05).

A session holds only what the current conversation needs: staged provisional
evidence, committed evidence per intent, and the answer version history.  It is
discarded on disconnect.  Two rules are enforced here rather than trusted:

* provisional evidence can never become a committed claim, and
* a claim can only cite a chunk that is committed for its own intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .grounding import CitationCandidateGuard
from .models import AnswerVersion, Claim, Evidence, Intent
from .text import content_tokens

ELLIPTICAL_LEAD_INS: tuple[str, ...] = (
    "what about",
    "what does that",
    "what does it",
    "and what",
    "how about",
    "is that",
    "yes",
    "no",
    "i do not",
    "i dont",
)

# A lead-in that is all that is left of the utterance is a pure reference back
# to the previous turn; anything else names something of its own.
_DEICTIC_TOKENS = frozenset(
    {"that", "this", "it", "its", "they", "them", "their", "those", "these", "one", "there", "then"}
)

ELLIPTICAL_TOKEN_LIMIT = 5


class CitationValidationError(RuntimeError):
    """A claim tried to cite evidence that is not committed for its intent."""


class SessionSynthesisError(RuntimeError):
    """A synthesis primitive was called with state that cannot satisfy it."""


def abstention_text(query: str) -> str:
    return f"I could not verify: {query}."


def is_elliptical(query: str) -> bool:
    """Follow-up that cannot stand on its own as a search query."""
    if len(content_tokens(query)) < ELLIPTICAL_TOKEN_LIMIT:
        return True
    return _matched_lead_in(query) is not None


def _matched_lead_in(query: str) -> str | None:
    lowered = query.strip().lower()
    return next((lead for lead in ELLIPTICAL_LEAD_INS if lowered.startswith(lead)), None)


def _is_pure_deictic(query: str) -> bool:
    lead_in = _matched_lead_in(query)
    if lead_in is None:
        return False
    return set(content_tokens(query.strip().lower()[len(lead_in) :])) <= _DEICTIC_TOKENS


@dataclass
class EphemeralSession:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    provisional: dict[str, tuple[Evidence, ...]] = field(default_factory=dict)
    committed: dict[str, tuple[Evidence, ...]] = field(default_factory=dict)
    intent_by_key: dict[str, Intent] = field(default_factory=dict)
    versions: list[AnswerVersion] = field(default_factory=list)
    invalidated_provisionals: set[str] = field(default_factory=set)
    root_query: str = ""
    citation_guard: CitationCandidateGuard = field(default_factory=CitationCandidateGuard)

    def __post_init__(self) -> None:
        # Dialogue context and staged intents are private bookkeeping: they are
        # not part of the documented session surface, and the staged intent is
        # deliberately never promoted to intent_by_key (provisional isolation).
        self._remembered_turns: list[tuple[Intent, ...]] = []
        self._staged_intents: dict[str, Intent] = {}

    @property
    def latest(self) -> AnswerVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def has_answer(self) -> bool:
        return self.latest is not None

    def stage_provisional(
        self, event_id: str, intent: Intent, evidence: tuple[Evidence, ...]
    ) -> None:
        """Stage tentative evidence.  Nothing here is ever committed."""
        self.provisional[event_id] = tuple(evidence)
        self._staged_intents[event_id] = intent

    def invalidate_provisionals(self) -> tuple[str, ...]:
        """Discard every staged retrieval and remember the ids as invalidated."""
        invalidated = tuple(self.provisional)
        self.invalidated_provisionals.update(invalidated)
        self.provisional.clear()
        for event_id in invalidated:
            self._staged_intents.pop(event_id, None)
        return invalidated

    def commit(self, intent: Intent, evidence: tuple[Evidence, ...]) -> None:
        self.committed[intent.key] = tuple(evidence)
        self.intent_by_key[intent.key] = intent

    def remember_committed_turn(self, intents: tuple[Intent, ...]) -> None:
        """Remember a committed turn as user intent text only.

        Assistant or answer text is never accepted here, so dialogue context
        can only ever be built from what the user asked.
        """
        self._remembered_turns.append(tuple(intents))

    def contextualize(
        self, intent: Intent, max_prior_turns: int = 1
    ) -> tuple[str, tuple[str, ...]]:
        """Expand an elliptical follow-up with prior *user* intent text.

        Returns the query to search with and the intent keys whose text was
        used.  A non-elliptical intent is returned unchanged.
        """
        if max_prior_turns <= 0:
            return intent.query, ()
        prior_turns = self._remembered_turns[-max_prior_turns:]
        prior_intents = tuple(remembered for turn in prior_turns for remembered in turn)
        if not prior_intents or not is_elliptical(intent.query):
            return intent.query, ()
        context = " ".join(remembered.query for remembered in prior_intents)
        context_intent_keys = tuple(remembered.key for remembered in prior_intents)
        if _is_pure_deictic(intent.query):
            # "yes" / "what about that" refer back to the whole prior request.
            return context, context_intent_keys
        # A short but specific follow-up keeps its own wording as a sidecar.
        return f"{intent.query} {context}", context_intent_keys

    def synthesize_initial(self, root_query: str) -> AnswerVersion:
        previous = self.latest
        if root_query and not self.root_query:
            self.root_query = root_query
        version = 1 if previous is None else previous.version + 1
        claims = tuple(
            self._claim_for(intent_key, self.committed[intent_key], f"c-{intent_key}")
            for intent_key in self.committed
        )
        answer = AnswerVersion(
            version=version,
            parent_version=None if previous is None else previous.version,
            claims=claims,
            changed_claim_ids=tuple(claim.claim_id for claim in claims),
        )
        self.versions.append(answer)
        return answer

    def refine(
        self, target_intent_key: str, delta_intent: Intent, evidence: tuple[Evidence, ...]
    ) -> AnswerVersion:
        previous = self.latest
        if previous is None:
            raise SessionSynthesisError("refine() requires an existing answer")
        if target_intent_key not in self.intent_by_key:
            raise SessionSynthesisError(f"{target_intent_key!r} is not a current intent")
        if not any(claim.intent_key == target_intent_key for claim in previous.claims):
            raise SessionSynthesisError(
                f"no current claim for intent {target_intent_key!r} to refine"
            )
        version = previous.version + 1
        # The delta evidence is committed before the claim is built so that the
        # citation check in _claim_for sees it as committed.
        self.committed[target_intent_key] = tuple(evidence)
        claims: list[Claim] = []
        changed_claim_ids: list[str] = []
        for claim in previous.claims:
            if claim.intent_key != target_intent_key:
                # Untouched claims keep their id, text, citations, and object.
                claims.append(claim)
                continue
            refined = self._claim_for(
                target_intent_key, tuple(evidence), f"{claim.claim_id}-v{version + 1}"
            )
            claims.append(refined)
            changed_claim_ids.append(refined.claim_id)
        answer = AnswerVersion(
            version=version,
            parent_version=previous.version,
            claims=tuple(claims),
            changed_claim_ids=tuple(changed_claim_ids),
        )
        self.versions.append(answer)
        return answer

    def _claim_for(
        self, intent_key: str, evidence: tuple[Evidence, ...], claim_id: str
    ) -> Claim:
        intent = self.intent_by_key.get(intent_key)
        query = intent.query if intent is not None else intent_key
        usable = tuple(
            item for item in evidence if item.retrieval_event_id not in self.invalidated_provisionals
        )
        if usable:
            top = usable[0]
            guard = self.citation_guard.enforce((top.chunk.chunk_id,), usable)
            if guard.accepted_citation_ids:
                citation_id = guard.accepted_citation_ids[0]
                self._require_committed_citation(intent_key, citation_id)
                return Claim(
                    claim_id=claim_id,
                    intent_key=intent_key,
                    text=top.chunk.text,
                    citations=(citation_id,),
                    evidence_event_ids=tuple(
                        dict.fromkeys(item.retrieval_event_id for item in usable)
                    ),
                )
        return Claim(
            claim_id=claim_id,
            intent_key=intent_key,
            text=abstention_text(query),
            citations=(),
            evidence_event_ids=(),
        )

    def _require_committed_citation(self, intent_key: str, chunk_id: str) -> None:
        committed_chunk_ids = {
            item.chunk.chunk_id for item in self.committed.get(intent_key, ())
        }
        if chunk_id not in committed_chunk_ids:
            raise CitationValidationError(
                f"claim for intent {intent_key!r} cites {chunk_id!r}, which is not in the "
                f"committed evidence set for that intent"
            )
