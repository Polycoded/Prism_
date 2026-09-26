"""Core-library orchestration: controller + retriever + session (Module 05).

The engine owns the flow a live runtime replays, with no transport of its own.
It drives the frozen Module 04 controller, calls the frozen Module 02 retriever
API ``search(query, retrieval_event_id, limit=5)``, and only lets endpoint and
delta results through claim selection and citation guarding.

The ``trigger`` labels ("provisional", "endpoint_commit", "late_detail_delta")
are Module 05 metadata.  They are recorded in telemetry and used to label the
code path; they are never passed to the retriever, whose signature is frozen.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import count
from uuid import uuid4

from .controller import RetrievalController
from .grounding import CitationCandidateGuard, ClaimEvidenceSelector, conflicting_evidence
from .models import (
    ControllerAction,
    Decision,
    EngineOutput,
    Evidence,
    Intent,
    TranscriptEvent,
)
from .session import EphemeralSession, SessionSynthesisError
from .telemetry import TelemetryRecorder

RETRIEVAL_LIMIT = 5
MAX_COMMIT_WORKERS = 4

TRIGGER_PROVISIONAL = "provisional"
TRIGGER_ENDPOINT_COMMIT = "endpoint_commit"
TRIGGER_LATE_DETAIL_DELTA = "late_detail_delta"


def _rendered(session: EphemeralSession) -> str | None:
    latest = session.latest
    return None if latest is None else latest.render()


class CiteFrontierEngine:
    """Runs a transcript event stream through controller, retriever, session."""

    def __init__(
        self,
        retriever,
        telemetry: TelemetryRecorder | None = None,
        use_dialogue_context: bool = False,
        claim_evidence_selector: ClaimEvidenceSelector | None = None,
    ):
        self._retriever = retriever
        self._telemetry = telemetry
        self._use_dialogue_context = use_dialogue_context
        self._claim_evidence_selector = (
            claim_evidence_selector
            if claim_evidence_selector is not None
            else ClaimEvidenceSelector()
        )
        self._controllers: dict[str, RetrievalController] = {}
        self._retrieval_ids = count(1)

    @property
    def claim_evidence_selector(self) -> ClaimEvidenceSelector:
        return self._claim_evidence_selector

    def new_session(self) -> EphemeralSession:
        return EphemeralSession(
            session_id=uuid4().hex, citation_guard=CitationCandidateGuard()
        )

    def begin_turn(self, session: EphemeralSession) -> None:
        """Reset controller state for a new turn.

        Session state (provisional, committed, versions) is deliberately
        untouched: only the controller's partial-transcript history restarts.
        """
        self._controllers[session.session_id] = RetrievalController()

    def _controller(self, session: EphemeralSession) -> RetrievalController:
        controller = self._controllers.get(session.session_id)
        if controller is None:
            controller = self._controllers.setdefault(session.session_id, RetrievalController())
        return controller

    def _next_retrieval_id(self, intent_key: str) -> str:
        return f"r-{next(self._retrieval_ids)}-{intent_key[:8]}"

    def _record(
        self, event_type: str, timestamp_s: float, session: EphemeralSession, **payload: object
    ) -> None:
        if self._telemetry is None:
            return
        self._telemetry.record(event_type, timestamp_s, session.session_id, **payload)

    def _search(self, query: str, retrieval_event_id: str) -> tuple[Evidence, ...]:
        return self._retriever.search(query, retrieval_event_id, limit=RETRIEVAL_LIMIT)

    def _query_for(self, session: EphemeralSession, intent: Intent) -> tuple[str, tuple[str, ...]]:
        if not self._use_dialogue_context:
            return intent.query, ()
        return session.contextualize(intent)

    def process_stream(
        self, session: EphemeralSession, event: TranscriptEvent
    ) -> EngineOutput:
        controller = self._controller(session)
        action = controller.decide(event, has_answer=session.has_answer)
        self._record(
            "controller_decision",
            event.timestamp_s,
            session,
            decision=action.decision.value,
            reason=action.reason,
            intents=[intent.key for intent in action.intents],
            invalidates_provisional=action.invalidates_provisional,
        )

        if action.invalidates_provisional:
            discarded_candidates = [
                item.chunk.chunk_id
                for evidence in session.provisional.values()
                for item in evidence
            ]
            invalidated = session.invalidate_provisionals()
            if invalidated:
                self._record(
                    "provisional_invalidated",
                    event.timestamp_s,
                    session,
                    retrieval_event_ids=list(invalidated),
                    discarded_candidates=discarded_candidates,
                )

        if action.decision is Decision.SUPPRESS:
            # Presentation-only: same answer object, re-rendered as bullets.
            return self._output(session, action, bullets=True)
        if action.decision is Decision.PROVISIONAL_RETRIEVE:
            return EngineOutput(
                action=action,
                answer=session.latest,
                rendered_answer=_rendered(session),
                retrieval_event_ids=self._stage_provisional(session, action, event.timestamp_s),
            )
        if action.decision is Decision.COMMIT_RETRIEVE:
            retrieval_event_ids = self._commit(
                session, action, event.timestamp_s, event.text.strip()
            )
            return EngineOutput(
                action=action,
                answer=session.latest,
                rendered_answer=_rendered(session),
                retrieval_event_ids=retrieval_event_ids,
            )
        return EngineOutput(
            action=action, answer=session.latest, rendered_answer=_rendered(session), retrieval_event_ids=()
        )

    def _output(
        self, session: EphemeralSession, action: ControllerAction, bullets: bool = False
    ) -> EngineOutput:
        latest = session.latest
        return EngineOutput(
            action=action,
            answer=latest,
            rendered_answer=None if latest is None else latest.render(bullets=bullets),
            retrieval_event_ids=(),
        )

    def _stage_provisional(
        self, session: EphemeralSession, action: ControllerAction, timestamp_s: float
    ) -> tuple[str, ...]:
        retrieval_event_ids: list[str] = []
        for intent in action.intents:
            query, context_intent_keys = self._query_for(session, intent)
            retrieval_id = self._next_retrieval_id(intent.key)
            self._record(
                "retrieval_started",
                timestamp_s,
                session,
                retrieval_event_id=retrieval_id,
                query=query,
                intent_id=intent.key,
                trigger=TRIGGER_PROVISIONAL,
                context_intent_keys=list(context_intent_keys),
            )
            evidence = self._search(query, retrieval_id)
            self._record(
                "retrieval_finished",
                timestamp_s,
                session,
                retrieval_event_id=retrieval_id,
                intent_id=intent.key,
                candidate_chunk_ids=[item.chunk.chunk_id for item in evidence],
                phase="provisional",
                trigger=TRIGGER_PROVISIONAL,
            )
            # Staged only.  Provisional evidence is never verified, selected,
            # or committed; a correction invalidates it below.
            session.stage_provisional(retrieval_id, intent, evidence)
            retrieval_event_ids.append(retrieval_id)
        return tuple(retrieval_event_ids)

    def _commit(
        self, session: EphemeralSession, action: ControllerAction, timestamp_s: float, root_query: str
    ) -> tuple[str, ...]:
        session.committed.clear()
        plan: list[tuple[Intent, str, str, tuple[str, ...]]] = []
        for intent in action.intents:
            query, context_intent_keys = self._query_for(session, intent)
            retrieval_id = self._next_retrieval_id(intent.key)
            self._record(
                "retrieval_started",
                timestamp_s,
                session,
                retrieval_event_id=retrieval_id,
                query=query,
                intent_id=intent.key,
                trigger=TRIGGER_ENDPOINT_COMMIT,
                context_intent_keys=list(context_intent_keys),
            )
            plan.append((intent, query, retrieval_id, context_intent_keys))

        with ThreadPoolExecutor(max_workers=MAX_COMMIT_WORKERS) as pool:
            results = list(pool.map(lambda item: self._search(item[1], item[2]), plan))

        retrieval_event_ids = tuple(item[2] for item in plan)
        for (intent, _query, retrieval_id, _context_keys), evidence in zip(plan, results, strict=True):
            self._record(
                "retrieval_finished",
                timestamp_s,
                session,
                retrieval_event_id=retrieval_id,
                intent_id=intent.key,
                candidate_chunk_ids=[item.chunk.chunk_id for item in evidence],
                phase="committed",
                trigger=TRIGGER_ENDPOINT_COMMIT,
            )
            accepted = self._select_claim_evidence(
                session,
                retrieval_id,
                evidence,
                timestamp_s,
                intent_key=intent.key,
                trigger=TRIGGER_ENDPOINT_COMMIT,
            )
            session.commit(intent, accepted)

        if plan:
            session.remember_committed_turn(action.intents)
            version = session.synthesize_initial(root_query)
            self._record(
                "answer_version",
                timestamp_s,
                session,
                version=version.version,
                parent_version=version.parent_version,
                changed=list(version.changed_claim_ids),
                claims=[claim.claim_id for claim in version.claims],
                retrieval_event_ids=list(retrieval_event_ids),
                trigger=TRIGGER_ENDPOINT_COMMIT,
            )
        return retrieval_event_ids

    def _select_claim_evidence(
        self,
        session: EphemeralSession,
        retrieval_id: str,
        evidence: tuple[Evidence, ...],
        timestamp_s: float,
        intent_key: str = "",
        trigger: str = TRIGGER_ENDPOINT_COMMIT,
    ) -> tuple[Evidence, ...]:
        """Select and citation-guard the evidence for one committed intent.

        INVARIANT (auditable): only endpoint-commit and late-detail-delta
        results may reach this method.  Provisional results are staged by
        ``EphemeralSession.stage_provisional`` and must never be verified,
        selected, or committed, so a speculative retrieval can never back a
        claim.  The two checks below are what enforce that.
        """
        if trigger == TRIGGER_PROVISIONAL:
            raise AssertionError(
                "provisional results must never pass through claim selection or commit"
            )
        if retrieval_id in session.invalidated_provisionals:
            raise AssertionError(f"invalidated retrieval {retrieval_id!r} cannot be committed")

        draft_claim = evidence[0].chunk.text if evidence else ""
        selection = self._claim_evidence_selector.select(draft_claim, evidence)
        accepted: tuple[str, ...] = ()
        rejected: tuple[str, ...] = ()
        withheld_reason: str | None = None
        if selection.evidence is not None:
            guard = session.citation_guard.enforce(
                (selection.evidence.chunk.chunk_id,), evidence
            )
            accepted, rejected = guard.accepted_citation_ids, guard.rejected_citation_ids
        if accepted and conflicting_evidence(selection.evidence, evidence):
            # Sources disagree on the requested quantity: abstain.
            withheld_reason = "conflicting_source_quantities"
            accepted, rejected = (), (selection.evidence.chunk.chunk_id,)

        verification = selection.verification
        self._record(
            "claim_evidence_selected",
            timestamp_s,
            session,
            retrieval_event_id=retrieval_id,
            intent_id=intent_key,
            trigger=trigger,
            evaluated_candidate_ids=list(selection.evaluated_candidate_ids),
            selected_chunk_id=None if selection.evidence is None else selection.evidence.chunk.chunk_id,
            accepted_citation_ids=list(accepted),
            rejected_citation_ids=list(rejected),
            verification=None if verification is None else verification.label,
            verification_score=None if verification is None else verification.score,
            uncertainty_required=not accepted,
            withheld_reason=withheld_reason,
        )
        if not accepted:
            return ()
        return tuple(item for item in evidence if item.chunk.chunk_id == accepted[0])

    def refine_with_late_detail(
        self,
        session: EphemeralSession,
        timestamp_s: float,
        target_intent_key: str,
        late_detail: str,
    ) -> EngineOutput:
        """Replace exactly one claim with a targeted delta. One retrieval."""
        latest = session.latest
        if latest is None:
            raise SessionSynthesisError(
                "late-detail refinement requires an existing answer"
            )
        target = session.intent_by_key.get(target_intent_key)
        if target is None:
            raise SessionSynthesisError(
                f"{target_intent_key!r} is not a current intent of session {session.session_id}"
            )

        delta_intent = Intent(
            key=f"{target.key}-delta-v{latest.version}",
            query=f"{target.query} {late_detail}",
        )
        retrieval_id = self._next_retrieval_id(delta_intent.key)
        self._record(
            "retrieval_started",
            timestamp_s,
            session,
            retrieval_event_id=retrieval_id,
            query=delta_intent.query,
            intent_id=target.key,
            trigger=TRIGGER_LATE_DETAIL_DELTA,
            delta_intent_key=delta_intent.key,
        )
        evidence = self._search(delta_intent.query, retrieval_id)
        self._record(
            "retrieval_finished",
            timestamp_s,
            session,
            retrieval_event_id=retrieval_id,
            intent_id=target.key,
            candidate_chunk_ids=[item.chunk.chunk_id for item in evidence],
            phase="committed",
            trigger=TRIGGER_LATE_DETAIL_DELTA,
        )
        accepted = self._select_claim_evidence(
            session,
            retrieval_id,
            evidence,
            timestamp_s,
            intent_key=target.key,
            trigger=TRIGGER_LATE_DETAIL_DELTA,
        )
        version = session.refine(target.key, delta_intent, accepted)
        self._record(
            "answer_version",
            timestamp_s,
            session,
            version=version.version,
            parent_version=version.parent_version,
            changed=list(version.changed_claim_ids),
            claims=[claim.claim_id for claim in version.claims],
            retrieval_event_ids=[retrieval_id],
            trigger=TRIGGER_LATE_DETAIL_DELTA,
        )
        action = ControllerAction(
            decision=Decision.COMMIT_RETRIEVE,
            intents=(delta_intent,),
            reason="targeted_delta",
        )
        return EngineOutput(
            action=action,
            answer=version,
            rendered_answer=version.render(),
            retrieval_event_ids=(retrieval_id,),
        )
