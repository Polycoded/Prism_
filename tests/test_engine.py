"""Unit tests for the core-library engine: controller + retriever + session (Module 05)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from citefrontier.engine import (
    TRIGGER_ENDPOINT_COMMIT,
    TRIGGER_LATE_DETAIL_DELTA,
    TRIGGER_PROVISIONAL,
    CiteFrontierEngine,
)
from citefrontier.grounding import ClaimEvidenceSelection, ClaimEvidenceSelector
from citefrontier.models import CorpusChunk, Decision, TranscriptEvent
from citefrontier.retrieval import HybridRetriever
from citefrontier.session import SessionSynthesisError
from citefrontier.telemetry import TelemetryRecorder

CHUNKS: tuple[CorpusChunk, ...] = (
    CorpusChunk(
        chunk_id="LumaPad_S1 §Warranty.1",
        doc_id="LumaPad_S1",
        section="Warranty.1",
        text="The LumaPad S1 has a 24-month limited warranty starting on the retail purchase date.",
        metadata={"entity": "LumaPad S1"},
    ),
    CorpusChunk(
        chunk_id="LumaPad_S1 §Battery.1",
        doc_id="LumaPad_S1",
        section="Battery.1",
        text="The LumaPad S1 battery capacity is covered for the first 12 months only.",
        metadata={"entity": "LumaPad S1"},
    ),
    CorpusChunk(
        chunk_id="LumaPad_S2 §Warranty.1",
        doc_id="LumaPad_S2",
        section="Warranty.1",
        text="The LumaPad S2 has a 36-month limited warranty starting on the retail purchase date.",
        metadata={"entity": "LumaPad S2"},
    ),
    CorpusChunk(
        chunk_id="LumaCam_C1 §Warranty.1",
        doc_id="LumaCam_C1",
        section="Warranty.1",
        text="The LumaCam C1 has a 24-month limited warranty beginning on its retail purchase date.",
        metadata={"entity": "LumaCam C1"},
    ),
)

CONFLICTING_CHUNKS: tuple[CorpusChunk, ...] = (
    CorpusChunk(
        chunk_id="Widget §Warranty.1",
        doc_id="Widget",
        section="Warranty.1",
        text="The Widget device has a 24-month limited warranty on the retail purchase date.",
        metadata={"entity": "Widget device"},
    ),
    CorpusChunk(
        chunk_id="Widget §Warranty.2",
        doc_id="Widget",
        section="Warranty.2",
        text="The Widget device has a 36-month limited warranty on the retail purchase date.",
        metadata={"entity": "Widget device"},
    ),
)


class RecordingRetriever:
    """Delegates to the frozen Module 02 retriever and counts its calls.

    The signature is deliberately exactly ``search(query, retrieval_event_id,
    limit=5)``: passing a ``trigger`` keyword would raise ``TypeError``.
    """

    def __init__(self, chunks: tuple[CorpusChunk, ...] = CHUNKS):
        self._inner = HybridRetriever(chunks)
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, retrieval_event_id: str, limit: int = 5):
        self.calls.append((query, retrieval_event_id, limit))
        return self._inner.search(query, retrieval_event_id, limit=limit)


class CountingSelector(ClaimEvidenceSelector):
    def __init__(self):
        super().__init__()
        self.select_calls: list[tuple[str, tuple[str, ...]]] = []

    def select(self, claim: str, candidates) -> ClaimEvidenceSelection:
        self.select_calls.append((claim, tuple(item.chunk.chunk_id for item in candidates)))
        return super().select(claim, candidates)


def build_engine(
    chunks: tuple[CorpusChunk, ...] = CHUNKS,
    use_dialogue_context: bool = False,
) -> tuple[CiteFrontierEngine, RecordingRetriever, TelemetryRecorder, CountingSelector]:
    retriever = RecordingRetriever(chunks)
    telemetry = TelemetryRecorder()
    selector = CountingSelector()
    engine = CiteFrontierEngine(
        retriever,
        telemetry=telemetry,
        use_dialogue_context=use_dialogue_context,
        claim_evidence_selector=selector,
    )
    return engine, retriever, telemetry, selector


def send(engine: CiteFrontierEngine, session, text: str, timestamp_s: float, is_final: bool = True):
    return engine.process_stream(session, TranscriptEvent(timestamp_s, text, is_final=is_final))


def chunk_text(chunk_id: str, chunks: tuple[CorpusChunk, ...] = CHUNKS) -> str:
    return next(chunk.text for chunk in chunks if chunk.chunk_id == chunk_id)


class ProvisionalAndCommitTests(unittest.TestCase):
    def test_provisional_turn_stages_evidence_without_building_claims(self):
        # Acceptance: provisional never commits (testing.md §1 "05 Session", §3.2).
        engine, retriever, _telemetry, selector = build_engine()
        session = engine.new_session()

        first = send(engine, session, "Find warranty details for", 0.0, is_final=False)
        self.assertEqual(first.action.decision, Decision.WAIT)
        self.assertEqual(len(retriever.calls), 0)
        second = send(
            engine,
            session,
            "Find warranty details for the LumaPad S1 warranty duration",
            0.8,
            is_final=False,
        )
        self.assertEqual(second.action.decision, Decision.PROVISIONAL_RETRIEVE)

        self.assertIsNone(second.answer)
        self.assertIsNone(second.rendered_answer)
        self.assertEqual(session.versions, [])
        self.assertEqual(session.committed, {})
        self.assertEqual(session.has_answer, False)
        self.assertEqual(len(session.provisional), len(second.retrieval_event_ids))
        self.assertEqual(len(retriever.calls), 1)
        # A provisional retrieval is never verified, selected, or committed.
        self.assertEqual(selector.select_calls, [])

    def test_correction_invalidates_provisionals_before_the_final_answer(self):
        # Acceptance: correction invalidates provisionals before the final answer
        # (testing.md §1 "05 Session", §2 "correction" family).
        engine, retriever, telemetry, _selector = build_engine()
        session = engine.new_session()

        send(engine, session, "Find warranty details for", 0.0, is_final=False)
        provisional = send(
            engine,
            session,
            "Find warranty details for the LumaPad S1 warranty duration",
            0.8,
            is_final=False,
        )
        provisional_ids = provisional.retrieval_event_ids
        self.assertEqual(len(provisional_ids), 1)

        final = send(engine, session, "LumaPad S2 36-month warranty instead", 1.4)
        self.assertEqual(final.action.decision, Decision.COMMIT_RETRIEVE)
        self.assertTrue(final.action.invalidates_provisional)
        self.assertEqual(session.provisional, {})
        self.assertEqual(session.invalidated_provisionals, set(provisional_ids))

        invalidated = telemetry.by_type("provisional_invalidated")
        self.assertEqual(len(invalidated), 1)
        self.assertEqual(invalidated[0].payload["retrieval_event_ids"], list(provisional_ids))

        (claim,) = final.answer.claims
        self.assertEqual(claim.citations, ("LumaPad_S2 §Warranty.1",))
        self.assertNotIn(provisional_ids[0], claim.evidence_event_ids)
        self.assertEqual(claim.evidence_event_ids, final.retrieval_event_ids)
        # The retracted product is not cited anywhere in the answer.
        self.assertNotIn("LumaPad_S1", final.rendered_answer)

    def test_citations_resolve_to_evaluated_candidates(self):
        # Acceptance: citations always resolve to committed candidates
        # (testing.md §1 "05 Session", §2 replay checks, §3.2).
        engine, _retriever, telemetry, _selector = build_engine()
        session = engine.new_session()
        output = send(engine, session, "LumaPad S1 warranty and LumaCam C1 warranty", 0.0)

        selections = {
            event.payload["intent_id"]: event.payload
            for event in telemetry.by_type("claim_evidence_selected")
        }
        self.assertEqual(len(output.answer.claims), 2)
        for claim in output.answer.claims:
            selection = selections[claim.intent_key]
            self.assertEqual(len(claim.citations), 1)
            self.assertIn(claim.citations[0], selection["evaluated_candidate_ids"])
            self.assertEqual(selection["accepted_citation_ids"], list(claim.citations))
            self.assertEqual(selection["rejected_citation_ids"], [])
            # Exact-extractive: the claim text is the cited chunk's text verbatim.
            self.assertEqual(claim.text, chunk_text(claim.citations[0]))

    def test_unknown_topic_abstains_without_citations(self):
        # Acceptance: unknown topic abstains (testing.md §1 "05 Session", §3.4).
        # Nothing in the corpus covers this request, so the intent abstains with
        # no citation instead of borrowing a neighbouring chunk.
        engine, _retriever, _telemetry, _selector = build_engine()
        session = engine.new_session()
        output = send(engine, session, "banana smoothie recipe", 0.0)
        (claim,) = output.answer.claims
        self.assertEqual(claim.citations, ())
        self.assertEqual(claim.evidence_event_ids, ())
        self.assertEqual(claim.text, "I could not verify: banana smoothie recipe.")
        self.assertEqual(output.rendered_answer, claim.text)

    def test_conflicting_sources_abstain_instead_of_picking_one(self):
        # Acceptance: conflicting quantities abstain (testing.md §3.4; module-05 §9).
        engine, _retriever, telemetry, _selector = build_engine(CONFLICTING_CHUNKS)
        session = engine.new_session()
        output = send(engine, session, "Widget warranty", 0.0)

        (claim,) = output.answer.claims
        self.assertEqual(claim.citations, ())
        self.assertEqual(claim.text, "I could not verify: widget warranty.")
        selection = telemetry.by_type("claim_evidence_selected")[0]
        self.assertEqual(len(selection.payload["evaluated_candidate_ids"]), 2)
        self.assertEqual(selection.payload["withheld_reason"], "conflicting_source_quantities")
        self.assertTrue(selection.payload["uncertainty_required"])


class SuppressionTests(unittest.TestCase):
    def test_suppress_rerenders_bullets_without_retrieval_or_state_change(self):
        # Acceptance: presentation suppression leaves search count, claims,
        # citations, and version unchanged (testing.md §3.6, §2 "presentation").
        engine, retriever, _telemetry, selector = build_engine()
        session = engine.new_session()
        answer = send(engine, session, "LumaPad S1 warranty", 0.0).answer
        calls_before = len(retriever.calls)
        select_calls_before = len(selector.select_calls)

        output = send(engine, session, "Please repeat my last answer in two bullets", 2.0, is_final=False)
        self.assertEqual(output.action.decision, Decision.SUPPRESS)
        self.assertIs(output.answer, answer)
        self.assertEqual(output.answer.version, 1)
        self.assertEqual(output.answer.claims, answer.claims)
        self.assertEqual(output.retrieval_event_ids, ())
        self.assertEqual(len(retriever.calls), calls_before)
        self.assertEqual(len(selector.select_calls), select_calls_before)
        self.assertEqual(len(session.versions), 1)
        self.assertEqual(
            output.rendered_answer,
            "\n".join(f"- {claim.text} [{', '.join(claim.citations)}]" for claim in answer.claims),
        )

    def test_mixed_factual_turn_is_not_suppressed(self):
        # testing.md §2 "presentation": mixed factual turns are NOT suppressed.
        engine, _retriever, _telemetry, _selector = build_engine()
        session = engine.new_session()
        send(engine, session, "LumaPad S1 warranty", 0.0)
        output = send(
            engine,
            session,
            "Repeat my last answer in two bullets and add the LumaCam C1 warranty",
            2.0,
        )
        self.assertNotEqual(output.action.decision, Decision.SUPPRESS)


class RefinementTests(unittest.TestCase):
    def _answered(self):
        engine, retriever, telemetry, selector = build_engine()
        session = engine.new_session()
        answer = send(engine, session, "LumaPad S1 warranty and LumaCam C1 warranty", 0.0).answer
        return engine, retriever, telemetry, selector, session, answer

    def test_late_detail_delta_uses_exactly_one_retrieval(self):
        # Acceptance: targeted delta uses exactly one retrieval and leaves the
        # unaffected claim preserved with the same claim_id (testing.md §1, §3.5).
        engine, retriever, _telemetry, _selector, session, answer = self._answered()
        calls_before = len(retriever.calls)

        output = engine.refine_with_late_detail(
            session, 2.0, "lumapad-s1-warranty", "battery capacity"
        )

        self.assertEqual(len(retriever.calls) - calls_before, 1)
        self.assertEqual(len(output.retrieval_event_ids), 1)
        self.assertEqual(output.answer.version, 2)
        self.assertEqual(output.answer.parent_version, 1)
        refined, preserved = output.answer.claims
        self.assertEqual(refined.citations, ("LumaPad_S1 §Battery.1",))
        self.assertEqual(refined.claim_id, f"{answer.claims[0].claim_id}-v3")
        self.assertEqual(output.answer.changed_claim_ids, (refined.claim_id,))
        self.assertIs(preserved, answer.claims[1])
        self.assertEqual(preserved.claim_id, answer.claims[1].claim_id)
        self.assertEqual(preserved.text, answer.claims[1].text)
        self.assertEqual(preserved.citations, answer.claims[1].citations)

    def test_delta_query_is_the_target_query_plus_the_late_detail(self):
        engine, retriever, telemetry, _selector, session, _answer = self._answered()
        engine.refine_with_late_detail(session, 2.0, "lumapad-s1-warranty", "battery capacity")
        started = telemetry.by_type("retrieval_started")[-1]
        self.assertEqual(started.payload["query"], "lumapad s1 warranty battery capacity")
        self.assertEqual(started.payload["delta_intent_key"], "lumapad-s1-warranty-delta-v1")
        self.assertEqual(started.payload["trigger"], TRIGGER_LATE_DETAIL_DELTA)
        self.assertEqual(retriever.calls[-1][0], "lumapad s1 warranty battery capacity")

    def test_delta_for_an_unknown_or_stale_target_raises(self):
        engine, _retriever, _telemetry, _selector, session, _answer = self._answered()
        with self.assertRaises(SessionSynthesisError):
            engine.refine_with_late_detail(session, 2.0, "lumacam-c9-warranty", "battery capacity")
        self.assertEqual(len(session.versions), 1)


class TurnLifecycleTests(unittest.TestCase):
    def test_begin_turn_resets_controller_state_only(self):
        engine, retriever, _telemetry, _selector = build_engine()
        session = engine.new_session()
        answer = send(engine, session, "LumaPad S1 warranty", 0.0).answer
        committed_before = dict(session.committed)
        engine.begin_turn(session)

        self.assertEqual(session.versions, [answer])
        self.assertEqual(session.committed, committed_before)
        self.assertTrue(session.has_answer)
        # The controller starts over: a bare partial waits again instead of
        # treating the previous turn's text as a stable prefix.
        output = send(engine, session, "LumaCam", 2.0, is_final=False)
        self.assertEqual(output.action.decision, Decision.WAIT)
        self.assertEqual(len(retriever.calls), 1)

    def test_new_session_starts_empty_with_its_own_id(self):
        engine, _retriever, _telemetry, _selector = build_engine()
        first, second = engine.new_session(), engine.new_session()
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(first.versions, [])
        self.assertEqual(first.provisional, {})
        self.assertEqual(first.committed, {})
        self.assertEqual(first.invalidated_provisionals, set())
        self.assertIsNot(first.citation_guard, second.citation_guard)


class DialogueContextEngineTests(unittest.TestCase):
    def test_elliptical_follow_up_searches_with_remembered_user_intent(self):
        # Acceptance: elliptical follow-up uses user-intent context only
        # (testing.md §1 "05 Session").
        engine, retriever, telemetry, _selector = build_engine(use_dialogue_context=True)
        session = engine.new_session()
        send(engine, session, "LumaPad S1 warranty", 0.0)
        send(engine, session, "what about the battery", 2.0)

        started = telemetry.by_type("retrieval_started")[-1]
        self.assertEqual(started.payload["query"], "about battery lumapad s1 warranty")
        self.assertEqual(started.payload["context_intent_keys"], ["lumapad-s1-warranty"])
        self.assertEqual(retriever.calls[-1][0], "about battery lumapad s1 warranty")
        # No answer text is injected into the follow-up query.
        self.assertNotIn("retail purchase date", retriever.calls[-1][0])

    def test_dialogue_context_is_off_by_default(self):
        engine, retriever, _telemetry, _selector = build_engine()
        session = engine.new_session()
        send(engine, session, "LumaPad S1 warranty", 0.0)
        send(engine, session, "what about the battery", 2.0)
        self.assertEqual(retriever.calls[-1][0], "about battery")


class TelemetryTests(unittest.TestCase):
    def test_expected_events_are_recorded_with_the_session_id(self):
        # Acceptance: every emitted event carries its session id and type
        # (docs/interfaces.md §5.4; module-05 §6).
        engine, _retriever, telemetry, _selector = build_engine()
        session = engine.new_session()
        send(engine, session, "LumaPad S1 warranty", 1.25)

        for event_type in ("controller_decision", "retrieval_started", "claim_evidence_selected", "answer_version"):
            with self.subTest(event_type=event_type):
                events = telemetry.by_type(event_type)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0].event_type, event_type)
                self.assertEqual(events[0].session_id, session.session_id)
                self.assertEqual(events[0].timestamp_s, 1.25)

        decision = telemetry.by_type("controller_decision")[0].payload
        self.assertEqual(decision["decision"], "commit_retrieve")
        self.assertEqual(decision["reason"], "final_transcript_endpoint")

        selection = telemetry.by_type("claim_evidence_selected")[0].payload
        self.assertEqual(selection["trigger"], TRIGGER_ENDPOINT_COMMIT)
        self.assertEqual(selection["verification"], "exact_extractive")
        self.assertEqual(selection["verification_score"], 1.0)
        self.assertEqual(selection["accepted_citation_ids"], ["LumaPad_S1 §Warranty.1"])
        self.assertIsNone(selection["withheld_reason"])

        version = telemetry.by_type("answer_version")[0].payload
        self.assertEqual(version["version"], 1)
        self.assertIsNone(version["parent_version"])
        self.assertEqual(version["changed"], ["c-lumapad-s1-warranty"])
        self.assertEqual(version["trigger"], TRIGGER_ENDPOINT_COMMIT)

        provisional_started = [
            event
            for event in telemetry.by_type("retrieval_started")
            if event.payload["trigger"] == TRIGGER_PROVISIONAL
        ]
        self.assertEqual(provisional_started, [])

    def test_write_jsonl_round_trips_every_event_in_order(self):
        # Acceptance: the session log is serializable via dataclasses.asdict
        # (module-05 §6; docs/module-10 §5).
        engine, _retriever, telemetry, _selector = build_engine()
        session = engine.new_session()
        send(engine, session, "LumaPad S1 warranty", 0.0)
        engine.refine_with_late_detail(session, 2.0, "lumapad-s1-warranty", "battery capacity")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "telemetry.jsonl"
            telemetry.write_jsonl(path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), len(telemetry.events))
        self.assertEqual(
            [json.loads(line) for line in lines], [asdict(event) for event in telemetry.events]
        )
        self.assertEqual(
            [json.loads(line)["event_type"] for line in lines],
            [event.event_type for event in telemetry.events],
        )

    def test_engine_works_without_a_telemetry_recorder(self):
        retriever = RecordingRetriever()
        engine = CiteFrontierEngine(retriever)
        session = engine.new_session()
        output = send(engine, session, "LumaPad S1 warranty", 0.0)
        self.assertEqual(output.answer.version, 1)


if __name__ == "__main__":
    unittest.main()
