"""Unit tests for session state, claim synthesis, and dialogue context (Module 05)."""

from __future__ import annotations

import unittest

from citefrontier.models import CorpusChunk, Evidence, Intent
from citefrontier.session import (
    CitationValidationError,
    EphemeralSession,
    SessionSynthesisError,
    is_elliptical,
)

WARRANTY_TEXT = (
    "The LumaPad S1 has a 24-month limited warranty starting on the retail purchase date."
)
BATTERY_TEXT = "The LumaPad S1 battery capacity is covered for the first 12 months only."
CAM_TEXT = "The LumaCam C1 has a 24-month limited warranty beginning on its retail purchase date."


def make_chunk(chunk_id: str, text: str, doc_id: str = "LumaPad_S1") -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section=chunk_id.split("§")[-1],
        text=text,
        metadata={"entity": doc_id.replace("_", " ")},
    )


def make_evidence(
    text: str,
    chunk_id: str = "LumaPad_S1 §Warranty.1",
    retrieval_event_id: str = "r-endpoint-1",
    doc_id: str = "LumaPad_S1",
) -> Evidence:
    return Evidence(
        chunk=make_chunk(chunk_id, text, doc_id=doc_id),
        score=1.0,
        coverage=1.0,
        origin="test",
        retrieval_event_id=retrieval_event_id,
    )


class ProvisionalIsolationTests(unittest.TestCase):
    def test_invalidated_provisional_never_reaches_a_claim(self):
        # Acceptance: provisional never commits (testing.md §1 "05 Session", §3.2).
        session = EphemeralSession(session_id="s-provisional")
        intent = Intent(key="lumapad-s1-warranty", query="lumapad s1 warranty")
        staged = make_evidence(WARRANTY_TEXT, retrieval_event_id="r-provisional-1")

        session.stage_provisional("r-provisional-1", intent, (staged,))
        self.assertEqual(session.invalidate_provisionals(), ("r-provisional-1",))
        self.assertEqual(session.provisional, {})
        self.assertEqual(session.invalidated_provisionals, {"r-provisional-1"})

        # Even if invalidated evidence is handed to commit, it cannot be cited.
        session.commit(intent, (staged,))
        answer = session.synthesize_initial("LumaPad S1 warranty")
        (claim,) = answer.claims
        self.assertEqual(claim.citations, ())
        self.assertEqual(claim.evidence_event_ids, ())
        self.assertEqual(claim.text, "I could not verify: lumapad s1 warranty.")

        # A fresh retrieval for the same intent does produce a cited claim.
        fresh = make_evidence(WARRANTY_TEXT, retrieval_event_id="r-endpoint-1")
        session.commit(intent, (fresh,))
        refined = session.synthesize_initial("LumaPad S1 warranty")
        (claim,) = refined.claims
        self.assertEqual(claim.citations, ("LumaPad_S1 §Warranty.1",))
        self.assertEqual(claim.evidence_event_ids, ("r-endpoint-1",))
        self.assertNotIn("r-provisional-1", claim.evidence_event_ids)

    def test_staged_intent_is_not_a_current_intent(self):
        # Acceptance: provisional isolation (testing.md §3.2) — a staged intent is
        # never promoted to a current intent that a claim or delta could target.
        session = EphemeralSession(session_id="s-isolation")
        intent = Intent(key="lumapad-s1-warranty", query="lumapad s1 warranty")
        session.stage_provisional("r-1", intent, (make_evidence(WARRANTY_TEXT),))
        self.assertEqual(session.intent_by_key, {})
        session.invalidate_provisionals()
        self.assertEqual(session.intent_by_key, {})


class SynthesisTests(unittest.TestCase):
    def _two_intent_session(self) -> EphemeralSession:
        session = EphemeralSession(session_id="s-synthesis")
        pad = Intent(key="lumapad-s1-warranty", query="lumapad s1 warranty")
        cam = Intent(key="lumacam-c1-warranty", query="lumacam c1 warranty")
        session.commit(pad, (make_evidence(WARRANTY_TEXT),))
        session.commit(
            cam,
            (make_evidence(CAM_TEXT, "LumaCam_C1 §Warranty.1", "r-endpoint-2", "LumaCam_C1"),),
        )
        return session

    def test_commit_and_synthesize_builds_one_claim_per_intent(self):
        # Acceptance: one claim per committed intent, version 1, all claims changed.
        session = self._two_intent_session()
        answer = session.synthesize_initial("LumaPad S1 warranty and LumaCam C1 warranty")
        self.assertEqual(answer.version, 1)
        self.assertIsNone(answer.parent_version)
        self.assertEqual(len(answer.claims), 2)
        self.assertEqual(
            [claim.intent_key for claim in answer.claims],
            ["lumapad-s1-warranty", "lumacam-c1-warranty"],
        )
        self.assertEqual(
            answer.changed_claim_ids, tuple(claim.claim_id for claim in answer.claims)
        )
        self.assertEqual(answer.claims[0].citations, ("LumaPad_S1 §Warranty.1",))
        self.assertEqual(answer.claims[1].citations, ("LumaCam_C1 §Warranty.1",))
        # The claim text is the cited chunk text, verbatim.
        self.assertEqual(answer.claims[0].text, WARRANTY_TEXT)
        self.assertEqual(session.root_query, "LumaPad S1 warranty and LumaCam C1 warranty")
        self.assertTrue(session.has_answer)
        self.assertIs(session.latest, answer)
        self.assertIn("[LumaPad_S1 §Warranty.1]", answer.render())
        self.assertIn("- The LumaPad S1", answer.render(bullets=True))

    def test_version_increments_for_a_later_synthesis(self):
        session = self._two_intent_session()
        first = session.synthesize_initial("LumaPad S1 warranty")
        second = session.synthesize_initial("LumaPad S1 warranty")
        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)
        self.assertEqual(second.parent_version, 1)
        self.assertEqual(session.root_query, "LumaPad S1 warranty")

    def test_abstention_when_there_is_no_usable_evidence(self):
        # Acceptance: unknown/insufficient evidence abstains (testing.md §3.4).
        session = EphemeralSession(session_id="s-abstain")
        intent = Intent(key="unknown-product", query="lumapad x99 warranty")
        session.commit(intent, ())
        (claim,) = session.synthesize_initial("LumaPad X99 warranty").claims
        self.assertEqual(claim.text, "I could not verify: lumapad x99 warranty.")
        self.assertEqual(claim.citations, ())
        self.assertEqual(claim.evidence_event_ids, ())
        self.assertNotIn("[", session.latest.render())

    def test_citation_must_be_committed_for_that_intent(self):
        # Acceptance: citing something not committed is a hard error, not an
        # abstention (module-05 §3.3).
        session = EphemeralSession(session_id="s-citation")
        session.commit(
            Intent(key="intent-a", query="lumapad s1 warranty"),
            (make_evidence(WARRANTY_TEXT, retrieval_event_id="r-a"),),
        )
        with self.assertRaises(CitationValidationError) as raised:
            session._claim_for(
                "intent-b",
                (make_evidence(WARRANTY_TEXT, retrieval_event_id="r-b"),),
                "c-intent-b",
            )
        self.assertIn("LumaPad_S1 §Warranty.1", str(raised.exception))
        self.assertIn("intent-b", str(raised.exception))

    def test_refine_requires_an_answer_and_a_current_claim(self):
        empty = EphemeralSession(session_id="s-empty")
        with self.assertRaises(SessionSynthesisError):
            empty.refine("lumapad-s1-warranty", Intent(key="d", query="delta"), ())
        session = self._two_intent_session()
        session.synthesize_initial("LumaPad S1 warranty")
        with self.assertRaises(SessionSynthesisError):
            session.refine("not-an-intent", Intent(key="d", query="delta"), ())


class RefinementTests(unittest.TestCase):
    def _session(self) -> tuple[EphemeralSession, object]:
        session = EphemeralSession(session_id="s-refine")
        pad = Intent(key="lumapad-s1-warranty", query="lumapad s1 warranty")
        cam = Intent(key="lumacam-c1-warranty", query="lumacam c1 warranty")
        session.commit(pad, (make_evidence(WARRANTY_TEXT, retrieval_event_id="r-endpoint-1"),))
        session.commit(
            cam,
            (make_evidence(CAM_TEXT, "LumaCam_C1 §Warranty.1", "r-endpoint-2", "LumaCam_C1"),),
        )
        first = session.synthesize_initial("LumaPad S1 warranty and LumaCam C1 warranty")
        return session, first

    def test_refine_replaces_only_the_target_claim(self):
        # Acceptance: targeted delta uses exactly one retrieval and preserves the
        # unaffected claim with the same claim_id (testing.md §1 "05 Session", §2
        # "refinement" family, §3.5).
        session, first = self._session()
        delta = Intent(key="lumapad-s1-warranty-delta-v1", query="lumapad s1 warranty battery")
        version = session.refine(
            "lumapad-s1-warranty",
            delta,
            (make_evidence(BATTERY_TEXT, "LumaPad_S1 §Battery.1", "r-delta-1"),),
        )

        self.assertEqual(version.version, 2)
        self.assertEqual(version.parent_version, 1)
        self.assertEqual(len(version.claims), 2)
        refined, preserved = version.claims
        self.assertEqual(refined.intent_key, "lumapad-s1-warranty")
        self.assertEqual(refined.text, BATTERY_TEXT)
        self.assertEqual(refined.citations, ("LumaPad_S1 §Battery.1",))
        # Literal id format from module-05 §3: "{claim_id}-v{version+1}".
        self.assertEqual(refined.claim_id, f"{first.claims[0].claim_id}-v3")
        self.assertEqual(version.changed_claim_ids, (refined.claim_id,))
        self.assertEqual(preserved.claim_id, first.claims[1].claim_id)
        self.assertEqual(preserved.text, first.claims[1].text)
        self.assertEqual(preserved.citations, first.claims[1].citations)

    def test_refine_preserves_the_unaffected_claim_object(self):
        # Acceptance: unaffected claims are byte-identical across versions
        # (testing.md §2 "unchanged claims byte-identical to the previous step").
        session, first = self._session()
        version = session.refine(
            "lumapad-s1-warranty",
            Intent(key="lumapad-s1-warranty-delta-v1", query="lumapad s1 warranty battery"),
            (make_evidence(BATTERY_TEXT, "LumaPad_S1 §Battery.1", "r-delta-1"),),
        )
        self.assertIs(version.claims[1], first.claims[1])
        self.assertEqual(version.claims[1], first.claims[1])

    def test_refine_abstains_when_the_delta_finds_nothing(self):
        session, first = self._session()
        version = session.refine(
            "lumapad-s1-warranty", Intent(key="delta", query="delta"), ()
        )
        self.assertEqual(version.claims[0].citations, ())
        self.assertEqual(version.claims[0].text, "I could not verify: lumapad s1 warranty.")
        self.assertEqual(version.claims[1], first.claims[1])


class DialogueContextTests(unittest.TestCase):
    def _remembered(self) -> EphemeralSession:
        session = EphemeralSession(session_id="s-context")
        session.remember_committed_turn(
            (
                Intent(key="lumapad-s1-warranty", query="lumapad s1 warranty"),
                Intent(key="lumacam-c1-warranty", query="lumacam c1 warranty"),
            )
        )
        return session

    def test_pure_deictic_follow_up_reuses_the_last_user_intent(self):
        # Acceptance: elliptical follow-up uses user-intent context only
        # (testing.md §1 "05 Session", §3; module-05 §3).
        session = self._remembered()
        for query in ("what about that", "yes", "and what", "what does it", "how about"):
            with self.subTest(query=query):
                expanded, keys = session.contextualize(Intent(key="follow-up", query=query))
                self.assertEqual(expanded, "lumapad s1 warranty lumacam c1 warranty")
                self.assertEqual(keys, ("lumapad-s1-warranty", "lumacam-c1-warranty"))

    def test_specific_short_question_keeps_its_own_wording(self):
        session = self._remembered()
        expanded, keys = session.contextualize(
            Intent(key="follow-up", query="how about the LumaPad S1 filter")
        )
        self.assertEqual(
            expanded, "how about the LumaPad S1 filter lumapad s1 warranty lumacam c1 warranty"
        )
        self.assertEqual(keys, ("lumapad-s1-warranty", "lumacam-c1-warranty"))

    def test_non_elliptical_query_is_returned_unchanged(self):
        session = self._remembered()
        query = "what is the battery capacity warranty period for the LumaPad S1 tablet"
        expanded, keys = session.contextualize(Intent(key="fresh", query=query))
        self.assertEqual(expanded, query)
        self.assertEqual(keys, ())
        self.assertFalse(is_elliptical(query))

    def test_contextualize_without_prior_turns_is_a_no_op(self):
        session = EphemeralSession(session_id="s-no-context")
        expanded, keys = session.contextualize(Intent(key="k", query="what about that"))
        self.assertEqual(expanded, "what about that")
        self.assertEqual(keys, ())
        expanded, keys = self._remembered().contextualize(
            Intent(key="k", query="what about that"), max_prior_turns=0
        )
        self.assertEqual(expanded, "what about that")
        self.assertEqual(keys, ())

    def test_context_never_comes_from_answer_text(self):
        # Acceptance: elliptical follow-up uses user-intent context only — the
        # committed claim text must never be injected into the next query.
        session = self._remembered()
        intent = Intent(key="lumapad-s1-warranty", query="lumapad s1 warranty")
        session.commit(intent, (make_evidence(WARRANTY_TEXT, retrieval_event_id="r-1"),))
        session.synthesize_initial("LumaPad S1 warranty")
        expanded, _keys = session.contextualize(Intent(key="follow-up", query="what about that"))
        self.assertEqual(expanded, "lumapad s1 warranty lumacam c1 warranty")
        self.assertNotIn(WARRANTY_TEXT, expanded)
        self.assertNotIn("warranty starting on the retail purchase date", expanded)

    def test_remember_committed_turn_records_intents_only(self):
        # module-05 §3: the remembered turn is user intent text; no assistant or
        # answer text can enter the dialogue context.
        session = self._remembered()
        self.assertEqual(len(session._remembered_turns), 1)
        self.assertEqual(
            session._remembered_turns[0],
            (
                Intent(key="lumapad-s1-warranty", query="lumapad s1 warranty"),
                Intent(key="lumacam-c1-warranty", query="lumacam c1 warranty"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
