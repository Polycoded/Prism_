"""Unit tests for grounding: verifiers, claim selector, citation guard (Module 05)."""

from __future__ import annotations

import importlib.util
import unittest

from citefrontier.grounding import (
    ClaimEvidenceSelector,
    CitationCandidateGuard,
    EntailmentResult,
    ExtractiveEntailmentVerifier,
    NliEntailmentVerifier,
    conflicting_evidence,
)
from citefrontier.models import CorpusChunk, Evidence

WARRANTY_TEXT = (
    "The LumaPad S1 has a 24-month limited warranty starting on the retail purchase date."
)


def make_chunk(
    chunk_id: str,
    text: str,
    section: str = "Warranty.1",
    doc_id: str = "LumaPad_S1",
    entity: str = "LumaPad S1",
) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id, doc_id=doc_id, section=section, text=text, metadata={"entity": entity}
    )


def make_evidence(chunk: CorpusChunk, retrieval_event_id: str = "r-1", score: float = 1.0) -> Evidence:
    return Evidence(
        chunk=chunk,
        score=score,
        coverage=1.0,
        origin="test",
        retrieval_event_id=retrieval_event_id,
    )


class ScriptedVerifier:
    """Verifier stub that returns a fixed result per candidate index."""

    def __init__(self, results):
        self._results = dict(results)
        self.calls: list[tuple[str, str]] = []

    def verify(self, claim: str, evidence_text: str) -> EntailmentResult:
        self.calls.append((claim, evidence_text))
        return self._results.get(evidence_text, EntailmentResult(False, 0.0, "not_entailed"))


class ExtractiveVerifierTests(unittest.TestCase):
    """interfaces.md §5.2: a cited claim text is a normalized-substring copy of its chunk."""

    def test_normalized_substring_entails(self):
        # Acceptance: exact-extractive verification (testing.md §3.3, §1 "05 Session").
        result = ExtractiveEntailmentVerifier().verify("24 month limited warranty", WARRANTY_TEXT)
        self.assertTrue(result.entailed)
        self.assertEqual(result.label, "exact_extractive")
        self.assertEqual(result.score, 1.0)

    def test_non_substring_does_not_entail(self):
        result = ExtractiveEntailmentVerifier().verify("36-month limited warranty", WARRANTY_TEXT)
        self.assertFalse(result.entailed)
        self.assertEqual(result.label, "not_entailed")
        self.assertEqual(result.score, 0.0)

    def test_empty_claim_does_not_entail(self):
        self.assertFalse(ExtractiveEntailmentVerifier().verify("", WARRANTY_TEXT).entailed)


class SelectorTests(unittest.TestCase):
    def test_tie_on_score_keeps_the_earlier_candidate(self):
        # Acceptance: selector picks max by (score, -index) for a stable tie-break
        # (module-05 §2). Both candidates entail the draft claim.
        top = make_chunk("C1", "The LumaPad S1 has a 24-month limited warranty.")
        later = make_chunk(
            "C2", "The LumaPad S1 has a 24-month limited warranty. Exclusions apply."
        )
        selection = ClaimEvidenceSelector().select(
            top.text, (make_evidence(top), make_evidence(later))
        )
        self.assertIsNotNone(selection.evidence)
        self.assertEqual(selection.evidence.chunk.chunk_id, "C1")
        self.assertEqual(selection.evaluated_candidate_ids, ("C1", "C2"))

    def test_highest_score_wins_over_position(self):
        first = make_chunk("C1", "first candidate")
        second = make_chunk("C2", "second candidate")
        verifier = ScriptedVerifier(
            {
                "first candidate": EntailmentResult(True, 0.8, "entailment"),
                "second candidate": EntailmentResult(True, 0.9, "entailment"),
            }
        )
        selection = ClaimEvidenceSelector(verifier=verifier).select(
            "draft", (make_evidence(first), make_evidence(second))
        )
        self.assertEqual(selection.evidence.chunk.chunk_id, "C2")
        self.assertEqual(selection.verification.label, "entailment")
        self.assertEqual(verifier.calls, [("draft", "first candidate"), ("draft", "second candidate")])

    def test_no_entailing_candidate_returns_all_evaluated_ids(self):
        candidates = (make_evidence(make_chunk("C1", "a")), make_evidence(make_chunk("C2", "b")))
        selection = ClaimEvidenceSelector().select("unrelated draft", candidates)
        self.assertIsNone(selection.evidence)
        self.assertIsNone(selection.verification)
        self.assertEqual(selection.evaluated_candidate_ids, ("C1", "C2"))

    def test_default_verifier_is_extractive(self):
        self.assertIsInstance(ClaimEvidenceSelector().verifier, ExtractiveEntailmentVerifier)


class CitationGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = CitationCandidateGuard()
        self.candidates = (
            make_evidence(make_chunk("C1", "one")),
            make_evidence(make_chunk("C2", "two")),
        )

    def test_accepts_only_ids_in_the_candidate_set(self):
        # Acceptance: citations always resolve to committed candidates
        # (testing.md §1 "05 Session"; module-05 §9).
        decision = self.guard.enforce(("C2", "C1"), self.candidates)
        self.assertEqual(decision.accepted_citation_ids, ("C2", "C1"))
        self.assertEqual(decision.rejected_citation_ids, ())

    def test_rejects_fabricated_and_stale_ids(self):
        stale = make_chunk("C9", "an earlier turn's chunk")
        decision = self.guard.enforce(
            ("C1", "LumaPad_S1 §Warranty.9", stale.chunk_id, "C7"), self.candidates
        )
        self.assertEqual(decision.accepted_citation_ids, ("C1",))
        self.assertEqual(
            decision.rejected_citation_ids,
            ("LumaPad_S1 §Warranty.9", stale.chunk_id, "C7"),
        )

    def test_fabricated_citation_is_structurally_impossible(self):
        # Acceptance: fabricated IDs = 0 (testing.md §2). Nothing outside the
        # candidate set can be accepted, no matter what is proposed.
        proposed = ["C1", "C2", "C3", "", "LumaPad_S1 §Warranty.1", "c-anything"]
        decision = self.guard.enforce(proposed, self.candidates)
        allowed = {candidate.chunk.chunk_id for candidate in self.candidates}
        self.assertTrue(set(decision.accepted_citation_ids) <= allowed)
        self.assertEqual(
            set(decision.accepted_citation_ids) | set(decision.rejected_citation_ids),
            set(proposed),
        )
        self.assertEqual(self.guard.enforce(proposed, ()).accepted_citation_ids, ())


class ConflictDetectionTests(unittest.TestCase):
    def test_competing_quantities_for_the_same_entity_conflict(self):
        # Acceptance: conflicting quantities abstain (testing.md §3.4; module-05 §9).
        top = make_evidence(
            make_chunk(
                "Widget §Warranty.1",
                "The Widget device has a 24-month limited warranty.",
                doc_id="Widget",
                entity="Widget device",
            )
        )
        rival = make_evidence(
            make_chunk(
                "Widget §Warranty.2",
                "The Widget device has a 36-month limited warranty.",
                doc_id="Widget",
                entity="Widget device",
            ),
            retrieval_event_id="r-2",
        )
        self.assertEqual(conflicting_evidence(top, (top, rival)), (rival,))

    def test_same_quantity_or_other_entity_does_not_conflict(self):
        top = make_evidence(
            make_chunk(
                "Widget §Warranty.1",
                "The Widget device has a 24-month limited warranty.",
                doc_id="Widget",
                entity="Widget device",
            )
        )
        same = make_evidence(
            make_chunk(
                "Widget §Warranty.2",
                "The Widget device has a 24-month limited warranty.",
                doc_id="Widget",
                entity="Widget device",
            )
        )
        other_entity = make_evidence(
            make_chunk(
                "Gadget §Warranty.1",
                "The Gadget device has a 36-month limited warranty.",
                doc_id="Gadget",
                entity="Gadget device",
            ),
            retrieval_event_id="r-3",
        )
        no_quantity = make_evidence(
            make_chunk(
                "Widget §Exclusions.1",
                "Water damage is excluded from the Widget device warranty.",
                doc_id="Widget",
                entity="Widget device",
            ),
            retrieval_event_id="r-4",
        )
        others = (same, other_entity, no_quantity)
        self.assertEqual(conflicting_evidence(top, (top, *others)), ())

    def test_top_without_quantities_never_conflicts(self):
        top = make_evidence(make_chunk("Widget §Setup.1", "Press and hold the Widget power button."))
        rival = make_evidence(
            make_chunk("Widget §Setup.2", "The Widget setup takes 5 minutes."), retrieval_event_id="r-2"
        )
        self.assertEqual(conflicting_evidence(top, (top, rival)), ())


class OptionalNliVerifierTests(unittest.TestCase):
    def test_nli_verifier_fails_closed_without_its_optional_extra(self):
        # module-05 §2: the optional NLI verifier must fail closed, never fall
        # back to extractive verification.
        if importlib.util.find_spec("sentence_transformers") is not None:
            self.skipTest("optional NLI extra is installed; the failure path cannot be exercised")
        with self.assertRaises(RuntimeError) as raised:
            NliEntailmentVerifier()
        self.assertIn("sentence-transformers", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
