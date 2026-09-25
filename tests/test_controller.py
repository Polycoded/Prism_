"""Unit tests for the Retrieval Controller (Module 04)."""
from __future__ import annotations

import unittest

from citefrontier.controller import RetrievalController
from citefrontier.models import Decision, TranscriptEvent
from prototype.controller import decide
from prototype.decomposition import Intent


def make_intent(key: str, query: str, topic: tuple[str, ...], entity_ids: tuple[str, ...] = ("D1",)) -> Intent:
    return Intent(
        key=key,
        query=query,
        subject="Widget 1",
        entity_ids=entity_ids,
        topic=tuple(sorted(topic)),
        constraints=(),
        source_spans=((0, len(query)),),
        method="structural_rules",
    )


class DependencyLightControllerTests(unittest.TestCase):
    def test_first_partial_waits(self):
        controller = RetrievalController()
        action = controller.decide(TranscriptEvent(0.0, "Find warranty details for", is_final=False))
        self.assertEqual(action.decision, Decision.WAIT)
        self.assertEqual(action.reason, "first_partial_has_no_stability_history")

    def test_extending_partial_triggers_provisional(self):
        controller = RetrievalController()
        controller.decide(TranscriptEvent(0.0, "Find warranty details for", is_final=False))
        action = controller.decide(
            TranscriptEvent(0.8, "Find warranty details for Model A device coverage", is_final=False)
        )
        self.assertEqual(action.decision, Decision.PROVISIONAL_RETRIEVE)
        self.assertEqual(action.reason, "stable_prefix_with_new_searchable_intent")
        self.assertTrue(action.intents)

    def test_emitted_intent_is_not_refetched(self):
        controller = RetrievalController()
        controller.decide(TranscriptEvent(0.0, "Find warranty details for", is_final=False))
        controller.decide(TranscriptEvent(0.8, "Find warranty details for Model A device coverage", is_final=False))
        action = controller.decide(
            TranscriptEvent(1.0, "Find warranty details for Model A device coverage and", is_final=False)
        )
        self.assertEqual(action.decision, Decision.WAIT)
        self.assertEqual(action.reason, "partial_is_not_yet_a_new_stable_searchable_intent")

    def test_final_commits(self):
        controller = RetrievalController()
        action = controller.decide(TranscriptEvent(0.0, "warranty model a", is_final=True))
        self.assertEqual(action.decision, Decision.COMMIT_RETRIEVE)
        self.assertEqual(action.reason, "final_transcript_endpoint")

    def test_non_prefix_revision_waits_and_invalidates(self):
        controller = RetrievalController()
        controller.decide(TranscriptEvent(0.0, "Model A warranty", is_final=False))
        action = controller.decide(TranscriptEvent(0.8, "Model B coverage details", is_final=False))
        self.assertEqual(action.decision, Decision.WAIT)
        self.assertEqual(action.reason, "partial_revision_wait_for_recommitment")
        self.assertTrue(action.invalidates_provisional)

    def test_explicit_correction_invalidates_on_final(self):
        controller = RetrievalController()
        controller.decide(TranscriptEvent(0.0, "Find warranty details for", is_final=False))
        controller.decide(TranscriptEvent(0.8, "Find warranty details for Model A device coverage", is_final=False))
        action = controller.decide(
            TranscriptEvent(1.2, "Find warranty details for Model A device coverage actually Model B instead",
                            is_final=True)
        )
        self.assertEqual(action.decision, Decision.COMMIT_RETRIEVE)
        self.assertTrue(action.invalidates_provisional)

    def test_chitchat_is_suppressed(self):
        action = RetrievalController().decide(TranscriptEvent(0.0, "hi", is_final=True))
        self.assertEqual(action.decision, Decision.SUPPRESS)
        self.assertEqual(action.reason, "non_retrieval_chitchat")

    def test_presentation_with_answer_is_suppressed(self):
        controller = RetrievalController()
        action = controller.decide(TranscriptEvent(0.0, "Please repeat my last answer in two bullets",
                                                   is_final=False), has_answer=True)
        self.assertEqual(action.decision, Decision.SUPPRESS)
        self.assertEqual(action.reason, "presentation_restructure_of_existing_answer")


class FullControllerDecisionTests(unittest.TestCase):
    def test_presentation_only(self):
        decision, reason = decide(text="Put that in bullets", parsed=(), previous=(),
                                  is_final=True, is_format=True, is_social=False,
                                  clarification=None, has_updates=False, has_additions=False,
                                  presentation_prefix_hit=False)
        self.assertEqual((decision, reason), ("suppress", "presentation_only"))

    def test_social_turn(self):
        decision, reason = decide(text="hello", parsed=(), previous=(),
                                  is_final=True, is_format=False, is_social=True,
                                  clarification=None, has_updates=False, has_additions=False,
                                  presentation_prefix_hit=False)
        self.assertEqual((decision, reason), ("suppress", "social_turn"))

    def test_clarification_waits(self):
        decision, reason = decide(text="warranty", parsed=(), previous=(),
                                  is_final=True, is_format=False, is_social=False,
                                  clarification="Which part should change?",
                                  has_updates=False, has_additions=False, presentation_prefix_hit=False)
        self.assertEqual((decision, reason), ("wait", "ambiguous_refinement"))

    def test_final_commits(self):
        decision, reason = decide(text="Widget 1 warranty period", parsed=(), previous=(),
                                  is_final=True, is_format=False, is_social=False,
                                  clarification=None, has_updates=False, has_additions=False,
                                  presentation_prefix_hit=False)
        self.assertEqual((decision, reason), ("commit_retrieve", "final_transcript"))

    def test_targeted_delta(self):
        decision, reason = decide(text="Actually warranty liquid damage exclusions", parsed=(), previous=(),
                                  is_final=True, is_format=False, is_social=False,
                                  clarification=None, has_updates=True, has_additions=False,
                                  presentation_prefix_hit=False)
        self.assertEqual((decision, reason), ("commit_retrieve", "targeted_delta"))

    def test_possible_presentation_prefix_waits(self):
        decision, reason = decide(text="please repeat", parsed=(), previous=(),
                                  is_final=False, is_format=False, is_social=False,
                                  clarification=None, has_updates=False, has_additions=False,
                                  presentation_prefix_hit=True)
        self.assertEqual((decision, reason), ("wait", "possible_presentation_request"))

    def test_stable_partial_retrieves(self):
        previous = (make_intent("k1", "widget 1 warranty", ("warranty", "widget")),)
        parsed = (make_intent("k2", "widget 1 warranty period", ("warranty", "widget", "period")),)
        decision, reason = decide(text="widget 1 warranty period", parsed=parsed, previous=previous,
                                  is_final=False, is_format=False, is_social=False,
                                  clarification=None, has_updates=False, has_additions=False,
                                  presentation_prefix_hit=False)
        self.assertEqual((decision, reason), ("provisional_retrieve", "stable_searchable_intents"))

    def test_unstable_partial_waits(self):
        previous = (make_intent("k1", "widget 1 warranty", ("warranty", "widget")),)
        parsed = (make_intent("k2", "what is the prize of widget 2", ("prize", "widget"), ("D2",)),)
        decision, reason = decide(text="what is the prize of widget 2", parsed=parsed, previous=previous,
                                  is_final=False, is_format=False, is_social=False,
                                  clarification=None, has_updates=False, has_additions=False,
                                  presentation_prefix_hit=False)
        self.assertEqual((decision, reason), ("wait", "awaiting_stable_intent"))

    def test_incomplete_suffix_waits(self):
        previous = (make_intent("k1", "widget 1 warranty", ("warranty", "widget")),)
        parsed = (make_intent("k2", "widget 1 warranty and", ("warranty", "widget")),)
        decision, reason = decide(text="widget 1 warranty and", parsed=parsed, previous=previous,
                                  is_final=False, is_format=False, is_social=False,
                                  clarification=None, has_updates=False, has_additions=False,
                                  presentation_prefix_hit=False)
        self.assertEqual((decision, reason), ("wait", "awaiting_stable_intent"))


if __name__ == "__main__":
    unittest.main()