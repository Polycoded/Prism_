"""Retrieval timing policy for incoming transcript chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ControllerAction, Decision, TranscriptEvent
from .text import (
    content_tokens,
    has_explicit_correction,
    is_non_retrieval_chitchat,
    is_presentation_request,
    normalized,
    split_intents,
)


@dataclass
class RetrievalController:
    """A disclosed, deterministic WAIT/RETRIEVE/SUPPRESS controller.

    A partial becomes provisionally searchable only after it extends the
    prior transcript by at least three content tokens. This is intentionally
    inspectable and will later be compared with a model-based controller.
    """

    previous_partial: str = ""
    emitted_intents: set[str] = field(default_factory=set)

    def decide(self, event: TranscriptEvent, has_answer: bool = False) -> ControllerAction:
        if is_non_retrieval_chitchat(event.text):
            return ControllerAction(
                decision=Decision.SUPPRESS,
                intents=(),
                reason="non_retrieval_chitchat",
            )
        if has_answer and is_presentation_request(event.text):
            return ControllerAction(
                decision=Decision.SUPPRESS,
                intents=(),
                reason="presentation_restructure_of_existing_answer",
            )

        intents = split_intents(event.text)
        current = normalized(event.text)
        previous = normalized(self.previous_partial)
        invalidates = has_explicit_correction(event.text) or bool(previous and not current.startswith(previous))

        if event.is_final:
            self.previous_partial = event.text
            self.emitted_intents.update(intent.key for intent in intents)
            return ControllerAction(
                decision=Decision.COMMIT_RETRIEVE,
                intents=intents,
                reason="final_transcript_endpoint",
                invalidates_provisional=invalidates,
            )

        if not previous:
            self.previous_partial = event.text
            return ControllerAction(Decision.WAIT, (), "first_partial_has_no_stability_history")

        if not current.startswith(previous):
            self.previous_partial = event.text
            return ControllerAction(
                Decision.WAIT,
                (),
                "partial_revision_wait_for_recommitment",
                invalidates_provisional=invalidates,
            )

        new_suffix = current[len(previous) :]
        enough_new_content = len(content_tokens(new_suffix)) >= 3
        new_intents = tuple(intent for intent in intents if intent.key not in self.emitted_intents)
        self.previous_partial = event.text

        if enough_new_content and new_intents:
            self.emitted_intents.update(intent.key for intent in new_intents)
            return ControllerAction(
                Decision.PROVISIONAL_RETRIEVE,
                new_intents,
                "stable_prefix_with_new_searchable_intent",
                invalidates_provisional=invalidates,
            )

        return ControllerAction(
            Decision.WAIT,
            (),
            "partial_is_not_yet_a_new_stable_searchable_intent",
            invalidates_provisional=invalidates,
        )