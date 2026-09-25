"""Full retrieval-controller decision used by the live runtime.

This is a pure extraction of the runtime's controller decision (Module 04 of
the reconstruction blueprints). It classifies one normalized transcript event
into a `(decision, reason)` pair, given flags computed by the normalizer,
decomposer, and refinement logic of the surrounding modules.

The runtime calls this after it has:
- normalized the text (transcript normalization),
- parsed the current intents with the decomposer,
- detected refinement updates / late-detail additions,
- classified formatting and social requests,
so the function stays deterministic and side-effect free.
"""
from __future__ import annotations

import re

from .decomposition import terms
from .decomposition import Intent


def decide(
    *,
    text: str,
    parsed: tuple[Intent, ...],
    previous: tuple[Intent, ...],
    is_final: bool,
    is_format: bool,
    is_social: bool,
    clarification: str | None,
    has_updates: bool,
    has_additions: bool,
    presentation_prefix_hit: bool,
) -> tuple[str, str]:
    """Return the controller `(decision, reason)` for one event.

    Decisions: ``wait``, ``provisional_retrieve``, ``commit_retrieve``,
    ``suppress``. Reasons are the stable strings consumed by telemetry and the
    accept tests.
    """
    if is_format or is_social:
        return "suppress", "presentation_only" if is_format else "social_turn"
    if clarification:
        return "wait", "ambiguous_refinement"
    if is_final:
        return "commit_retrieve", "targeted_delta" if (has_updates or has_additions) else "final_transcript"
    if presentation_prefix_hit:
        return "wait", "possible_presentation_request"

    stable = tuple(
        intent
        for intent in parsed
        if intent.topic
        and any(
            (prior.entity_ids == intent.entity_ids or not prior.entity_ids and intent.entity_ids)
            and set(prior.topic) <= set(intent.topic)
            for prior in previous
        )
    )
    incomplete = bool(
        re.search(r"\b(?:and|or|not|without|between|instead of|rather than|for|in|with)\s*$", text, re.I)
    )
    if stable and not incomplete and len(terms(text)) >= 3:
        return "provisional_retrieve", "stable_searchable_intents"
    return "wait", "awaiting_stable_intent"