# Module 04 — Retrieval Controller

**STATUS: SHIPPED — reference implementation added.** The blueprint below
remains the specification of record; the code may be treated as the reference
implementation of it.

## 1. Purpose

Decide **when** retrieval may start from partial transcript text and **when**
the system must wait or suppress. Tentative candidates may be collected early,
but they can never synthesize a claim; only the final-event controller decision
(`commit_retrieve`) permits claim emission.

## 1.1 Reference files (shipped)

| File | Role |
|---|---|
| `citefrontier/controller.py` | Dependency-light `RetrievalController` (blueprint §3). |
| `prototype/controller.py` | Full `decide(...)` decision function used by the live runtime (blueprint §4). |
| `tests/test_controller.py` | Unit tests for both flavors (blueprint §6). |

Both reuse the shipped shared records in `citefrontier/models.py`
(`Decision`, `ControllerAction`, `TranscriptEvent`) and the text helpers in
`citefrontier/text.py` (`content_tokens`, `normalized`,
`is_presentation_request`, `is_non_retrieval_chitchat`,
`has_explicit_correction`, `split_intents`).

## 2. Shared records

Create `citefrontier/models.py` extensions (or place them where the runtime
needs them):

```python
class Decision(str, Enum):
    WAIT = "wait"
    PROVISIONAL_RETRIEVE = "provisional_retrieve"
    COMMIT_RETRIEVE = "commit_retrieve"
    SUPPRESS = "suppress"

@dataclass(frozen=True)
class ControllerAction:
    decision: Decision
    intents: tuple[Intent, ...]
    reason: str
    invalidates_provisional: bool = False
```

Note: `citefrontier/models.py` and `citefrontier/text.py` ship with Module 02.
`citefrontier/text.py` already provides `content_tokens`, `normalized`,
`is_presentation_request`, `is_non_retrieval_chitchat`,
`has_explicit_correction`, `split_intents`; reuse them rather than duplicating.

## 3. Dependency-light controller (`citefrontier/controller.py`)

```
RetrievalController
  previous_partial: str = ""
  emitted_intents: set[str] = set()
  decide(event: TranscriptEvent, has_answer: bool = False) -> ControllerAction
```

`TranscriptEvent(timestamp_s, text, is_final)`. Decision rules, in order:

1. `is_non_retrieval_chitchat(text)` → `SUPPRESS`, reason
   `non_retrieval_chitchat`.
2. `has_answer and is_presentation_request(text)` → `SUPPRESS`, reason
   `presentation_restructure_of_existing_answer`. (`is_presentation_request`
   returns False when the text contains `and|also|but`, so a mixed factual
   request is never suppressed.)
3. `intents = split_intents(text)`; `current = normalized(text)`;
   `previous = normalized(previous_partial)`;
   `invalidates = has_explicit_correction(text) or (previous and not
   current.startswith(previous))`.
4. `event.is_final` → COMMIT_RETRIEVE, reason `final_transcript_endpoint`;
   store the event text and remember all intent keys.
5. No `previous` → WAIT, `first_partial_has_no_stability_history`.
6. `current` does not start with `previous` → WAIT,
   `partial_revision_wait_for_recommitment`, `invalidates_provisional=True`.
7. `new_suffix = current[len(previous):]`; `enough = len(content_tokens(
   new_suffix)) >= 3`; `new_intents` = intents whose key was not already
   emitted. If `enough and new_intents` → PROVISIONAL_RETRIEVE with the new
   intents, reason `stable_prefix_with_new_searchable_intent`.
8. Otherwise WAIT, `partial_is_not_yet_a_new_stable_searchable_intent`.

`has_explicit_correction` is True when ` normalized` contains
` instead | actually | rather than ` as a word boundary.

## 4. Full controller decision (`prototype/controller.py`)

The live runtime computes flags first (using Module 01 classifiers and Module
05 refinement deltas), then calls a `decide` compatible function:

```
def decide(*, text, parsed, previous, is_final, is_format, is_social,
           clarification, has_updates, has_additions,
           presentation_prefix_hit) -> (decision: str, reason: str)
```

where `parsed` is the current event's decomposed intents and `previous` is the
prior event's decomposed intents (the runtime maintains that rolling value).
Rules, in order:

1. `is_format or is_social` → `suppress`, reason `presentation_only` or
   `social_turn`.
2. `clarification` present → `wait`, reason `ambiguous_refinement`.
3. `is_final` → `commit_retrieve`, reason `targeted_delta` if
   `has_updates or has_additions` else `final_transcript`.
4. `presentation_prefix_hit` → `wait`, reason `possible_presentation_request`.
5. Partial stability: `stable` = parsed intents from the current text such that
   `i.topic` is non-empty and some previous intent has the same or a missing
   entity and `set(prior.topic) <= set(intent.topic)`. `incomplete` = the text
   ends with a connective
   (`\b(?:and|or|not|without|between|instead of|rather than|for|in|with)\s*$`,
   case-insensitive). If `stable and not incomplete and len(terms(text)) >= 3`
   → `provisional_retrieve`, reason `stable_searchable_intents`.
6. Else → `wait`, reason `awaiting_stable_intent`.

Reasons for `wait` also include `possible_presentation_request`
(Module 01 `presentation_prefix`) and the clarifiers `ambiguous_refinement`,
`translation_unsupported` (Module 01 `translation_request`), and the
multi-product clarifier.

## 5. Interfaces

- Consumed by Module 06 runtime and by the core-library engine orchestration.
- Emits `ControllerAction`/`(decision, reason)`; the runtime records
  `controller_decision` telemetry with decision, reason, and revision.
- Depends on: Module 01 classifiers (`presentation`, `social`,
  `presentation_prefix`, `translation_request`), Module 03 normalization
  output, and Module 05 refinement detection (`has_updates`/`has_additions`).

## 6. Acceptance criteria

- First partial → WAIT (no history).
- A partial extending the previous text by ≥ 3 content tokens with a new
  intent → PROVISIONAL_RETRIEVE; the same intent already emitted → WAIT.
- A non-prefix revision → WAIT with `invalidates_provisional=True`.
- Final event → COMMIT_RETRIEVE.
- Chitchat → SUPPRESS; a presentation request with an existing answer →
  SUPPRESS; a mixed factual request is never suppressed.
- Word-by-word formatting prefixes ("plea", "please", "please repeat", …)
  must not trigger retrieval (Module 01 `presentation_prefix` → WAIT).
- The 60-scenario acceptance behavior in `docs/testing.md` must reproduce the
  decision/reason pairs listed there.