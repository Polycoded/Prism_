# Module 05 — Session-Aware Synthesis

**STATUS: BUILD.** No code is shipped. Build to this spec.

This module turns decomposed intents and verified evidence into **versioned
claims and answers**, and owns the session-scoped machinery that makes answers
safe: exact-extractive verification, citation guarding, answer versioning,
targeted delta refinement, dialogue-context reuse, and abstention.

## 1. Scope

Create these files:

- `citefrontier/grounding.py` — verifiers, claim-slot selector, citation guard.
- `citefrontier/session.py` — ephemeral session state + claim/answer synthesis.
- `citefrontier/engine.py` — the core-library orchestration that runs a
  transcript event stream through controller + retriever + session.
- The claim builder and answer-versioning helpers the live runtime uses are
  specified in §7 of this module (the runtime calls these semantics directly).

## 2. Verification contract (`citefrontier/grounding.py`)

### Entailment
```python
@dataclass(frozen=True)
class EntailmentResult:
    entailed: bool; score: float; label: str

class EntailmentVerifier(Protocol):
    def verify(self, claim: str, evidence_text: str) -> EntailmentResult: ...

class ExtractiveEntailmentVerifier:
    def verify(self, claim, evidence_text):
        # entailed iff normalized(claim) is a substring of normalized(evidence)
```
`normalized` = space-joined lowercase alnum tokens (from `citefrontier/text.py`).
Set `os.environ.setdefault("HF_HUB_DISABLE_XET","1")` to keep model downloads
reproducible.

Optional `NliEntailmentVerifier(model_name="cross-encoder/nli-MiniLM2-L6-H768",
threshold=0.70)` — scores `(evidence, claim)`, requires the entailment label
with softmax probability ≥ threshold. Fail closed with a clear message if
optional deps are missing.

### Selector and guard
```python
@dataclass(frozen=True)
class ClaimEvidenceSelection:
    evidence: Evidence | None; verification: EntailmentResult | None
    evaluated_candidate_ids: tuple[str, ...]

class ClaimEvidenceSelector:
    def __init__(self, verifier=None):  # default ExtractiveEntailmentVerifier
    def select(self, claim: str, candidates: tuple[Evidence, ...]) -> ClaimEvidenceSelection:
        # verify claim against each candidate in order; keep entailed ones;
        # pick max by (score, -index) for stable tie-break; else (None, None, ids)

@dataclass(frozen=True)
class CitationGuardDecision:
    accepted_citation_ids: tuple[str, ...]; rejected_citation_ids: tuple[str, ...]

class CitationCandidateGuard:
    def enforce(self, proposed_citation_ids, candidates) -> CitationGuardDecision:
        # accepted = proposed ids that ARE chunk_ids in the current candidate set
```
The guard is what makes fabricated citations structurally impossible.

## 3. Session state and synthesis primitives (`citefrontier/session.py`)

```
class EphemeralSession:
    session_id; provisional: dict[str, Evidence-tuple]   # keyed by retrieval event id
    committed: dict[str, Evidence-tuple]                 # keyed by intent key
    intent_by_key: dict[str, Intent]
    versions: list[AnswerVersion]
    invalidated_provisionals: set[str]
    root_query: str
    citation_guard: CitationCandidateGuard

    latest -> AnswerVersion | None
    has_answer -> bool
    stage_provisional(event_id, intent, evidence)
    invalidate_provisionals() -> tuple[event_id,...]
    commit(intent, evidence)
    remember_committed_turn(intents)        # keeps only user intent text
    contextualize(intent, max_prior_turns=1) -> (query, context_intent_keys)
    synthesize_initial(root_query) -> AnswerVersion
    refine(target_intent_key, delta_intent, evidence) -> AnswerVersion
```

### `_claim_for(intent_key, evidence, claim_id) -> Claim`
1. Filter evidence whose `retrieval_event_id` is NOT in
   `invalidated_provisionals`.
2. Empty or guard-rejected → Claim with
   `text = "I could not verify: {intent.query}.", citations=(), evidence_event_ids=()`
   (abstention).
3. Top evidence chunk text becomes the claim text; citations = the guard's
   accepted id. Raise `CitationValidationError` if a citation is not in the
   current committed evidence set (cite-nothing-not-committed as an exception).

### Dialogue context (`contextualize`)
For elliptical follow-ups: `content_tokens(query) < 5` or the query starts
with `what about | what does that | what does it | and what | how about | is
that | yes | no | i do not | i dont`. When elliptical and prior user intents
exist, return `(context|"{query} {context}", prior keys)` where `context` is
the last turn's user intent text joined — never assistant or answer text. Pure
deictic forms reuse the last user intent verbatim; more specific short
questions keep their wording as a sidecar.

### Answer versioning
`AnswerVersion(version, parent_version, claims, changed_claim_ids)` with
`render(bullets=False)` → lines `{text} [{citations}]` or `- {text} [...]`.
`synthesize_initial` sets version 1 (or latest+1) and marks every claim as
changed. `refine` increments the version, replaces only the target intent's
claim (`id = "{claim_id}-v{version+1}"`), preserving the others and their ids.

## 4. Core-library orchestration (`citefrontier/engine.py`)

```
class CiteFrontierEngine(retriever, telemetry=None, use_dialogue_context=False,
                         claim_evidence_selector=None):
    new_session() -> EphemeralSession
    begin_turn(session)        # reset controller only
    process_stream(session, event) -> EngineOutput
    refine_with_late_detail(session, timestamp_s, target_intent_key,
                            late_detail) -> EngineOutput
```

`process_stream`:
1. `controller.decide(event, has_answer=session.has_answer)` (Module 04);
   record `controller_decision` telemetry.
2. If `invalidates_provisional` → `session.invalidate_provisionals()` +
   `provisional_invalidated` telemetry.
3. `SUPPRESS` → re-render latest with bullets; no retrieval.
4. `PROVISIONAL_RETRIEVE` → per intent `retriever.search(query, id,
   trigger="provisional")`, `stage_provisional`; return (no claims).
5. `COMMIT_RETRIEVE` → clear committed; retrieve each intent
   (`trigger="endpoint_commit"`) across a bounded `ThreadPoolExecutor`
   (max 4); for each: `_select_claim_evidence` (selector + guard) then
   `session.commit`; then `session.synthesize_initial` + `answer_version`
   telemetry.

`_select_claim_evidence(session, retrieval_id, evidence, timestamp_s)`:
`draft_claim = evidence[0].chunk.text if evidence else ""`;
`selection = selector.select(draft_claim, evidence)`;
`guard = citation_guard.enforce((selection.evidence.chunk_id,), evidence)`;
commit evidence only when the guard accepts. Record `claim_evidence_selected`
telemetry. Only endpoint/delta results may call this — provisional results
never do.

`refine_with_late_detail`: build a delta intent
(`key=f"{target.key}-delta-v{n}", query=f"{target.query} {late_detail}"`),
one `late_detail_delta` retrieval, `session.refine`, `answer_version` event.

## 5. Records (`citefrontier/models.py` extensions, alongside Module 02)

```python
@dataclass(frozen=True)
class Intent:    key: str; query: str          # minimal form for this module
@dataclass(frozen=True)
class TranscriptEvent:  timestamp_s: float; text: str; is_final: bool = False
@dataclass(frozen=True)
class Claim: claim_id; intent_key; text; citations; evidence_event_ids
@dataclass(frozen=True)
class AnswerVersion: version; parent_version; claims; changed_claim_ids
@dataclass(frozen=True)
class EngineOutput: action; answer; rendered_answer; retrieval_event_ids
```

## 6. Telemetry (`citefrontier/telemetry.py`)

`TelemetryRecorder.record(event_type, timestamp_s, session_id, **payload)`,
`.by_type`, `.write_jsonl`. `TraceEvent(event_type, timestamp_s, session_id,
payload)` is serializable via `asdict`.

## 7. The live-runtime claim builder (used by Module 06)

The production runtime implements the same synthesis directly:

### Conflict check (before committing the top candidate)
For each alternative candidate of the same intent, compare with the top: same
section root (`section.split(".")[0]`), same entity, a shared non-generic
property (`intent.topic - {period,month,year,day,people}` intersecting both
chunks' topic tokens), and competing numeric quantities
(`\b\d+[ -](months?|years?|days?|people)\b`). If any compete → abstention text
`"The retrieved sources conflict on this request: {query}."` with
`withheld_reason="conflicting_source_quantities"`, no citation.

### Claim record shape
```json
{"claim_id":"c-<turn>-<key8>","claim_revision":1,"intent_key":"…",
 "query":"…","dependencies":{"entities":[…],"topic":[…],"constraints":[…]},
 "text":"…","citations":["…"],"evidence_event_ids":["r-…"],
 "verification":"exact_extractive"|"uncertain","evidence":[…],
 "candidate_ids":[…],"verified_turn_id":"…","withheld_reason":null}
```
Evidence items copy the chunk and add `source_offsets{start,end,basis:
"normalized_source"}`, `corpus_hash`, `phase:"committed"`, `final_turn_id`,
`intent_revision`, `source_hash`.

### Claim commit rule
Only when candidates exist, the top `evidence_id` is in its candidate set,
and `ExtractiveEntailmentVerifier.verify(chunk.text, chunk.text).entailed` →
exact extract. Otherwise `"I could not verify: {query}."`,
`verification="uncertain"`, `withheld_reason="insufficient_relevant_evidence"`.

### Targeted refinement (`compute delta for text, target_key`)
- Requires an existing answer; if `target_key` given it must name a current
  intent.
- Detection: `target_key` set, or text matches
  `\s*(?:actually|instead|what about|how about|change|make (?:it|that)|for
  (?:the )?\w+[, ]+what about)\b`.
- **Entity change** (exactly one named entity + `instead|actually|change`):
  rewrite the selected intent's query, replacing the old alias with the new
  alias.
- **Property detail**: strip `for X,` / `what about` / `actually` prefixes and
  append the detail to the old query; a numeric-constraint replacement is
  handled specially (replace `N units` with the new value, else append).
- Re-decompose the rewritten query; if it yields >1 intent, clarify
  ("refine one claim at a time"). Selection prefers a fronted property; else
  the prior intent with the best topic overlap; else clarify
  ("Which part should change? …").
- Returns `{old_intent_key: new_intent} | None`, plus an optional clarifier.

### Late-detail additions (declarative follow-ups)
Only when there is an answer, the text starts with `the|this|that|it|they|
their`, contains no question word (`what|when|where|how|which|who|tell|show|
find|list`), refers to exactly one prior entity, and the new terms overlap the
prior topics. Each `;`/`and`-separated clause (≥2 terms) becomes a new scoped
intent appended to the answer.

### Answer versioning (live)
- Non-refinement: all new claims, `version += 1`, patch
  `{version, parent_version, added, removed, replaced, preserved}` appended to
  `history`.
- Refinement: unchanged claims keep id/text/citation; changed claims keep
  `claim_id` but `claim_revision += 1`; `replaced` = changed∩old ids;
  `preserved` = (new∩old) − replaced.

## 8. Interfaces

- Consumed by Module 06 (calls the claim builder, refinement, versioning, and
  the engine/session primitives), Module 07 (`GET /session/{id}/telemetry`),
  Module 09 (claims/answers render from the update envelope).
- Depends on Modules 01 (Intent, aliases, classifiers), 02 (CorpusChunk,
  verifier's evidence chunks, `SearchIndex.finalize` output), 03 (normalized
  text), 04 (decision).
- Emits telemetry: `claim_verified`, `answer_version`, `claim_evidence_selected`
  (engine flavor), `transcript_normalized` is recorded by Module 06.

## 9. Acceptance criteria

- Utterance parts of this module must pass the behaviors in `docs/testing.md`:
  provisional never commits; correction invalidates provisionals before the
  final answer; unaffected claim preserved with the same claim_id; targeted
  delta uses exactly one retrieval; conflicting quantities abstain;
  instruction-like passages abstain; unknown topic abstains; elliptical
  follow-up uses user-intent context only; citations always resolve to
  committed candidates.