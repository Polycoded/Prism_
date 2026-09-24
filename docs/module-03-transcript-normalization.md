# Module 03 — Transcript Normalization

**STATUS: BUILD.** No code is shipped. Create this module from scratch
following the spec exactly.

## 1. Purpose

Normalize raw spoken transcript text into the string the decomposer sees,
without any generative rewriting. Two deterministic transformations run, in
order, on every inbound event before any controller decision:

1. Disfluency cleanup — removes a narrow allowed set of speech fillers and
   immediately repeated words.
2. Correction resolution — resolves an explicit local value correction
   ("60, not 6", "Tuesday, actually Wednesday") or a trailing unambiguous
   subject correction ("…, I meant S2") against the corpus alias map.

Both must be fully deterministic and report exactly what changed.

## 2. Where it lives

Create `prototype/transcript.py` with the functions below. It must import
nothing but `re` (standard library) and depends on the alias map passed in.

## 3. Function contract

- `normalize_disfluencies(text) -> (cleaned_text, changes)` where `changes` is
  a list of `{"type": "filler"|"repetition", "text": <removed text>}`.
- `normalize_correction(text, aliases) -> (normalized_text, change|None,
  ambiguous)` where `change` is one of
  `{"type":"explicit_value","from":<old>,"to":<new>}`,
  `{"type":"accepted_value","from":<old>,"to":<new>}`, or
  `{"from":<old subject>,"to":<new subject>}`, and `ambiguous=True` when a
  correction cannot be resolved to exactly one replacement.

## 4. Behavior specification (reproduce exactly)

### 4.1 `normalize_disfluencies`

1. Filler regex (case-insensitive, word-boundary-safe):
   `(?<!\w)(?:um+|uh+|erm+|er+|ah+)(?!\w)[,\s]*`. The fillers are exactly
   `um/umm`, `uh`, `erm/er`, `ah` (optional trailing comma/space). Record each
   match as a `filler` change; replace with a single space.
2. Repetition loop: regex `\b([a-z][a-z0-9'-]{1,30})(?:\s+\1\b)+`; repeatedly
   collapse an immediately repeated word to one occurrence, recording each
   removal as a `repetition` change. (So `what what is` → `what is`.)
3. Cleanup: collapse `\s+([,?.!])` to `$1`; collapse runs of spaces
   (`\s{2,}` → ` `); strip leading/trailing ` ,`.

### 4.2 Explicit value correction (`_normalize_explicit_value_correction`,
called first by `normalize_correction`)

Fix the single-token pattern `_CORRECTION_VALUE =
[A-Za-z0-9][A-Za-z0-9:.%'-]*`.

- Pattern 1 — rejected value first:
  `<old> (—|–|-|,) (sorry(, I meant)?|no|I mean|actually|make that),? <new>`.
  If neither token matches model-id form `[A-Za-z]+\d+`, keep `<new>`, drop
  `<old>`; return `change={"type":"explicit_value","from":old,"to":new}`.
- Pattern 2 — accepted value first: `<new>,? not <old>` (same single token
  rule and the same model-id exclusion); keep `<new>`,
  `change={"type":"accepted_value","from":old,"to":new}`.
- Rewrite: `text[:match.start()] + new + text[match.end()]`, then
  collapse `\s+([,.?!])`, collapse spaces, strip ` ,`.
- Unmatched uses of "actually"/"not" are left untouched.

### 4.3 `normalize_correction` subject path

1. Trailing-subject regex (after the explicit-value step):
   `[,;.!?\s]+(?:sorry[,\s]+(?:I\s+meant\s+)?|I\s+meant\s+|actually[\s,]*
   (?:I\s+meant\s+)?)([^,;.!?]+)[.!?\s]*$`. Capture the prefix and the
   replacement.
2. `subjects` = aliases occurring in the prefix. None → return unchanged.
3. `candidates` = aliases equal to `replacement.strip()`, or (when exactly one
   subject) aliases in the same family whose final token equals the
   replacement (`LumaPad S1` → `S2` resolves to `LumaPad S2`).
4. If the match was not explicit (`sorry`/`I meant` absent), there are no
   candidates, and the replacement is not `[A-Za-z]+\s*\d+`, return unchanged.
5. Exactly one subject AND one candidate → substitute the subject in the
   prefix and return `{"from":subject,"to":candidate}` (not ambiguous).
6. Otherwise return `ambiguous=True`. Unknown replacements
   ("…, I meant S99") must be ambiguous so the earlier product never silently
   wins retrieval.

## 5. Config

None. The filler set and correction markers are fixed.

## 6. Interfaces

- Called by Module 06 at the top of each event's handling, before the
  controller decision. On any change, the runtime replaces the event's text
  and records a `transcript_normalized` telemetry event with `raw_text`,
  `normalized_text`, `correction`, `disfluencies`.
- Depends on the Module 01 alias map (`decomposer.aliases`).

## 7. Acceptance criteria

Behaviors that must hold:

- `'umm what what is uh the LumaPad S1 warranty'` →
  `'what is the LumaPad S1 warranty'` with changes `[filler, filler,
  repetition]`.
- `'The room is for 6—sorry, 7 people and what is parking like?'` →
  `'The room is for 7 people and what is parking like?'` (explicit_value
  6→7); also `6, make that 7`, `6, I mean 7`, `6, no, 7`.
- `'It is 7, not 6 people'` → `'It is 7 people'` (accepted_value).
- `'Book it Tuesday, actually Wednesday and include lunch'` →
  `'Book it Wednesday and include lunch'`.
- Model-like values keep alias safety: `'LumaPad S1 warranty, sorry I meant
  S99'` with aliases `{s1:'LumaPad S1', s2:'LumaPad S2'}` stays unchanged and
  ambiguous.
- `'LumaPad S1 warranty period, sorry I meant S2'` → `'LumaPad S2 warranty
  period'` (subject correction). A compound `'For LumaPad S1, …, I meant S2'`
  keeps both intents after re-decomposition.
- `'Does LumaPad S1 actually support wireless charging?'` unchanged.
- Multi-subject text (`'Compare LumaPad S1 and LumaPad S2 …, sorry I meant
  S1'`) is ambiguous.