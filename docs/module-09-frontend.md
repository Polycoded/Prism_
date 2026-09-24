# Module 09 — Frontend Web App

**STATUS: BUILD.** No code is shipped. Build to this spec.

## 1. Purpose

A single-page static client served by Module 07. It opens the WebSocket,
streams partial hypotheses as the user types, renders committed claims with
citations and exact source inspection, refines single claims, reformats the
answer as bullets, uploads an organizer corpus, and runs a labeled benchmark.
All behavior must work without the motion layer and honor
`prefers-reduced-motion`.

## 2. Files to create under `prototype/static/`

| File | Role |
|---|---|
| `index.html` | Markup: header, sidebar (scenario buttons, corpus note, disclosure), pipeline stages, proof strip, composer, answer/evidence panes, trace pane, dialogs (upload, corpus, validation). |
| `style.css` | Layout with focus vs inspector mode; responsive breakpoints 390/768/1440 px; no horizontal overflow. |
| `app.js` | All behavior (WebSocket client, renderer, upload, benchmark). |
| `motion.js` | `CiteMotion` optional GSAP layer (no-op guard). |
| `vendor/gsap.min.js`, `vendor/GSAP-NOTICE.txt` | Vendored GSAP 3.13.0 + license provenance. |
| `favicon.svg` | Brand mark. |

## 3. Connection and streaming

- On load: `connect()` opens `(wss|ws)://<host>/ws/stream`, then fetches
  `/demo-scenarios` (returns `{}` unless `CITEFRONTIER_DEMO=1`); empty →
  scenario/replay controls disabled.
- Typing: after a 650 ms debounce, send the current textarea value as a
  non-final event; keep a single pending partial queue; strictly increasing
  `sequence`.
- Submit: **Ask question** button or `Ctrl+Enter` → `is_final: true`.
- Every message: `{event_id, turn_id, sequence, timestamp_ms, text,
  is_final, revision_of?, target_intent_key?}` with a 60 s timeout.
- `duplicate` responses are ignored by the renderer.

## 4. Rendering (`update`)

- Timeline: prepend each `retrieval_events` record (≤ 60 entries) with event
  type + a short human summary (reason/query/verification/version/counts).
- Stage badge: `wait`→Understand, `provisional_retrieve`→Retrieve early,
  `commit_retrieve`/`suppress`→Answer.
- Sub-queries: `sub_queries` as `N · query` chips.
- Corrections: on a `transcript_normalized` event show
  "Understood as: <normalized_text>".
- Claims: article per claim with label
  (`SOURCE-SUPPORTED EXTRACT` | `INSUFFICIENT EVIDENCE`) and
  New/Updated/Unchanged; the extract text; a citation button opening the
  evidence inspector (exact quote + doc/section/evidence id/retrieval/
  source/offsets + "Exact source-text match" note); a **Refine this claim**
  inline form that sends `target_intent_key` in a new turn; a diff
  `<details>` for changed wording of updated claims.
- Versions: `answer_version` and the latest `history` patch
  (`added/replaced/preserved`).
- Presentation: on a suppress/bullets decision, `presentation_items` groups
  claims into numbered bullet groups; session stats stay unchanged.

## 5. Controls

- **Show as bullets** → send `Put that in bullets` (final).
- **Copy answer with citations** → clipboard lines `text [chunk_id]`.
- **Export audit** → JSON download `{session_id, corpus, corpus_hash, events}`.
- **New session** → close + reconnect, clear workspace, restore bundled label.
- **Organizer corpus dialog** — file picker (`.md|.txt|.json`); apply sends
  `corpus_upload`; on `corpus_loaded` swap session and disable bundled replay/
  benchmark; **Run benchmark** reads the JSON file and sends `benchmark_run`.
- **Validation dialog** — `GET /scorecard`; render "a replay run has not been
  recorded" when pending; otherwise a neutral scenario/metric table with an
  explicit "not a certified evaluation" disclaimer.

## 6. Motion layer

`window.CiteMotion = {stage, claims, intents, trace, evidence, inspector,
dialog}`; each call guarded by (`gsap` presence + `matchMedia('(prefers-
reduced-motion: reduce)')`); fall back to a no-op Proxy in `app.js`. Vendored
GSAP via `defer`; no network requests.

## 7. Interfaces

- HTTP: `/demo-scenarios`, `/scorecard`, `/session/{id}/corpus` (with bearer
  token).
- WebSocket: `ready`, `update`, `corpus_loaded`, `benchmark_result`, `error`
  (Modules 07/10).
- Never reads retrieval internals; renders only the server envelope.

## 8. Acceptance criteria

- No voice/microphone controls exist.
- Focus mode is the default; inspector toggle reveals evidence/trace.
- Claims, citation inspector, diff, bullets suppression (search count
  unchanged), and clarification ("more than one product") all render.
- Responsive at 390/768/1440 px with zero horizontal overflow; zero JS errors
  in tested flows; reduced-motion honored.
- Upload flow: reject training JSON with a visible "not accepted" message,
  keep the bundled corpus across a failed upload, load an organizer corpus and
  answer from it, keep bundled scores absent, reset restores the bundled
  corpus.