# Module 07 — WebSocket Server and HTTP API

**STATUS: BUILD.** No code is shipped. Build to this spec.

## 1. Purpose

The delivery layer: a FastAPI application that mounts the Module 09 frontend,
exposes a small HTTP API, and owns the WebSocket `/ws/stream` endpoint that
streams transcript hypotheses into a `LiveConnection` (Module 06). Includes the
CLI entry point.

## 2. Files to create

- `prototype/server.py` — FastAPI app, `TranscriptMessage` model, endpoints.
- `prototype/server_types.py` — `make_transcript_message(**values)` that imports
  `TranscriptMessage` lazily (breaks the server ↔ benchmark import cycle).
- `prototype/__main__.py` — CLI: argparse `--backend` (`lightweight|dense`),
  `--parser` (`rules|spacy|bert|shadow|auto`), `--port` (default 8000); set
  `CITEFRONTIER_BACKEND`/`CITEFRONTIER_PARSER`, then
  `uvicorn.run("prototype.server:app", host="127.0.0.1", port=args.port)`.

Dependencies: `fastapi==0.115.12`, `uvicorn==0.34.2`, `websockets==15.0.1`,
`httpx==0.28.1` (+ Module 03–06 requirements).

## 3. Inbound message model

`TranscriptMessage` (Pydantic, `model_config = ConfigDict(extra="forbid")`):

| Field | Type | Constraint |
|---|---|---|
| `event_id` | `str` | 1..100 |
| `turn_id` | `str` | 1..100 |
| `text` | `str` | 1..4000, non-blank (validator) |
| `timestamp_ms` | `float` | ≥ 0, finite (`allow_inf_nan=False`) |
| `is_final` | `bool` | default False |
| `revision_of` | `str\|None` | ≤ 100 |
| `target_intent_key` | `str\|None` | ≤ 500 |
| `sequence` | `int\|None` | ≥ 0 |

## 4. Lifespan

On startup build one `LiveRuntime` via `asyncio.to_thread(LiveRuntime)` into
`app.state.runtime`; init `app.state.sessions = {}`. Clear sessions on
shutdown. Mount `/static` from `prototype/static`.

## 5. HTTP endpoints

| Method & path | Behavior |
|---|---|
| `GET /` | serve `prototype/static/index.html`, `Cache-Control: no-store`. |
| `GET /health` | `{status, backend, corpus_chunks, generation, parser, corpus_hash}`. |
| `GET /corpus` | bundled corpus chunks as JSON. |
| `GET /demo-scenarios` | if `CITEFRONTIER_DEMO != "1"` return `{}`; else load `prototype/demo/scenarios.json` (return `{}` if absent). Retrieval never reads this file. |
| `GET /session/{id}/telemetry` | the session's telemetry; requires bearer token → else 403 (404 if session unknown). Use `secrets.compare_digest` against `connection.token`. |
| `GET /session/{id}/corpus` | the session's (possibly uploaded) corpus; token-gated like telemetry. |
| `GET /scorecard` | if `prototype/reports/scorecard.json` exists return it, else `{"status":"pending",...}`. |

## 6. WebSocket `/ws/stream`

1. **Origin guard.** If `Origin` header present and not equal to
   `http(s)://<host>` → close 1008. CLI clients without Origin are allowed.
2. Accept; `connection = runtime.connection()`; register in `app.state.sessions`;
   send `ready`:
   `{type:"ready", session_id, session_token, backend, parser,
   corpus_hash}`.
3. Loop (≤ 500 messages), each step awaiting `ws.receive_text()` with a 900 s
   idle timeout:
   - Wire cap: reject raw payloads > 1_000_000 bytes.
   - `corpus_upload` envelope (`{type, files:[{name, text}]}`): drain pending;
     `chunks = parse_upload(files)` (Module 08); build a new
     `LiveRuntime(chunks, backend, parser="spacy")` off-thread; close old
     connection; swap session; send `corpus_loaded` with the new session id,
     token, corpus_hash, chunk/doc counts, backend, parser. Invalid uploads
     leave the session intact and send `type:error`.
   - `benchmark_run` envelope (`{type, cases}`): drain pending; result =
     `run_benchmark(connection.runtime, cases)` (Module 08); send
     `benchmark_result`.
   - Else: 16 000-char message cap; validate with
     `TranscriptMessage.model_validate_json`; ≤ 8 pending events; dispatch
     `respond(message)` as a task (`respond` calls
     `connection.handle`, maps `ValueError`→`error` detail, other exceptions→
     generic "Processing failed; no answer was committed. Start a new
     session."); serialize all sends under one `send_lock`.
4. On `WebSocketDisconnect`/timeout: close the connection, cancel pending,
   drop the session.

## 7. Session authorization

- Token issued in `ready`/`corpus_loaded`.
- `GET /session/{id}/telemetry` and `/corpus` compare the `Authorization:
  Bearer <token>` header with `secrets.compare_digest`.
- All state is in-memory and dropped on disconnect; nothing persists across
  sessions.

## 8. Interfaces

- Calls `LiveConnection.handle` (06), `parse_upload`/`run_benchmark` (08),
  `make_transcript_message` (shared constructor, 07).
- Consolidated by Module 09 (browser) and by `websockets`-based scripts.

## 9. Acceptance criteria

- `ready` carries the corpus hash; handle/update flow works via
  `fastapi.testclient.TestClient`.
- Unknown session → 404; wrong/missing token → 403; after disconnect the
  session is gone (404).
- Invalid input (`{"text":""}`) replies `type:error` without disconnecting;
  a subsequent valid message still processes.
- 60-scenario WebSocket replay (in `docs/testing.md`) passes end-to-end.