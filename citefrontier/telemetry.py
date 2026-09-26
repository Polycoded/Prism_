"""Session telemetry recording for the core library (Module 05).

``TraceEvent`` is the frozen record from ``citefrontier.models``; this module
only owns the per-session log and its JSONL serialization.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import TraceEvent


class TelemetryRecorder:
    """An in-memory, insertion-ordered event log for one process."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(
        self, event_type: str, timestamp_s: float, session_id: str, **payload: object
    ) -> TraceEvent:
        event = TraceEvent(
            event_type=event_type,
            timestamp_s=timestamp_s,
            session_id=session_id,
            payload=dict(**payload),
        )
        self.events.append(event)
        return event

    def by_type(self, event_type: str) -> tuple[TraceEvent, ...]:
        return tuple(event for event in self.events if event.event_type == event_type)

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        if target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        return target
