from __future__ import annotations

import json
import re
from collections import deque
from typing import Any


BOTIFIED_ALLOWED_EVENTS = (
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
)

_BOTIFIED_OPEN = "<botified>"
_BOTIFIED_CLOSE = "</botified>"
_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def format_botified_frame(event: dict[str, Any], timeout_secs: int = 8) -> str | None:
    if not isinstance(event, dict):
        return None
    if event.get("type") != "semantic_event":
        return None
    if event.get("event") not in BOTIFIED_ALLOWED_EVENTS:
        return None

    event_id = event.get("event_id")
    if not isinstance(event_id, str) or _EVENT_ID_RE.fullmatch(event_id) is None:
        return None

    payload = {
        "id": f"visual:{event_id}",
        "urgency": "normal",
        "timeout_secs": int(timeout_secs),
        "request": _format_request(event),
        "expect": "ack",
    }
    inner = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    inner = inner.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{_BOTIFIED_OPEN}{inner}{_BOTIFIED_CLOSE}"


class BotifiedEventMapper:
    def __init__(self, max_seen_event_ids: int = 1024) -> None:
        self._max_seen_event_ids = max(0, int(max_seen_event_ids))
        self._seen_event_ids: set[str] = set()
        self._seen_order: deque[str] = deque()

    def frames_from_visual_state(self, visual_state: dict[str, Any]) -> list[str]:
        semantic_events = visual_state.get("semantic_events")
        if not isinstance(semantic_events, list):
            return []

        frames: list[str] = []
        emitted_this_frame: set[str] = set()
        for event in semantic_events:
            if not isinstance(event, dict):
                continue
            event_id = event.get("event_id")
            if not isinstance(event_id, str):
                continue
            if event_id in self._seen_event_ids or event_id in emitted_this_frame:
                continue

            frame = format_botified_frame(event)
            if frame is None:
                continue

            frames.append(frame)
            emitted_this_frame.add(event_id)
            self._remember(event_id)

        return frames

    def _remember(self, event_id: str) -> None:
        if self._max_seen_event_ids == 0:
            return
        self._seen_event_ids.add(event_id)
        self._seen_order.append(event_id)
        while len(self._seen_order) > self._max_seen_event_ids:
            expired_event_id = self._seen_order.popleft()
            self._seen_event_ids.discard(expired_event_id)


def _format_request(event: dict[str, Any]) -> str:
    return (
        f"event={event.get('event')} "
        f"camera={event.get('camera')} "
        f"track_id={event.get('track_id')} "
        f"confidence={event.get('confidence')} "
        f"duration_ms={event.get('duration_ms')} "
        f"text={event.get('text')}"
    )
