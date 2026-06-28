from __future__ import annotations

import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, TextIO


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
_PENDING_EVENTS = {"person_passing_by", "person_approaching_robot"}
_IMMEDIATE_EVENTS = {"person_stopped_near_robot", "person_waving"}
_SUPPRESSED_EVENTS = {"person_appeared"}
_EVENT_PRIORITY = {
    "person_waving": 50,
    "person_stopped_near_robot": 40,
    "person_left": 30,
    "person_approaching_robot": 20,
    "person_passing_by": 10,
    "person_appeared": 0,
}


@dataclass(frozen=True)
class BotifiedNotificationConfig:
    coalesce_window_ms: int = 800
    same_key_min_gap_ms: int = 8000
    global_60s_limit: int = 12
    burst_1s_limit: int = 3


@dataclass
class _PendingEvent:
    event: dict[str, Any]
    key: tuple[str, Any]
    aliases: frozenset[tuple[str, Any]]
    created_ms: int


def format_botified_frame(
    event: dict[str, Any],
    timeout_secs: int = 8,
    visual_context: dict[str, Any] | None = None,
) -> str | None:
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
        "request": _format_request(event, visual_context=visual_context),
        "expect": "ack",
    }
    inner = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    inner = inner.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{_BOTIFIED_OPEN}{inner}{_BOTIFIED_CLOSE}"


class BotifiedEventMapper:
    def __init__(
        self,
        max_seen_event_ids: int = 1024,
        *,
        config: BotifiedNotificationConfig | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._max_seen_event_ids = max(0, int(max_seen_event_ids))
        self._seen_event_ids: set[str] = set()
        self._seen_order: deque[str] = deque()
        self._config = config or BotifiedNotificationConfig()
        self._clock_ms = clock_ms or _wall_clock_ms
        self._pending_by_key: dict[tuple[str, Any], _PendingEvent] = {}
        self._last_emitted_by_key: dict[tuple[str, Any], int] = {}
        self._left_emitted_by_key: dict[tuple[str, Any], bool] = {}
        self._emitted_timestamps: deque[int] = deque()

    def frames_from_visual_state(self, visual_state: dict[str, Any]) -> list[str]:
        now_ms = int(self._clock_ms())
        semantic_events = visual_state.get("semantic_events")
        if not isinstance(semantic_events, list):
            semantic_events = []

        frames: list[str] = []
        emitted_this_frame: set[str] = set()
        for event in sorted(
            (event for event in semantic_events if isinstance(event, dict)),
            key=lambda item: -_EVENT_PRIORITY.get(str(item.get("event")), -1),
        ):
            if not isinstance(event, dict):
                continue
            event_id = event.get("event_id")
            if not isinstance(event_id, str):
                continue
            if event_id in self._seen_event_ids or event_id in emitted_this_frame:
                continue

            emitted_this_frame.add(event_id)
            self._remember(event_id)
            frame = self._handle_event(event, visual_state, now_ms=now_ms)
            if frame is None:
                continue
            frames.append(frame)

        frames.extend(
            self._flush_expired_pending(
                visual_state,
                now_ms=now_ms,
            )
        )
        return frames

    def _handle_event(
        self,
        event: dict[str, Any],
        visual_state: dict[str, Any],
        *,
        now_ms: int,
    ) -> str | None:
        event_name = event.get("event")
        if event_name not in BOTIFIED_ALLOWED_EVENTS:
            return None
        if event_name in _SUPPRESSED_EVENTS:
            return None

        key, aliases = self._event_keys(event, visual_state)
        if key is None:
            return None

        if event_name == "person_passing_by" and self._is_fast_passing(event, visual_state):
            return None

        pending_key = self._find_pending_key(key, aliases)

        if event_name in _PENDING_EVENTS:
            if self._same_key_gap_active(key, aliases, now_ms=now_ms):
                return None
            if pending_key is not None:
                self._pending_by_key.pop(pending_key, None)
            self._pending_by_key[key] = _PendingEvent(
                event=dict(event),
                key=key,
                aliases=frozenset(aliases),
                created_ms=now_ms,
            )
            self._left_emitted_by_key.setdefault(key, False)
            return None

        if event_name in _IMMEDIATE_EVENTS:
            if pending_key is not None:
                self._pending_by_key.pop(pending_key, None)
            if self._same_key_gap_active(key, aliases, now_ms=now_ms):
                return None
            return self._emit_event(event, visual_state, key=key, now_ms=now_ms)

        if event_name == "person_left":
            left_key = self._find_known_key(key, aliases)
            if left_key is None:
                return None
            pending_key = self._find_pending_key(key, aliases)
            if pending_key is not None:
                self._pending_by_key.pop(pending_key, None)
                left_key = pending_key
            if self._left_emitted_by_key.get(left_key) is True:
                return None
            frame = self._emit_event(
                event,
                visual_state,
                key=left_key,
                now_ms=now_ms,
                apply_same_key_gap=False,
            )
            if frame is not None:
                for known_key in {left_key, key, *aliases}:
                    self._left_emitted_by_key[known_key] = True
            return frame

        return None

    def _flush_expired_pending(
        self,
        visual_state: dict[str, Any],
        *,
        now_ms: int,
    ) -> list[str]:
        frames: list[str] = []
        expired = [
            key
            for key, pending in self._pending_by_key.items()
            if now_ms - pending.created_ms >= int(self._config.coalesce_window_ms)
        ]
        for key in expired:
            pending = self._pending_by_key.pop(key, None)
            if pending is None:
                continue
            if self._same_key_gap_active(
                pending.key,
                pending.aliases,
                now_ms=now_ms,
            ):
                continue
            frame = self._emit_event(
                pending.event,
                visual_state,
                key=pending.key,
                now_ms=now_ms,
            )
            if frame is not None:
                frames.append(frame)
        return frames

    def _emit_event(
        self,
        event: dict[str, Any],
        visual_state: dict[str, Any],
        *,
        key: tuple[str, Any],
        now_ms: int,
        apply_same_key_gap: bool = True,
    ) -> str | None:
        aliases = self._event_aliases(event, visual_state)
        if apply_same_key_gap and self._same_key_gap_active(key, aliases, now_ms=now_ms):
            return None
        if not self._rate_limit_allows(now_ms):
            return None

        context = _visual_context(event, visual_state, now_ms=now_ms)
        frame = format_botified_frame(event, visual_context=context)
        if frame is None:
            return None

        for known_key in {key, *aliases}:
            self._last_emitted_by_key[known_key] = now_ms
            self._left_emitted_by_key[known_key] = False
        self._emitted_timestamps.append(now_ms)
        return frame

    def _rate_limit_allows(self, now_ms: int) -> bool:
        while self._emitted_timestamps and now_ms - self._emitted_timestamps[0] >= 60_000:
            self._emitted_timestamps.popleft()
        global_count = len(self._emitted_timestamps)
        if global_count >= int(self._config.global_60s_limit):
            return False

        burst_count = sum(
            1 for emitted_ms in self._emitted_timestamps if now_ms - emitted_ms < 1000
        )
        return burst_count < int(self._config.burst_1s_limit)

    def _same_key_gap_active(
        self,
        key: tuple[str, Any],
        aliases: frozenset[tuple[str, Any]] | set[tuple[str, Any]],
        *,
        now_ms: int,
    ) -> bool:
        min_gap_ms = int(self._config.same_key_min_gap_ms)
        keys = {key, *aliases}
        for candidate in keys:
            last_ms = self._last_emitted_by_key.get(candidate)
            if last_ms is not None and now_ms - last_ms < min_gap_ms:
                return True
        return False

    def _find_pending_key(
        self,
        key: tuple[str, Any],
        aliases: frozenset[tuple[str, Any]] | set[tuple[str, Any]],
    ) -> tuple[str, Any] | None:
        keys = {key, *aliases}
        for pending_key, pending in self._pending_by_key.items():
            if pending_key in keys or pending.aliases & keys:
                return pending_key
        return None

    def _find_known_key(
        self,
        key: tuple[str, Any],
        aliases: frozenset[tuple[str, Any]] | set[tuple[str, Any]],
    ) -> tuple[str, Any] | None:
        pending_key = self._find_pending_key(key, aliases)
        if pending_key is not None:
            return pending_key

        keys = {key, *aliases}
        for candidate in keys:
            if candidate in self._last_emitted_by_key:
                return candidate
        return None

    def _event_keys(
        self,
        event: dict[str, Any],
        visual_state: dict[str, Any],
    ) -> tuple[tuple[str, Any] | None, frozenset[tuple[str, Any]]]:
        evidence = event.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}

        slot = evidence.get("runtime_person_slot")
        track_id = event.get("track_id")
        key: tuple[str, Any] | None = None
        if isinstance(slot, (int, str)) and str(slot) != "":
            key = ("slot", slot)
        elif isinstance(track_id, (int, str)) and str(track_id) != "":
            key = ("track", track_id)

        return key, self._event_aliases(event, visual_state)

    def _event_aliases(
        self,
        event: dict[str, Any],
        visual_state: dict[str, Any],
    ) -> frozenset[tuple[str, Any]]:
        aliases: set[tuple[str, Any]] = set()
        track_id = event.get("track_id")
        if isinstance(track_id, (int, str)) and str(track_id) != "":
            aliases.add(("track", track_id))

        evidence = event.get("evidence")
        if isinstance(evidence, dict):
            self._add_reacquire_aliases(aliases, evidence)

        scene_context = visual_state.get("scene_context")
        if isinstance(scene_context, dict):
            target_reacquired = scene_context.get("target_reacquired")
            if isinstance(target_reacquired, dict):
                self._add_reacquire_aliases(aliases, target_reacquired)
        return frozenset(aliases)

    def _add_reacquire_aliases(
        self,
        aliases: set[tuple[str, Any]],
        payload: dict[str, Any],
    ) -> None:
        for field in ("reacquired_from_track_id", "reacquired_to_track_id"):
            value = payload.get(field)
            if isinstance(value, (int, str)) and str(value) != "":
                aliases.add(("track", value))

    def _is_fast_passing(
        self,
        event: dict[str, Any],
        visual_state: dict[str, Any],
    ) -> bool:
        evidence = event.get("evidence")
        if isinstance(evidence, dict) and evidence.get("passing_speed_class") == "fast":
            return True
        scene_context = visual_state.get("scene_context")
        if not isinstance(scene_context, dict):
            return False
        reasons = scene_context.get("no_engage_reasons")
        return isinstance(reasons, list) and "passing_fast" in reasons

    def _remember(self, event_id: str) -> None:
        if self._max_seen_event_ids == 0:
            return
        self._seen_event_ids.add(event_id)
        self._seen_order.append(event_id)
        while len(self._seen_order) > self._max_seen_event_ids:
            expired_event_id = self._seen_order.popleft()
            self._seen_event_ids.discard(expired_event_id)


class BotifiedPipeClosed(Exception):
    """Raised when Botified stdout closes while frames are being written."""


class BotifiedStdoutWriter:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        max_queue_size: int = 32,
    ) -> None:
        self._stream = stream or sys.stdout
        self._max_queue_size = max(0, int(max_queue_size))
        self._queue: deque[str] = deque()
        self._queued_frames: set[str] = set()
        self.dropped_count = 0

    def enqueue(self, frame: str) -> bool:
        if frame in self._queued_frames:
            self.dropped_count += 1
            return False

        if self._max_queue_size == 0:
            self.dropped_count += 1
            return False

        while len(self._queue) >= self._max_queue_size:
            dropped = self._queue.popleft()
            self._queued_frames.discard(dropped)
            self.dropped_count += 1

        self._queue.append(frame)
        self._queued_frames.add(frame)
        return True

    def drain_available(self) -> None:
        try:
            while self._queue:
                frame = self._queue.popleft()
                self._queued_frames.discard(frame)
                self._stream.write(frame + "\n")
                self._stream.flush()
        except BrokenPipeError as exc:
            raise BotifiedPipeClosed("botified stdout closed") from exc


def _format_request(
    event: dict[str, Any],
    *,
    visual_context: dict[str, Any] | None = None,
) -> str:
    request = (
        f"event={event.get('event')} "
        f"camera={event.get('camera')} "
        f"track_id={event.get('track_id')} "
        f"confidence={event.get('confidence')} "
        f"duration_ms={event.get('duration_ms')} "
        f"text={event.get('text')}"
    )
    if visual_context is not None:
        context = json.dumps(
            {"visual_context": visual_context},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        request = f"{request} visual_context={context}"
    return request


def _visual_context(
    event: dict[str, Any],
    visual_state: dict[str, Any],
    *,
    now_ms: int,
) -> dict[str, Any]:
    track = _find_track(visual_state, event.get("track_id"))
    scene_context = visual_state.get("scene_context")
    if not isinstance(scene_context, dict):
        scene_context = {}
    frame_timestamp_ms = _as_int(visual_state.get("frame_timestamp_ms"), now_ms)
    target_track_id = scene_context.get("target_track_id")
    attention_available = scene_context.get("attention_available") is True
    fresh_target = attention_available and target_track_id is not None
    person_count = _person_count(visual_state)

    return {
        "event_target": {
            "track_id": event.get("track_id"),
            "runtime_person_slot": _runtime_person_slot(event),
            "visible_now": _track_visible(track),
            "matches_attention_target": fresh_target
            and event.get("track_id") == target_track_id,
            "event_age_ms": max(0, now_ms - _event_timestamp_ms(event, frame_timestamp_ms)),
            "position": _track_position(track, visual_state),
            "size": _track_size(track),
            "bbox_area_ratio": _track_value(track, "bbox_area_ratio"),
        },
        "trigger_evidence": _project_evidence(event),
        "current_scene": {
            "camera": visual_state.get("camera", event.get("camera")),
            "frame_age_ms": max(0, now_ms - frame_timestamp_ms),
            "person_count": person_count,
            "attention_target": _attention_target(visual_state, target_track_id)
            if fresh_target
            else None,
            "other_people_count": _other_people_count(visual_state, target_track_id)
            if fresh_target
            else person_count,
            "engagement_state": scene_context.get("engagement_state"),
            "no_engage_reasons": _list_or_empty(scene_context.get("no_engage_reasons")),
        },
    }


def _project_evidence(event: dict[str, Any]) -> dict[str, Any]:
    evidence = event.get("evidence")
    if not isinstance(evidence, dict):
        return {}
    allowed_by_event = {
        "person_appeared": (
            "runtime_person_slot",
            "visible_duration_ms",
            "bbox_area_ratio",
            "salient_reason",
        ),
        "person_left": (
            "runtime_person_slot",
            "lost_duration_ms",
            "last_bbox_area_ratio",
        ),
        "person_passing_by": (
            "runtime_person_slot",
            "dx_ratio",
            "avg_vx_px_s",
            "crossed_side_bands",
            "camera_motion_state",
            "passing_speed_class",
        ),
        "person_approaching_robot": (
            "runtime_person_slot",
            "bbox_area_ratio_start",
            "bbox_area_ratio_end",
            "area_growth_ratio",
            "area_delta",
            "camera_motion_state",
        ),
        "person_stopped_near_robot": (
            "runtime_person_slot",
            "bbox_area_ratio",
            "speed_px_s_p95",
            "stationary_duration_ms",
            "camera_motion_state",
        ),
        "person_waving": (
            "runtime_person_slot",
            "wrist_x_span_px",
            "wrist_x_span_bbox_ratio",
            "wrist_y_relative_to_shoulder_px",
            "wave_duration_ms",
            "keypoint_min_confidence",
        ),
    }
    keys = allowed_by_event.get(str(event.get("event")), ())
    projected: dict[str, Any] = {}
    for key in keys:
        if key in evidence:
            projected[key] = evidence[key]
    return projected


def _find_track(visual_state: dict[str, Any], track_id: Any) -> dict[str, Any] | None:
    tracks = visual_state.get("tracks")
    if not isinstance(tracks, list):
        return None
    for track in tracks:
        if isinstance(track, dict) and track.get("track_id") == track_id:
            return track
    return None


def _track_visible(track: dict[str, Any] | None) -> bool:
    if track is None:
        return False
    return _as_int(track.get("lost_ms"), 0) == 0


def _track_position(
    track: dict[str, Any] | None,
    visual_state: dict[str, Any],
) -> str:
    if track is None:
        return "unknown"
    center_uv = track.get("center_uv")
    if not isinstance(center_uv, list) or not center_uv:
        return "unknown"
    image_width = _image_width(visual_state)
    if image_width <= 0:
        return "unknown"
    x = _as_float(center_uv[0])
    if x < image_width / 3:
        return "left"
    if x > image_width * 2 / 3:
        return "right"
    return "center"


def _track_size(track: dict[str, Any] | None) -> str:
    if track is None:
        return "unknown"
    area_ratio = track.get("bbox_area_ratio")
    if area_ratio is None:
        return "unknown"
    value = _as_float(area_ratio)
    if value <= 0:
        return "unknown"
    if value < 0.04:
        return "far"
    if value < 0.12:
        return "mid"
    return "near"


def _track_value(track: dict[str, Any] | None, field: str) -> Any:
    if track is None:
        return None
    return track.get(field)


def _runtime_person_slot(event: dict[str, Any]) -> Any:
    evidence = event.get("evidence")
    if not isinstance(evidence, dict):
        return None
    return evidence.get("runtime_person_slot")


def _event_timestamp_ms(event: dict[str, Any], default_ms: int) -> int:
    return _as_int(event.get("timestamp_ms"), default_ms)


def _person_count(visual_state: dict[str, Any]) -> int:
    scene_flags = visual_state.get("scene_flags")
    if isinstance(scene_flags, dict):
        person_count = scene_flags.get("person_count")
        if isinstance(person_count, int) and not isinstance(person_count, bool):
            return max(0, person_count)

    tracks = visual_state.get("tracks")
    if not isinstance(tracks, list):
        return 0
    return sum(
        1
        for track in tracks
        if isinstance(track, dict)
        and track.get("class") == "person"
        and _track_visible(track)
    )


def _other_people_count(visual_state: dict[str, Any], target_track_id: Any) -> int:
    scene_flags = visual_state.get("scene_flags")
    if isinstance(scene_flags, dict):
        person_count = scene_flags.get("person_count")
        if isinstance(person_count, int) and not isinstance(person_count, bool):
            return max(0, person_count - 1)

    tracks = visual_state.get("tracks")
    if not isinstance(tracks, list):
        return 0
    return sum(
        1
        for track in tracks
        if isinstance(track, dict)
        and track.get("class") == "person"
        and _track_visible(track)
        and track.get("track_id") != target_track_id
    )


def _attention_target(
    visual_state: dict[str, Any],
    target_track_id: Any,
) -> dict[str, Any]:
    track = _find_track(visual_state, target_track_id)
    return {
        "track_id": target_track_id,
        "position": _track_position(track, visual_state),
        "size": _track_size(track),
        "center_uv": _track_value(track, "center_uv"),
        "bbox_area_ratio": _track_value(track, "bbox_area_ratio"),
    }


def _image_width(visual_state: dict[str, Any]) -> int:
    image_size = visual_state.get("image_size")
    if not isinstance(image_size, list) or not image_size:
        return 0
    return _as_int(image_size[0], 0)


def _list_or_empty(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _wall_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
