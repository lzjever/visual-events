from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from visual_events_server.protocol import (
    MAX_JPEG_BYTES,
    SCHEMA_VERSION,
    encode_frame_message,
)

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
_JPEG_GLOBS = ("*.jpeg", "*.jpg")
_FILENAME_NUMBER = re.compile(r"(\d+)")
_ASSOCIATION_IOU_THRESHOLD = 0.5
_S4_STABLE_ATTENTION_SCENES = {"pci_stand", "pic_walk_in_stop"}
_S4_STABLE_ATTENTION_MIN_COVERAGE = 0.85
_S4_STABLE_ATTENTION_MAX_SWITCHES = 2
_S4_STABLE_ATTENTION_SWITCH_DWELL_MS = 750
DEFAULT_SEMANTIC_EVENT_COOLDOWN_MS = 5000
_SEMANTIC_EVENT_ID = re.compile(r"^[^:]+:evt_\d{6}$")
_SEMANTIC_EVENT_TYPES = {
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
    "attention_target_changed",
}
_MOTION_SENSITIVE_EVENT_TYPES = {
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
}
SEMANTIC_EVENT_FIRST_FRAME_TOLERANCE = 3
_SCENE_EXPECTED_EVENTS = {
    "pci_stand": {"person_appeared", "person_stopped_near_robot"},
    "pic_1_l_to_r": {"person_passing_by"},
    "pic_1_r_to_l": {"person_passing_by"},
    "pic_hello": {"person_waving"},
    "pic_leave": {"person_left"},
    "pic_persone_walk_in": {"person_approaching_robot"},
    "pic_walk_in_stop": {
        "person_approaching_robot",
        "person_stopped_near_robot",
    },
}
_SCENE_EXPECTED_EVENT_FIRST_FRAMES = {
    "pci_stand": {"person_stopped_near_robot": 44},
    "pic_1_l_to_r": {"person_passing_by": 40},
    "pic_1_r_to_l": {"person_passing_by": 47},
    "pic_hello": {"person_waving": 12},
    "pic_leave": {"person_left": 75},
    "pic_persone_walk_in": {"person_approaching_robot": 39},
    "pic_walk_in_stop": {
        "person_approaching_robot": 9,
        "person_stopped_near_robot": 63,
    },
}
_SCENE_UNEXPECTED_EVENTS = {
    "pci_stand": {"person_waving"},
    "pic_1_l_to_r": {"person_stopped_near_robot", "person_waving"},
    "pic_1_r_to_l": {"person_stopped_near_robot", "person_waving"},
    "pic_leave": {"person_waving"},
    "pic_persone_walk_in": {"person_passing_by", "person_waving"},
    "pic_walk_in_stop": {"person_waving"},
}
_SCENE_EVENT_ORDER_REQUIREMENTS = {
    "pic_walk_in_stop": (
        ("person_approaching_robot", "person_stopped_near_robot"),
    ),
}
# Botified notification expectations intentionally differ from raw server
# semantic-event expectations: low-value/default-suppressed facts should not
# require an agent wake-up.
_BOTIFIED_EVENT_ORACLE_REQUIRED_EVENTS = {
    "pci_stand": {"person_stopped_near_robot"},
    "pic_hello": {"person_waving"},
    "pic_persone_walk_in": {"person_approaching_robot"},
    "pic_walk_in_stop": {
        "person_approaching_robot",
        "person_stopped_near_robot",
    },
}
_SCENE_DUPLICATE_GREETING_CONTRACTS = {
    "pic_hello": (
        {
            "person_label": "primary_person",
            "event": "person_waving",
            "max_count": 1,
        },
    ),
}
_BOTIFIED_EVENT_ORACLE_IGNORED_EVENTS = {"attention_target_changed"}


def botified_event_oracle_facts(scene: str, head_motion: str) -> dict[str, Any]:
    required_events = set(_BOTIFIED_EVENT_ORACLE_REQUIRED_EVENTS.get(scene, set()))
    required_events -= _BOTIFIED_EVENT_ORACLE_IGNORED_EVENTS
    if head_motion != "stationary":
        required_events -= _MOTION_SENSITIVE_EVENT_TYPES

    forbidden_events = set(_SCENE_UNEXPECTED_EVENTS.get(scene, set()))
    forbidden_events -= _BOTIFIED_EVENT_ORACLE_IGNORED_EVENTS

    order_requirements: list[tuple[str, str]] = []
    for before_event, after_event in _SCENE_EVENT_ORDER_REQUIREMENTS.get(scene, ()):
        if (
            before_event in _BOTIFIED_EVENT_ORACLE_IGNORED_EVENTS
            or after_event in _BOTIFIED_EVENT_ORACLE_IGNORED_EVENTS
        ):
            continue
        if head_motion != "stationary" and (
            before_event in _MOTION_SENSITIVE_EVENT_TYPES
            or after_event in _MOTION_SENSITIVE_EVENT_TYPES
        ):
            continue
        order_requirements.append((before_event, after_event))

    return {
        "required_events": sorted(required_events),
        "forbidden_events": sorted(forbidden_events),
        "order_requirements": sorted(order_requirements),
        "duplicate_greeting_contracts": _active_duplicate_greeting_contracts(
            scene,
            head_motion,
        ),
    }


@dataclass(frozen=True)
class ReplayFrame:
    path: Path
    header: dict[str, Any]


@dataclass(frozen=True)
class ReplayStats:
    scene: str
    frames_sent: int
    frames_ok: int
    errors: int
    elapsed_s: float
    head_motion: str = "stationary"
    frames_with_person: int = 0
    frame_id_mismatch: int = 0
    track_frames: int = 0
    largest_bbox_track_switches: int = 0
    largest_bbox_track_id: int | None = None
    largest_bbox_track_coverage: float = 0.0
    largest_bbox_track_max_gap_ms: int = 0
    duplicate_track_id_frames: int = 0
    single_visible_id_switches: int = 0
    adjacent_track_matches: int = 0
    association_id_switches: int = 0
    visible_counts_by_id: dict[str, int] = field(default_factory=dict)
    track_schema_errors: int = 0
    age_monotonic_violations: int = 0
    attention_frames: int = 0
    attention_null_frames: int = 0
    attention_target_switches: int = 0
    attention_target_counts_by_id: dict[str, int] = field(default_factory=dict)
    attention_schema_errors: int = 0
    attention_invalid_uv_frames: int = 0
    attention_target_missing_track_frames: int = 0
    attention_target_lost_frames: int = 0
    attention_max_lost_hold_ms: int = 0
    attention_largest_bbox_disagreement_frames: int = 0
    attention_actionable_largest_bbox_disagreement_frames: int = 0
    semantic_event_frames: int = 0
    semantic_event_count: int = 0
    semantic_event_counts_by_type: dict[str, int] = field(default_factory=dict)
    semantic_event_first_frame_by_type: dict[str, int] = field(default_factory=dict)
    semantic_event_schema_errors: int = 0
    semantic_event_unknown_type_count: int = 0
    semantic_event_id_format_errors: int = 0
    semantic_event_duplicate_id_count: int = 0
    semantic_event_duplicate_track_event_count: int = 0
    semantic_event_cooldown_ms: int = DEFAULT_SEMANTIC_EVENT_COOLDOWN_MS
    semantic_event_type_cooldown_errors: int = 0
    semantic_event_confidence_errors: int = 0
    semantic_event_duration_errors: int = 0
    semantic_event_empty_text_count: int = 0
    semantic_event_track_missing_frames: int = 0
    semantic_event_motion_sensitive_count: int = 0
    semantic_event_expected_missing: int = 0
    semantic_event_unexpected_by_scene: int = 0
    semantic_event_first_frame_tolerance: int = SEMANTIC_EVENT_FIRST_FRAME_TOLERANCE
    semantic_event_expected_first_frame_by_type: dict[str, int] = field(
        default_factory=dict
    )
    semantic_event_first_frame_diagnostics: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    semantic_event_trigger_timing_errors: int = 0
    semantic_event_forbidden_events_by_type: dict[str, int] = field(
        default_factory=dict
    )
    semantic_event_order_violations: int = 0
    semantic_event_order_diagnostics: list[dict[str, Any]] = field(
        default_factory=list
    )
    semantic_event_duplicate_greeting_violation_count: int = 0
    semantic_event_duplicate_greeting_violations: list[dict[str, Any]] = field(
        default_factory=list
    )
    semantic_event_timeline_violations: list[dict[str, Any]] = field(
        default_factory=list
    )

    @property
    def ok_rate(self) -> float:
        if self.frames_sent == 0:
            return 0.0
        return self.frames_ok / self.frames_sent

    @property
    def person_frame_rate(self) -> float:
        if self.frames_sent == 0:
            return 0.0
        return self.frames_with_person / self.frames_sent

    @property
    def attention_coverage(self) -> float:
        if self.frames_ok == 0:
            return 0.0
        return self.attention_frames / self.frames_ok


def discover_scene_dirs(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    if _has_jpegs(data_dir):
        return [data_dir]

    scene_dirs = [
        child
        for child in sorted(data_dir.iterdir())
        if child.is_dir() and _has_jpegs(child)
    ]
    if not scene_dirs:
        raise FileNotFoundError(f"no JPEG frames found under {data_dir}")
    return scene_dirs


def iter_scene_frames(
    scene_dir: Path,
    *,
    camera: str,
    fps: float,
    head_motion: str,
) -> list[ReplayFrame]:
    if fps <= 0:
        raise ValueError("fps must be positive")

    paths = _sorted_jpeg_paths(Path(scene_dir))
    if not paths:
        raise FileNotFoundError(f"no JPEG frames found in {scene_dir}")

    frames: list[ReplayFrame] = []
    fallback_step_ms = 1000.0 / fps
    for frame_id, path in enumerate(paths):
        timestamp_ms = _timestamp_ms_from_filename(path)
        if timestamp_ms is None:
            timestamp_ms = int(round(frame_id * fallback_step_ms))
        frames.append(
            ReplayFrame(
                path=path,
                header={
                    "type": "frame",
                    "schema_version": SCHEMA_VERSION,
                    "camera": camera,
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "encoding": "jpeg",
                    "width": DEFAULT_WIDTH,
                    "height": DEFAULT_HEIGHT,
                    "head_motion": {"state": head_motion},
                },
            )
        )
    return frames


async def replay_scene(
    *,
    server: str,
    scene_dir: Path,
    camera: str,
    fps: float,
    head_motion: str,
    save_jsonl: Path | None = None,
    append_jsonl: bool = False,
    connector: Callable[..., Any] | None = None,
    realtime: bool = True,
    response_timeout_ms: int | None = None,
    continue_on_timeout: bool = False,
    semantic_event_cooldown_ms: int = DEFAULT_SEMANTIC_EVENT_COOLDOWN_MS,
) -> ReplayStats:
    if semantic_event_cooldown_ms < 0:
        raise ValueError("semantic_event_cooldown_ms must be non-negative")

    frames = iter_scene_frames(
        scene_dir,
        camera=camera,
        fps=fps,
        head_motion=head_motion,
    )
    connect = connector or _default_connector()
    interval_s = 1.0 / fps
    start_s = time.perf_counter()
    frames_sent = 0
    frames_ok = 0
    errors = 0
    frames_with_person = 0
    frame_id_mismatch = 0
    tracking_stats = _TrackingStatsAccumulator()
    attention_stats = _AttentionStatsAccumulator()
    semantic_event_stats = _SemanticEventStatsAccumulator(
        scene=Path(scene_dir).name,
        head_motion=head_motion,
        cooldown_ms=semantic_event_cooldown_ms,
    )

    jsonl_file = None
    try:
        if save_jsonl is not None:
            save_jsonl.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append_jsonl else "w"
            jsonl_file = save_jsonl.open(mode, encoding="utf-8")

        websocket_cm = None
        websocket = None
        try:
            for frame in frames:
                if websocket is None:
                    next_websocket_cm = connect(server, max_size=None)
                    websocket = await next_websocket_cm.__aenter__()
                    websocket_cm = next_websocket_cm

                frame_started_s = time.perf_counter()
                jpeg_bytes = frame.path.read_bytes()
                payload = encode_frame_message(frame.header, jpeg_bytes)
                await websocket.send(payload)
                frames_sent += 1

                try:
                    raw_response = await _recv_with_timeout(
                        websocket,
                        response_timeout_ms=response_timeout_ms,
                    )
                except TimeoutError:
                    errors += 1
                    if jsonl_file is not None:
                        jsonl_file.write(
                            json.dumps(
                                {
                                    "scene": Path(scene_dir).name,
                                    "frame_id": frame.header["frame_id"],
                                    "latency_ms": (
                                        time.perf_counter() - frame_started_s
                                    )
                                    * 1000.0,
                                    "response": {
                                        "type": "error",
                                        "code": "response_timeout",
                                    },
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    if websocket_cm is not None:
                        await websocket_cm.__aexit__(None, None, None)
                    websocket_cm = None
                    websocket = None
                    if continue_on_timeout:
                        continue
                    break
                latency_ms = (time.perf_counter() - frame_started_s) * 1000.0
                response = _decode_response(raw_response)
                if response.get("type") == "visual_state":
                    frames_ok += 1
                    scene_flags = response.get("scene_flags", {})
                    if isinstance(scene_flags, dict) and scene_flags.get("has_person"):
                        frames_with_person += 1
                    if response.get("frame_id") != frame.header["frame_id"]:
                        frame_id_mismatch += 1
                    tracking_stats.observe(response)
                    attention_stats.observe(response)
                    semantic_event_stats.observe(response)
                else:
                    errors += 1

                if jsonl_file is not None:
                    jsonl_file.write(
                        json.dumps(
                            {
                                "scene": Path(scene_dir).name,
                                "frame_id": frame.header["frame_id"],
                                "latency_ms": latency_ms,
                                "response": response,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

                if realtime:
                    elapsed = time.perf_counter() - frame_started_s
                    if elapsed < interval_s:
                        await asyncio.sleep(interval_s - elapsed)
        finally:
            if websocket_cm is not None:
                await websocket_cm.__aexit__(None, None, None)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    tracking_summary = tracking_stats.summary()
    attention_summary = attention_stats.summary()
    semantic_event_summary = semantic_event_stats.summary()
    return ReplayStats(
        scene=Path(scene_dir).name,
        frames_sent=frames_sent,
        frames_ok=frames_ok,
        errors=errors,
        elapsed_s=time.perf_counter() - start_s,
        head_motion=head_motion,
        frames_with_person=frames_with_person,
        frame_id_mismatch=frame_id_mismatch,
        track_frames=tracking_summary["track_frames"],
        largest_bbox_track_switches=tracking_summary["largest_bbox_track_switches"],
        largest_bbox_track_id=tracking_summary["largest_bbox_track_id"],
        largest_bbox_track_coverage=tracking_summary["largest_bbox_track_coverage"],
        largest_bbox_track_max_gap_ms=tracking_summary[
            "largest_bbox_track_max_gap_ms"
        ],
        duplicate_track_id_frames=tracking_summary["duplicate_track_id_frames"],
        single_visible_id_switches=tracking_summary["single_visible_id_switches"],
        adjacent_track_matches=tracking_summary["adjacent_track_matches"],
        association_id_switches=tracking_summary["association_id_switches"],
        visible_counts_by_id=tracking_summary["visible_counts_by_id"],
        track_schema_errors=tracking_summary["track_schema_errors"],
        age_monotonic_violations=tracking_summary["age_monotonic_violations"],
        attention_frames=attention_summary["attention_frames"],
        attention_null_frames=attention_summary["attention_null_frames"],
        attention_target_switches=attention_summary["attention_target_switches"],
        attention_target_counts_by_id=attention_summary[
            "attention_target_counts_by_id"
        ],
        attention_schema_errors=attention_summary["attention_schema_errors"],
        attention_invalid_uv_frames=attention_summary["attention_invalid_uv_frames"],
        attention_target_missing_track_frames=attention_summary[
            "attention_target_missing_track_frames"
        ],
        attention_target_lost_frames=attention_summary["attention_target_lost_frames"],
        attention_max_lost_hold_ms=attention_summary["attention_max_lost_hold_ms"],
        attention_largest_bbox_disagreement_frames=attention_summary[
            "attention_largest_bbox_disagreement_frames"
        ],
        attention_actionable_largest_bbox_disagreement_frames=attention_summary[
            "attention_actionable_largest_bbox_disagreement_frames"
        ],
        semantic_event_frames=semantic_event_summary["semantic_event_frames"],
        semantic_event_count=semantic_event_summary["semantic_event_count"],
        semantic_event_counts_by_type=semantic_event_summary[
            "semantic_event_counts_by_type"
        ],
        semantic_event_first_frame_by_type=semantic_event_summary[
            "semantic_event_first_frame_by_type"
        ],
        semantic_event_schema_errors=semantic_event_summary[
            "semantic_event_schema_errors"
        ],
        semantic_event_unknown_type_count=semantic_event_summary[
            "semantic_event_unknown_type_count"
        ],
        semantic_event_id_format_errors=semantic_event_summary[
            "semantic_event_id_format_errors"
        ],
        semantic_event_duplicate_id_count=semantic_event_summary[
            "semantic_event_duplicate_id_count"
        ],
        semantic_event_duplicate_track_event_count=semantic_event_summary[
            "semantic_event_duplicate_track_event_count"
        ],
        semantic_event_cooldown_ms=semantic_event_summary[
            "semantic_event_cooldown_ms"
        ],
        semantic_event_type_cooldown_errors=semantic_event_summary[
            "semantic_event_type_cooldown_errors"
        ],
        semantic_event_confidence_errors=semantic_event_summary[
            "semantic_event_confidence_errors"
        ],
        semantic_event_duration_errors=semantic_event_summary[
            "semantic_event_duration_errors"
        ],
        semantic_event_empty_text_count=semantic_event_summary[
            "semantic_event_empty_text_count"
        ],
        semantic_event_track_missing_frames=semantic_event_summary[
            "semantic_event_track_missing_frames"
        ],
        semantic_event_motion_sensitive_count=semantic_event_summary[
            "semantic_event_motion_sensitive_count"
        ],
        semantic_event_expected_missing=semantic_event_summary[
            "semantic_event_expected_missing"
        ],
        semantic_event_unexpected_by_scene=semantic_event_summary[
            "semantic_event_unexpected_by_scene"
        ],
        semantic_event_first_frame_tolerance=semantic_event_summary[
            "semantic_event_first_frame_tolerance"
        ],
        semantic_event_expected_first_frame_by_type=semantic_event_summary[
            "semantic_event_expected_first_frame_by_type"
        ],
        semantic_event_first_frame_diagnostics=semantic_event_summary[
            "semantic_event_first_frame_diagnostics"
        ],
        semantic_event_trigger_timing_errors=semantic_event_summary[
            "semantic_event_trigger_timing_errors"
        ],
        semantic_event_forbidden_events_by_type=semantic_event_summary[
            "semantic_event_forbidden_events_by_type"
        ],
        semantic_event_order_violations=semantic_event_summary[
            "semantic_event_order_violations"
        ],
        semantic_event_order_diagnostics=semantic_event_summary[
            "semantic_event_order_diagnostics"
        ],
        semantic_event_duplicate_greeting_violation_count=semantic_event_summary[
            "semantic_event_duplicate_greeting_violation_count"
        ],
        semantic_event_duplicate_greeting_violations=semantic_event_summary[
            "semantic_event_duplicate_greeting_violations"
        ],
        semantic_event_timeline_violations=semantic_event_summary[
            "semantic_event_timeline_violations"
        ],
    )


async def replay_data_dir(
    *,
    server: str,
    data_dir: Path,
    camera: str,
    fps: float,
    head_motion: str,
    save_jsonl: Path | None,
    realtime: bool = True,
    response_timeout_ms: int | None = None,
    continue_on_timeout: bool = False,
    semantic_event_cooldown_ms: int = DEFAULT_SEMANTIC_EVENT_COOLDOWN_MS,
) -> list[ReplayStats]:
    if semantic_event_cooldown_ms < 0:
        raise ValueError("semantic_event_cooldown_ms must be non-negative")

    scene_dirs = discover_scene_dirs(data_dir)
    append_jsonl = False
    if save_jsonl is not None and len(scene_dirs) > 1:
        save_jsonl.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl.write_text("", encoding="utf-8")
        append_jsonl = True

    stats: list[ReplayStats] = []
    for scene_dir in scene_dirs:
        stats.append(
            await replay_scene(
                server=server,
                scene_dir=scene_dir,
                camera=camera,
                fps=fps,
                head_motion=head_motion,
                save_jsonl=save_jsonl,
                append_jsonl=append_jsonl,
                realtime=realtime,
                response_timeout_ms=response_timeout_ms,
                continue_on_timeout=continue_on_timeout,
                semantic_event_cooldown_ms=semantic_event_cooldown_ms,
            )
        )
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay val-data JPEG scenes into visual-events-server."
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--head-motion",
        choices=("stationary", "moving", "unknown"),
        default="stationary",
    )
    parser.add_argument("--save-jsonl", type=Path)
    parser.add_argument(
        "--response-timeout-ms",
        type=int,
        default=None,
        help="Stop waiting for a frame response after this many milliseconds.",
    )
    parser.add_argument(
        "--continue-on-timeout",
        action="store_true",
        help="Reconnect and continue replaying later frames after a response timeout.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Write per-scene replay summary JSON.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send the next frame as soon as a response arrives.",
    )
    parser.add_argument(
        "--gate",
        choices=("tracking", "attention", "events", "all", "none"),
        default="tracking",
        help="Validation gate to apply to the replay summary.",
    )
    parser.add_argument(
        "--semantic-event-cooldown-ms",
        type=int,
        default=DEFAULT_SEMANTIC_EVENT_COOLDOWN_MS,
        help="Cooldown window used by replay event duplicate checks.",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = await replay_data_dir(
        server=args.server,
        data_dir=args.data_dir,
        camera=args.camera,
        fps=args.fps,
        head_motion=args.head_motion,
        save_jsonl=args.save_jsonl,
        realtime=not args.no_realtime,
        response_timeout_ms=args.response_timeout_ms,
        continue_on_timeout=args.continue_on_timeout,
        semantic_event_cooldown_ms=args.semantic_event_cooldown_ms,
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                [_stats_to_summary(item, gate=args.gate) for item in stats],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    for item in stats:
        print(
            json.dumps(
                stats_to_summary(item, gate=args.gate),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return 0 if all(stats_passed(item, gate=args.gate) for item in stats) else 1


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


def _has_jpegs(path: Path) -> bool:
    return path.is_dir() and any(match for glob in _JPEG_GLOBS for match in path.glob(glob))


def _sorted_jpeg_paths(path: Path) -> list[Path]:
    paths = [match for glob in _JPEG_GLOBS for match in path.glob(glob)]
    return sorted(paths, key=lambda item: (_filename_sort_key(item), item.name))


def _filename_sort_key(path: Path) -> int:
    timestamp_ms = _timestamp_ms_from_filename(path)
    return timestamp_ms if timestamp_ms is not None else 0


def _timestamp_ms_from_filename(path: Path) -> int | None:
    match = _FILENAME_NUMBER.search(path.stem)
    if match is None:
        return None
    return int(match.group(1)) // 1_000_000


def _decode_response(raw_response: str | bytes) -> dict[str, Any]:
    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8")
    response = json.loads(raw_response)
    if not isinstance(response, dict):
        raise ValueError("server response must be a JSON object")
    return response


async def _recv_with_timeout(
    websocket: Any,
    *,
    response_timeout_ms: int | None,
) -> str | bytes:
    if response_timeout_ms is None:
        return await websocket.recv()
    if response_timeout_ms <= 0:
        raise ValueError("response_timeout_ms must be positive")
    try:
        return await asyncio.wait_for(
            websocket.recv(),
            timeout=response_timeout_ms / 1000.0,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError("timed out waiting for server response") from exc


def stats_to_summary(item: ReplayStats, *, gate: str = "tracking") -> dict[str, Any]:
    tracking_pass = _tracking_stats_passed(item)
    attention_pass = _attention_stats_passed(item)
    events_pass = _events_stats_passed(item)
    return {
        "scene": item.scene,
        "frames_sent": item.frames_sent,
        "frames_ok": item.frames_ok,
        "errors": item.errors,
        "ok_rate": item.ok_rate,
        "frames_with_person": item.frames_with_person,
        "person_frame_rate": item.person_frame_rate,
        "frame_id_mismatch": item.frame_id_mismatch,
        "track_frames": item.track_frames,
        "largest_bbox_track_switches": item.largest_bbox_track_switches,
        "largest_bbox_track_id": item.largest_bbox_track_id,
        "largest_bbox_track_coverage": item.largest_bbox_track_coverage,
        "largest_bbox_track_max_gap_ms": item.largest_bbox_track_max_gap_ms,
        "duplicate_track_id_frames": item.duplicate_track_id_frames,
        "single_visible_id_switches": item.single_visible_id_switches,
        "adjacent_track_matches": item.adjacent_track_matches,
        "association_id_switches": item.association_id_switches,
        "visible_counts_by_id": item.visible_counts_by_id,
        "track_schema_errors": item.track_schema_errors,
        "age_monotonic_violations": item.age_monotonic_violations,
        "attention_frames": item.attention_frames,
        "attention_null_frames": item.attention_null_frames,
        "attention_coverage": item.attention_coverage,
        "attention_target_switches": item.attention_target_switches,
        "attention_target_counts_by_id": item.attention_target_counts_by_id,
        "attention_schema_errors": item.attention_schema_errors,
        "attention_invalid_uv_frames": item.attention_invalid_uv_frames,
        "attention_target_missing_track_frames": (
            item.attention_target_missing_track_frames
        ),
        "attention_target_lost_frames": item.attention_target_lost_frames,
        "attention_max_lost_hold_ms": item.attention_max_lost_hold_ms,
        "attention_largest_bbox_disagreement_frames": (
            item.attention_largest_bbox_disagreement_frames
        ),
        "attention_actionable_largest_bbox_disagreement_frames": (
            item.attention_actionable_largest_bbox_disagreement_frames
        ),
        "semantic_event_frames": item.semantic_event_frames,
        "semantic_event_count": item.semantic_event_count,
        "semantic_event_counts_by_type": item.semantic_event_counts_by_type,
        "semantic_event_first_frame_by_type": item.semantic_event_first_frame_by_type,
        "semantic_event_schema_errors": item.semantic_event_schema_errors,
        "semantic_event_unknown_type_count": item.semantic_event_unknown_type_count,
        "semantic_event_id_format_errors": item.semantic_event_id_format_errors,
        "semantic_event_duplicate_id_count": item.semantic_event_duplicate_id_count,
        "semantic_event_duplicate_track_event_count": (
            item.semantic_event_duplicate_track_event_count
        ),
        "semantic_event_cooldown_ms": item.semantic_event_cooldown_ms,
        "semantic_event_type_cooldown_errors": (
            item.semantic_event_type_cooldown_errors
        ),
        "semantic_event_confidence_errors": item.semantic_event_confidence_errors,
        "semantic_event_duration_errors": item.semantic_event_duration_errors,
        "semantic_event_empty_text_count": item.semantic_event_empty_text_count,
        "semantic_event_track_missing_frames": (
            item.semantic_event_track_missing_frames
        ),
        "semantic_event_motion_sensitive_count": (
            item.semantic_event_motion_sensitive_count
        ),
        "semantic_event_expected_missing": item.semantic_event_expected_missing,
        "semantic_event_unexpected_by_scene": item.semantic_event_unexpected_by_scene,
        "semantic_event_first_frame_tolerance": (
            item.semantic_event_first_frame_tolerance
        ),
        "semantic_event_expected_first_frame_by_type": (
            item.semantic_event_expected_first_frame_by_type
        ),
        "semantic_event_first_frame_diagnostics": (
            item.semantic_event_first_frame_diagnostics
        ),
        "semantic_event_trigger_timing_errors": (
            item.semantic_event_trigger_timing_errors
        ),
        "semantic_event_forbidden_events_by_type": (
            item.semantic_event_forbidden_events_by_type
        ),
        "semantic_event_order_violations": item.semantic_event_order_violations,
        "semantic_event_order_diagnostics": item.semantic_event_order_diagnostics,
        "semantic_event_duplicate_greeting_violation_count": (
            item.semantic_event_duplicate_greeting_violation_count
        ),
        "semantic_event_duplicate_greeting_violations": (
            item.semantic_event_duplicate_greeting_violations
        ),
        "semantic_event_timeline_violations": (
            item.semantic_event_timeline_violations
        ),
        "tracking_pass": tracking_pass,
        "attention_pass": attention_pass,
        "events_pass": events_pass,
        "passed": stats_passed(item, gate=gate),
        "elapsed_s": item.elapsed_s,
    }


def stats_passed(item: ReplayStats, *, gate: str = "tracking") -> bool:
    if gate == "tracking":
        return _tracking_stats_passed(item)
    if gate == "attention":
        return _attention_stats_passed(item)
    if gate == "events":
        return _events_stats_passed(item)
    if gate == "all":
        return (
            _tracking_stats_passed(item)
            and _attention_stats_passed(item)
            and _events_stats_passed(item)
        )
    if gate == "none":
        return True
    raise ValueError("gate must be one of: tracking, attention, events, all, none")


def _stats_to_summary(item: ReplayStats, *, gate: str = "tracking") -> dict[str, Any]:
    return stats_to_summary(item, gate=gate)


def _stats_passed(item: ReplayStats, *, gate: str = "tracking") -> bool:
    return stats_passed(item, gate=gate)


def _tracking_stats_passed(item: ReplayStats) -> bool:
    return (
        item.errors == 0
        and item.frame_id_mismatch == 0
        and item.track_frames > 0
        and bool(item.visible_counts_by_id)
        and item.track_schema_errors == 0
        and item.age_monotonic_violations == 0
        and item.duplicate_track_id_frames == 0
        and item.single_visible_id_switches == 0
        and item.association_id_switches == 0
    )


def _attention_stats_passed(item: ReplayStats) -> bool:
    generic_pass = (
        item.errors == 0
        and item.frame_id_mismatch == 0
        and item.frames_ok > 0
        and item.attention_frames > 0
        and item.attention_schema_errors == 0
        and item.attention_invalid_uv_frames == 0
        and item.attention_target_missing_track_frames == 0
    )
    if not generic_pass:
        return False
    if item.scene not in _S4_STABLE_ATTENTION_SCENES:
        return True
    return (
        item.attention_coverage >= _S4_STABLE_ATTENTION_MIN_COVERAGE
        and item.attention_target_switches <= _S4_STABLE_ATTENTION_MAX_SWITCHES
        and item.attention_actionable_largest_bbox_disagreement_frames == 0
    )


def _events_stats_passed(item: ReplayStats) -> bool:
    base_pass = (
        item.errors == 0
        and item.frame_id_mismatch == 0
        and item.frames_ok > 0
        and item.semantic_event_schema_errors == 0
        and item.semantic_event_unknown_type_count == 0
        and item.semantic_event_id_format_errors == 0
        and item.semantic_event_duplicate_id_count == 0
        and item.semantic_event_duplicate_track_event_count == 0
        and item.semantic_event_type_cooldown_errors == 0
        and item.semantic_event_confidence_errors == 0
        and item.semantic_event_duration_errors == 0
        and item.semantic_event_empty_text_count == 0
        and item.semantic_event_track_missing_frames == 0
        and item.semantic_event_expected_missing == 0
        and item.semantic_event_unexpected_by_scene == 0
        and item.semantic_event_trigger_timing_errors == 0
        and item.semantic_event_order_violations == 0
        and item.semantic_event_duplicate_greeting_violation_count == 0
    )
    if not base_pass:
        return False
    if item.head_motion in {"moving", "unknown"}:
        return item.semantic_event_motion_sensitive_count == 0
    return True


def _active_expected_first_frames(scene: str, head_motion: str) -> dict[str, int]:
    expected = dict(_SCENE_EXPECTED_EVENT_FIRST_FRAMES.get(scene, {}))
    if head_motion in {"moving", "unknown"}:
        expected = {
            event: frame
            for event, frame in expected.items()
            if event not in _MOTION_SENSITIVE_EVENT_TYPES
        }
    return dict(sorted(expected.items()))


def _active_duplicate_greeting_contracts(
    scene: str,
    head_motion: str,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for contract in _SCENE_DUPLICATE_GREETING_CONTRACTS.get(scene, ()):
        event = str(contract["event"])
        if head_motion != "stationary" and event in _MOTION_SENSITIVE_EVENT_TYPES:
            continue
        contracts.append(
            {
                "person_label": str(contract["person_label"]),
                "event": event,
                "max_count": int(contract["max_count"]),
            }
        )
    return sorted(
        contracts,
        key=lambda item: (item["person_label"], item["event"]),
    )


def _semantic_event_contract_summary(
    *,
    scene: str,
    head_motion: str,
    counts_by_type: dict[str, int],
    first_frame_by_type: dict[str, int],
    event_records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_events = set(_SCENE_EXPECTED_EVENTS.get(scene, set()))
    if head_motion in {"moving", "unknown"}:
        expected_events -= _MOTION_SENSITIVE_EVENT_TYPES
    forbidden_events = set(_SCENE_UNEXPECTED_EVENTS.get(scene, set()))
    expected_first_frames = _active_expected_first_frames(scene, head_motion)

    first_frame_diagnostics: dict[str, dict[str, Any]] = {}
    timeline_violations: list[dict[str, Any]] = []
    trigger_timing_errors = 0
    for event, expected_frame in expected_first_frames.items():
        actual_frame = first_frame_by_type.get(event)
        delta_frames = (
            None if actual_frame is None else int(actual_frame) - int(expected_frame)
        )
        within_tolerance = (
            delta_frames is not None
            and abs(delta_frames) <= SEMANTIC_EVENT_FIRST_FRAME_TOLERANCE
        )
        first_frame_diagnostics[event] = {
            "expected_frame": expected_frame,
            "actual_frame": actual_frame,
            "delta_frames": delta_frames,
            "within_tolerance": within_tolerance,
        }
        if within_tolerance:
            continue

        trigger_timing_errors += 1
        timeline_violations.append(
            {
                "code": (
                    "expected_trigger_missing"
                    if actual_frame is None
                    else "trigger_frame_outside_tolerance"
                ),
                "event": event,
                "expected_frame": expected_frame,
                "actual_frame": actual_frame,
                "delta_frames": delta_frames,
                "tolerance_frames": SEMANTIC_EVENT_FIRST_FRAME_TOLERANCE,
            }
        )

    forbidden_events_by_type = {
        event: counts_by_type[event]
        for event in sorted(forbidden_events)
        if counts_by_type.get(event, 0) > 0
    }

    order_diagnostics: list[dict[str, Any]] = []
    order_violations = 0
    for before_event, after_event in _SCENE_EVENT_ORDER_REQUIREMENTS.get(scene, ()):
        if head_motion in {"moving", "unknown"} and (
            before_event in _MOTION_SENSITIVE_EVENT_TYPES
            or after_event in _MOTION_SENSITIVE_EVENT_TYPES
        ):
            continue
        before_frame = first_frame_by_type.get(before_event)
        after_frame = first_frame_by_type.get(after_event)
        if before_frame is None or after_frame is None:
            continue
        passed = before_frame < after_frame
        order_diagnostics.append(
            {
                "before_event": before_event,
                "after_event": after_event,
                "before_frame": before_frame,
                "after_frame": after_frame,
                "passed": passed,
            }
        )
        if passed:
            continue

        order_violations += 1
        timeline_violations.append(
            {
                "code": "event_order_violation",
                "before_event": before_event,
                "after_event": after_event,
                "before_frame": before_frame,
                "after_frame": after_frame,
            }
        )

    duplicate_greeting_diagnostics = duplicate_greeting_violations(
        scene=scene,
        head_motion=head_motion,
        event_records=event_records,
    )

    return {
        "semantic_event_expected_missing": sum(
            1 for event in expected_events if counts_by_type.get(event, 0) == 0
        ),
        "semantic_event_unexpected_by_scene": len(forbidden_events_by_type),
        "semantic_event_first_frame_tolerance": (
            SEMANTIC_EVENT_FIRST_FRAME_TOLERANCE
        ),
        "semantic_event_expected_first_frame_by_type": expected_first_frames,
        "semantic_event_first_frame_diagnostics": first_frame_diagnostics,
        "semantic_event_trigger_timing_errors": trigger_timing_errors,
        "semantic_event_forbidden_events_by_type": forbidden_events_by_type,
        "semantic_event_order_violations": order_violations,
        "semantic_event_order_diagnostics": order_diagnostics,
        "semantic_event_duplicate_greeting_violation_count": len(
            duplicate_greeting_diagnostics
        ),
        "semantic_event_duplicate_greeting_violations": (
            duplicate_greeting_diagnostics
        ),
        "semantic_event_timeline_violations": timeline_violations,
    }


def duplicate_greeting_violations(
    *,
    scene: str,
    head_motion: str,
    event_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for contract in _active_duplicate_greeting_contracts(scene, head_motion):
        event = contract["event"]
        records = [record for record in event_records if record.get("event") == event]
        observed_count = len(records)
        max_count = contract["max_count"]
        if observed_count <= max_count:
            continue

        frames = _int_record_values(records, "frame")
        timestamps_ms = _int_record_values(records, "timestamp")
        lines = _int_record_values(records, "line")
        violation = {
            "scene": scene,
            "person_label": contract["person_label"],
            "event": event,
            "max_count": max_count,
            "observed_count": observed_count,
            "track_ids": sorted(set(_int_record_values(records, "track_id"))),
            "event_ids": [
                record["event_id"]
                for record in records
                if isinstance(record.get("event_id"), str)
            ],
        }
        if frames:
            violation["frames"] = frames
        if timestamps_ms:
            violation["timestamps_ms"] = timestamps_ms
        if lines:
            violation["lines"] = lines
        violations.append(violation)
    return violations


def _int_record_values(records: list[dict[str, Any]], key: str) -> list[int]:
    return [
        value
        for record in records
        if isinstance((value := record.get(key)), int) and not isinstance(value, bool)
    ]


@dataclass
class _SemanticEventStatsAccumulator:
    scene: str
    head_motion: str
    cooldown_ms: int
    semantic_event_frames: int = 0
    semantic_event_count: int = 0
    semantic_event_schema_errors: int = 0
    semantic_event_unknown_type_count: int = 0
    semantic_event_id_format_errors: int = 0
    semantic_event_duplicate_id_count: int = 0
    semantic_event_duplicate_track_event_count: int = 0
    semantic_event_type_cooldown_errors: int = 0
    semantic_event_confidence_errors: int = 0
    semantic_event_duration_errors: int = 0
    semantic_event_empty_text_count: int = 0
    semantic_event_track_missing_frames: int = 0
    semantic_event_motion_sensitive_count: int = 0

    def __post_init__(self) -> None:
        self._counts_by_type: dict[str, int] = {}
        self._first_frame_by_type: dict[str, int] = {}
        self._seen_event_ids: set[str] = set()
        self._last_track_event_ms: dict[tuple[int, str], int] = {}
        self._last_event_type_ms: dict[str, int] = {}
        self._event_records: list[dict[str, Any]] = []

    def observe(self, response: dict[str, Any]) -> None:
        raw_events = response.get("semantic_events", [])
        if not isinstance(raw_events, list):
            self.semantic_event_schema_errors += 1
            return
        if raw_events:
            self.semantic_event_frames += 1

        frame_id = response.get("frame_id", 0)
        frame_index = int(frame_id) if _is_number(frame_id) else 0
        timestamp_ms = response.get("frame_timestamp_ms", frame_index)
        timestamp = int(timestamp_ms) if _is_number(timestamp_ms) else frame_index
        valid_tracks = _valid_tracks_from_response(response)
        visible_or_lost_track_ids = {int(track["track_id"]) for track in valid_tracks}
        frame_has_missing_track_event = False

        for raw_event in raw_events:
            if not isinstance(raw_event, dict) or not _valid_semantic_event_schema(raw_event):
                self.semantic_event_schema_errors += 1
                continue

            self.semantic_event_count += 1
            event_name = str(raw_event["event"])
            event_id = str(raw_event["event_id"])
            track_id = int(raw_event["track_id"])
            self._event_records.append(
                {
                    "event": event_name,
                    "frame": frame_index,
                    "timestamp": timestamp,
                    "track_id": track_id,
                    "event_id": event_id,
                }
            )
            self._counts_by_type[event_name] = self._counts_by_type.get(event_name, 0) + 1
            self._first_frame_by_type.setdefault(event_name, frame_index)

            if event_name not in _SEMANTIC_EVENT_TYPES:
                self.semantic_event_unknown_type_count += 1
            if event_name in _MOTION_SENSITIVE_EVENT_TYPES:
                self.semantic_event_motion_sensitive_count += 1

            if not _SEMANTIC_EVENT_ID.fullmatch(event_id):
                self.semantic_event_id_format_errors += 1
            if event_id in self._seen_event_ids:
                self.semantic_event_duplicate_id_count += 1
            self._seen_event_ids.add(event_id)

            track_event_key = (track_id, event_name)
            previous_ms = self._last_track_event_ms.get(track_event_key)
            if (
                previous_ms is not None
                and timestamp - previous_ms < self.cooldown_ms
            ):
                self.semantic_event_duplicate_track_event_count += 1
            self._last_track_event_ms[track_event_key] = timestamp

            previous_type_ms = self._last_event_type_ms.get(event_name)
            if (
                previous_type_ms is not None
                and timestamp - previous_type_ms < self.cooldown_ms
            ):
                self.semantic_event_type_cooldown_errors += 1
            self._last_event_type_ms[event_name] = timestamp

            confidence = float(raw_event["confidence"])
            if (
                not math.isfinite(confidence)
                or confidence < 0.0
                or confidence > 1.0
            ):
                self.semantic_event_confidence_errors += 1
            if int(raw_event["duration_ms"]) < 0:
                self.semantic_event_duration_errors += 1
            if str(raw_event["text"]).strip() == "":
                self.semantic_event_empty_text_count += 1
            if event_name != "person_left" and track_id not in visible_or_lost_track_ids:
                frame_has_missing_track_event = True

        if frame_has_missing_track_event:
            self.semantic_event_track_missing_frames += 1

    def summary(self) -> dict[str, Any]:
        contract_summary = _semantic_event_contract_summary(
            scene=self.scene,
            head_motion=self.head_motion,
            counts_by_type=self._counts_by_type,
            first_frame_by_type=self._first_frame_by_type,
            event_records=self._event_records,
        )
        return {
            "semantic_event_frames": self.semantic_event_frames,
            "semantic_event_count": self.semantic_event_count,
            "semantic_event_counts_by_type": dict(sorted(self._counts_by_type.items())),
            "semantic_event_first_frame_by_type": dict(
                sorted(self._first_frame_by_type.items())
            ),
            "semantic_event_schema_errors": self.semantic_event_schema_errors,
            "semantic_event_unknown_type_count": self.semantic_event_unknown_type_count,
            "semantic_event_id_format_errors": self.semantic_event_id_format_errors,
            "semantic_event_duplicate_id_count": self.semantic_event_duplicate_id_count,
            "semantic_event_duplicate_track_event_count": (
                self.semantic_event_duplicate_track_event_count
            ),
            "semantic_event_cooldown_ms": self.cooldown_ms,
            "semantic_event_type_cooldown_errors": (
                self.semantic_event_type_cooldown_errors
            ),
            "semantic_event_confidence_errors": self.semantic_event_confidence_errors,
            "semantic_event_duration_errors": self.semantic_event_duration_errors,
            "semantic_event_empty_text_count": self.semantic_event_empty_text_count,
            "semantic_event_track_missing_frames": (
                self.semantic_event_track_missing_frames
            ),
            "semantic_event_motion_sensitive_count": (
                self.semantic_event_motion_sensitive_count
            ),
            **contract_summary,
        }


@dataclass
class _AttentionStatsAccumulator:
    attention_frames: int = 0
    attention_null_frames: int = 0
    attention_target_switches: int = 0
    attention_schema_errors: int = 0
    attention_invalid_uv_frames: int = 0
    attention_target_missing_track_frames: int = 0
    attention_target_lost_frames: int = 0
    attention_max_lost_hold_ms: int = 0
    attention_largest_bbox_disagreement_frames: int = 0
    attention_actionable_largest_bbox_disagreement_frames: int = 0

    def __post_init__(self) -> None:
        self._last_target_track_id: int | None = None
        self._target_counts_by_id: dict[str, int] = {}
        self._largest_bbox_challenger_id: int | None = None
        self._largest_bbox_challenger_start_ms: int | None = None

    def observe(self, response: dict[str, Any]) -> None:
        attention = response.get("attention")
        if attention is None:
            self.attention_null_frames += 1
            self._reset_largest_bbox_challenger_dwell()
            return
        if not isinstance(attention, dict) or not _valid_attention_schema(attention):
            self.attention_schema_errors += 1
            self._reset_largest_bbox_challenger_dwell()
            return

        self.attention_frames += 1
        target_id = int(attention["target_track_id"])
        target_key = str(target_id)
        self._target_counts_by_id[target_key] = (
            self._target_counts_by_id.get(target_key, 0) + 1
        )
        if (
            self._last_target_track_id is not None
            and target_id != self._last_target_track_id
        ):
            self.attention_target_switches += 1
        self._last_target_track_id = target_id

        image_size = response.get("image_size")
        if (
            not _number_list(image_size, length=2)
            or float(image_size[0]) < 0.0
            or float(image_size[1]) < 0.0
            or not _uv_in_image(attention["target_uv"], image_size=image_size)
        ):
            self.attention_invalid_uv_frames += 1

        tracks = _valid_tracks_from_response(response)
        tracks_by_id = {int(track["track_id"]): track for track in tracks}
        target_track = tracks_by_id.get(target_id)
        target_lost = False
        if target_track is None:
            self.attention_target_missing_track_frames += 1
        else:
            lost_ms = int(target_track.get("lost_ms", 0))
            if lost_ms > 0:
                target_lost = True
                self.attention_target_lost_frames += 1
                self.attention_max_lost_hold_ms = max(
                    self.attention_max_lost_hold_ms,
                    lost_ms,
                )

        visible_tracks = [
            track for track in tracks if int(track.get("lost_ms", 0)) == 0
        ]
        if visible_tracks:
            largest = max(
                visible_tracks,
                key=lambda track: float(track["bbox_area_ratio"]),
            )
            largest_id = int(largest["track_id"])
            if largest_id != target_id:
                self.attention_largest_bbox_disagreement_frames += 1
                self._observe_actionable_largest_bbox_disagreement(
                    challenger_id=largest_id,
                    response=response,
                    attention=attention,
                    target_lost=target_lost,
                )
            else:
                self._reset_largest_bbox_challenger_dwell()
        else:
            self._reset_largest_bbox_challenger_dwell()

    def _observe_actionable_largest_bbox_disagreement(
        self,
        *,
        challenger_id: int,
        response: dict[str, Any],
        attention: dict[str, Any],
        target_lost: bool,
    ) -> None:
        if attention.get("reason") == "held_lost_target" or target_lost:
            self._reset_largest_bbox_challenger_dwell()
            return

        timestamp_ms = response.get("frame_timestamp_ms")
        has_valid_timestamp = _is_finite_number(timestamp_ms)
        timestamp = int(timestamp_ms) if has_valid_timestamp else None

        if challenger_id != self._largest_bbox_challenger_id:
            self._largest_bbox_challenger_id = challenger_id
            self._largest_bbox_challenger_start_ms = timestamp
            return

        if timestamp is None:
            return
        if self._largest_bbox_challenger_start_ms is None:
            self._largest_bbox_challenger_start_ms = timestamp
            return
        if (
            timestamp - self._largest_bbox_challenger_start_ms
            >= _S4_STABLE_ATTENTION_SWITCH_DWELL_MS
        ):
            self.attention_actionable_largest_bbox_disagreement_frames += 1

    def _reset_largest_bbox_challenger_dwell(self) -> None:
        self._largest_bbox_challenger_id = None
        self._largest_bbox_challenger_start_ms = None

    def summary(self) -> dict[str, Any]:
        return {
            "attention_frames": self.attention_frames,
            "attention_null_frames": self.attention_null_frames,
            "attention_target_switches": self.attention_target_switches,
            "attention_target_counts_by_id": dict(
                sorted(self._target_counts_by_id.items())
            ),
            "attention_schema_errors": self.attention_schema_errors,
            "attention_invalid_uv_frames": self.attention_invalid_uv_frames,
            "attention_target_missing_track_frames": (
                self.attention_target_missing_track_frames
            ),
            "attention_target_lost_frames": self.attention_target_lost_frames,
            "attention_max_lost_hold_ms": self.attention_max_lost_hold_ms,
            "attention_largest_bbox_disagreement_frames": (
                self.attention_largest_bbox_disagreement_frames
            ),
            "attention_actionable_largest_bbox_disagreement_frames": (
                self.attention_actionable_largest_bbox_disagreement_frames
            ),
        }


@dataclass
class _TrackingStatsAccumulator:
    track_frames: int = 0
    largest_bbox_track_switches: int = 0
    duplicate_track_id_frames: int = 0
    single_visible_id_switches: int = 0
    adjacent_track_matches: int = 0
    association_id_switches: int = 0
    track_schema_errors: int = 0
    age_monotonic_violations: int = 0

    def __post_init__(self) -> None:
        self._last_largest_bbox_track_id: int | None = None
        self._largest_bbox_counts: dict[int, int] = {}
        self._largest_bbox_timestamps_by_id: dict[int, list[int]] = {}
        self._last_age_by_id: dict[int, int] = {}
        self._previous_visible_tracks: list[dict[str, Any]] = []
        self._previous_single_visible_id: int | None = None
        self._visible_counts_by_id: dict[str, int] = {}

    def observe(self, response: dict[str, Any]) -> None:
        timestamp_ms = response.get("frame_timestamp_ms")
        if not _is_number(timestamp_ms):
            timestamp_ms = response.get("frame_id", 0)
        timestamp_ms = int(timestamp_ms) if _is_number(timestamp_ms) else 0

        raw_tracks = response.get("tracks", [])
        if not isinstance(raw_tracks, list):
            self.track_schema_errors += 1
            self._observe_visible_tracks([], timestamp_ms=timestamp_ms)
            return

        valid_tracks: list[dict[str, Any]] = []
        for raw_track in raw_tracks:
            if not isinstance(raw_track, dict) or not _valid_track_schema(raw_track):
                self.track_schema_errors += 1
                continue
            track_id = int(raw_track["track_id"])
            age_ms = int(raw_track["age_ms"])
            previous_age = self._last_age_by_id.get(track_id)
            if previous_age is not None and age_ms < previous_age:
                self.age_monotonic_violations += 1
            self._last_age_by_id[track_id] = age_ms
            valid_tracks.append(raw_track)

        if not valid_tracks:
            self._observe_visible_tracks([], timestamp_ms=timestamp_ms)
            return

        self.track_frames += 1
        self._observe_duplicate_track_ids(valid_tracks)
        visible_tracks = [
            track for track in valid_tracks if int(track.get("lost_ms", 0)) == 0
        ]
        self._observe_visible_tracks(visible_tracks, timestamp_ms=timestamp_ms)

    def summary(self) -> dict[str, Any]:
        largest_bbox_track_id: int | None = None
        largest_bbox_count = 0
        if self._largest_bbox_counts:
            largest_bbox_track_id, largest_bbox_count = max(
                self._largest_bbox_counts.items(),
                key=lambda item: (item[1], -item[0]),
            )
        timestamps = (
            self._largest_bbox_timestamps_by_id.get(largest_bbox_track_id, [])
            if largest_bbox_track_id is not None
            else []
        )
        max_gap_ms = 0
        if len(timestamps) > 1:
            max_gap_ms = max(
                int(timestamps[index] - timestamps[index - 1])
                for index in range(1, len(timestamps))
            )
        coverage = (
            largest_bbox_count / self.track_frames
            if self.track_frames > 0 and largest_bbox_track_id is not None
            else 0.0
        )
        return {
            "track_frames": self.track_frames,
            "largest_bbox_track_switches": self.largest_bbox_track_switches,
            "largest_bbox_track_id": largest_bbox_track_id,
            "largest_bbox_track_coverage": coverage,
            "largest_bbox_track_max_gap_ms": max_gap_ms,
            "duplicate_track_id_frames": self.duplicate_track_id_frames,
            "single_visible_id_switches": self.single_visible_id_switches,
            "adjacent_track_matches": self.adjacent_track_matches,
            "association_id_switches": self.association_id_switches,
            "visible_counts_by_id": dict(sorted(self._visible_counts_by_id.items())),
            "track_schema_errors": self.track_schema_errors,
            "age_monotonic_violations": self.age_monotonic_violations,
        }

    def _observe_visible_tracks(
        self,
        visible_tracks: list[dict[str, Any]],
        *,
        timestamp_ms: int,
    ) -> None:
        visible_ids = [int(track["track_id"]) for track in visible_tracks]
        unique_visible_ids = set(visible_ids)
        for track_id in unique_visible_ids:
            key = str(track_id)
            self._visible_counts_by_id[key] = self._visible_counts_by_id.get(key, 0) + 1

        if len(visible_tracks) == 1:
            current_id = int(visible_tracks[0]["track_id"])
            if (
                self._previous_single_visible_id is not None
                and current_id != self._previous_single_visible_id
            ):
                self.single_visible_id_switches += 1
            self._previous_single_visible_id = current_id
        else:
            self._previous_single_visible_id = None

        for previous, current in _associate_adjacent_tracks(
            self._previous_visible_tracks,
            visible_tracks,
        ):
            self.adjacent_track_matches += 1
            if int(previous["track_id"]) != int(current["track_id"]):
                self.association_id_switches += 1
        self._previous_visible_tracks = visible_tracks

        if not visible_tracks:
            self._last_largest_bbox_track_id = None
            return

        largest = max(visible_tracks, key=lambda track: float(track["bbox_area_ratio"]))
        largest_id = int(largest["track_id"])
        if (
            self._last_largest_bbox_track_id is not None
            and largest_id != self._last_largest_bbox_track_id
        ):
            self.largest_bbox_track_switches += 1
        self._last_largest_bbox_track_id = largest_id
        self._largest_bbox_counts[largest_id] = (
            self._largest_bbox_counts.get(largest_id, 0) + 1
        )
        self._largest_bbox_timestamps_by_id.setdefault(largest_id, []).append(
            timestamp_ms
        )

    def _observe_duplicate_track_ids(self, valid_tracks: list[dict[str, Any]]) -> None:
        track_ids = [int(track["track_id"]) for track in valid_tracks]
        if len(track_ids) != len(set(track_ids)):
            self.duplicate_track_id_frames += 1


def _valid_track_schema(track: dict[str, Any]) -> bool:
    required = {
        "track_id",
        "class",
        "bbox_xyxy",
        "bbox_area_ratio",
        "center_uv",
        "head_uv",
        "velocity_uv_s",
        "age_ms",
        "lost_ms",
        "confidence",
        "pose_confidence",
    }
    if not required.issubset(track):
        return False
    return (
        isinstance(track["track_id"], int)
        and not isinstance(track["track_id"], bool)
        and isinstance(track["class"], str)
        and _number_list(track["bbox_xyxy"], length=4)
        and _is_number(track["bbox_area_ratio"])
        and float(track["bbox_area_ratio"]) >= 0.0
        and _number_list(track["center_uv"], length=2)
        and _number_list(track["head_uv"], length=2)
        and _number_list(track["velocity_uv_s"], length=2)
        and _non_negative_int(track["age_ms"])
        and _non_negative_int(track["lost_ms"])
        and _is_number(track["confidence"])
        and _is_number(track["pose_confidence"])
    )


def _valid_tracks_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tracks = response.get("tracks", [])
    if not isinstance(raw_tracks, list):
        return []
    return [
        raw_track
        for raw_track in raw_tracks
        if isinstance(raw_track, dict) and _valid_track_schema(raw_track)
    ]


def _valid_attention_schema(attention: dict[str, Any]) -> bool:
    required = {"target_track_id", "target_uv", "reason", "confidence"}
    if not required.issubset(attention):
        return False
    return (
        isinstance(attention["target_track_id"], int)
        and not isinstance(attention["target_track_id"], bool)
        and _number_list(attention["target_uv"], length=2)
        and isinstance(attention["reason"], str)
        and _is_number(attention["confidence"])
    )


def _valid_semantic_event_schema(event: dict[str, Any]) -> bool:
    required = {
        "type",
        "event_id",
        "event",
        "camera",
        "track_id",
        "confidence",
        "duration_ms",
        "text",
    }
    if not required.issubset(event):
        return False
    return (
        event["type"] == "semantic_event"
        and isinstance(event["event_id"], str)
        and isinstance(event["event"], str)
        and isinstance(event["camera"], str)
        and isinstance(event["track_id"], int)
        and not isinstance(event["track_id"], bool)
        and _is_number(event["confidence"])
        and _non_bool_int(event["duration_ms"])
        and isinstance(event["text"], str)
    )


def _uv_in_image(value: Any, *, image_size: list[Any]) -> bool:
    if not _number_list(value, length=2):
        return False
    x, y = [float(item) for item in value]
    width, height = [float(item) for item in image_size]
    return (
        _is_finite_number(x)
        and _is_finite_number(y)
        and _is_finite_number(width)
        and _is_finite_number(height)
        and 0.0 <= x <= width
        and 0.0 <= y <= height
    )


def _associate_adjacent_tracks(
    previous_tracks: list[dict[str, Any]],
    current_tracks: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[float, int, int]] = []
    for previous_index, previous in enumerate(previous_tracks):
        for current_index, current in enumerate(current_tracks):
            iou = _bbox_iou(previous["bbox_xyxy"], current["bbox_xyxy"])
            if iou >= _ASSOCIATION_IOU_THRESHOLD:
                pairs.append((iou, previous_index, current_index))

    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _iou, previous_index, current_index in pairs:
        if previous_index in used_previous or current_index in used_current:
            continue
        used_previous.add(previous_index)
        used_current.add(current_index)
        matches.append((previous_tracks[previous_index], current_tracks[current_index]))
    return matches


def _bbox_iou(first: list[Any], second: list[Any]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _number_list(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_is_number(item) for item in value)
    )


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _non_bool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _default_connector() -> Callable[..., Any]:
    import websockets

    return websockets.connect


if __name__ == "__main__":
    main()
