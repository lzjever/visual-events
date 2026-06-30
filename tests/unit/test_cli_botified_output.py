from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VISUAL_STATE_TRACKING_SAMPLE = (
    REPO_ROOT / "common" / "schema" / "samples" / "visual_state_tracking.json"
)

EXPECTED_ALLOWED_EVENTS = {
    "familiar_unknown_present",
    "known_person_present",
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
    "scene_activated",
}
BOTIFIED_OPEN = "<botified>"
BOTIFIED_CLOSE = "</botified>"
VALID_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def import_botified_output():
    try:
        import visual_events_cli.botified_output as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.botified_output module: {exc}")
    return module


def load_visual_state_tracking() -> dict[str, Any]:
    return json.loads(VISUAL_STATE_TRACKING_SAMPLE.read_text(encoding="utf-8"))


def semantic_event(
    *,
    event_id: str = "front:evt_000456",
    event: str = "person_waving",
    track_id: int = 7,
    evidence: dict[str, Any] | None = None,
    text: str = "有人在机器人前方挥手",
) -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": event_id,
        "event": event,
        "camera": "front",
        "track_id": track_id,
        "confidence": 0.86,
        "duration_ms": 900,
        "evidence": evidence
        if evidence is not None
        else {
            "runtime_person_slot": 3,
            "wrist_x_span_px": 84.0,
            "wrist_x_span_bbox_ratio": 0.42,
            "wrist_y_relative_to_shoulder_px": 18.0,
            "wave_duration_ms": 900,
            "keypoint_min_confidence": 0.72,
        },
        "text": text,
    }


def known_person_event(
    *,
    event_id: str = "front:mem_evt_000001",
    person_id: str = "person_000001",
    track_id: int = 7,
    memory_match_id: str = "match_000001",
) -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": event_id,
        "event": "known_person_present",
        "camera": "front",
        "track_id": track_id,
        "confidence": 0.86,
        "duration_ms": 0,
        "lifecycle_state": "confirmed",
        "evidence": {
            "memory_match_id": memory_match_id,
            "matched_type": "person",
            "matched_id": person_id,
            "embedding_id": "emb_face_000001",
            "match_type": "face",
            "match_score": 0.86,
            "top2_margin": 0.09,
            "source_target_mode": "track_id",
            "source_frame_id": 1234,
            "source_frame_timestamp_ms": 1780000000000,
            "embedding_model": "local-face-v1",
            "vector_blob": "must-not-leak",
        },
        "memory_context": {
            "person": {
                "person_id": person_id,
                "display_name": "张三",
                "description": "店长，熟悉新品陈列和现场活动",
                "tags": ["staff", "manager"],
                "match_confidence": 0.86,
                "embedding": [0.1, 0.2],
            },
            "conversation_summaries": [
                "上次问过新品尺码，偏好浅色外套。",
                "第二条摘要不应进入短上下文。",
            ],
            "raw_notes": "must-not-leak",
        },
        "text": "看到已知人物：张三",
    }


def scene_memory_event(
    *,
    event_id: str = "front:mem_evt_scene_000001",
    scene_id: str = "scene_000001",
    memory_match_id: str = "match_scene_000001",
) -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": event_id,
        "event": "scene_activated",
        "camera": "front",
        "track_id": None,
        "confidence": 0.82,
        "duration_ms": 0,
        "lifecycle_state": "confirmed",
        "evidence": {
            "memory_match_id": memory_match_id,
            "matched_type": "scene",
            "matched_id": scene_id,
            "embedding_id": "emb_scene_000001",
            "match_type": "scene",
            "match_score": 0.82,
            "top2_margin": 0.07,
            "source_target_mode": "scene",
        },
        "memory_context": {
            "scene": {
                "scene_id": scene_id,
                "title": "新品展示区",
                "description": "这里是本周新品展示区。",
                "activation_hint": "可以介绍新品活动。",
                "match_confidence": 0.82,
                "embedding": [0.1, 0.2],
            }
        },
        "text": "看到已教学场景：新品展示区",
    }


def familiar_unknown_event(
    *,
    event_id: str = "front:mem_evt_anon_000001",
    anonymous_id: str = "anon_000123",
    track_id: int = 7,
    memory_match_id: str = "match_anon_000001",
) -> dict[str, Any]:
    return {
        "type": "semantic_event",
        "event_id": event_id,
        "event": "familiar_unknown_present",
        "camera": "front",
        "track_id": track_id,
        "confidence": 0.81,
        "duration_ms": 0,
        "lifecycle_state": "confirmed",
        "evidence": {
            "memory_match_id": memory_match_id,
            "matched_type": "anonymous_person",
            "matched_id": anonymous_id,
            "embedding_id": "emb_face_000245",
            "match_type": "face",
            "match_score": 0.81,
            "top2_margin": 0.08,
            "source_target_mode": "track_id",
        },
        "memory_context": {
            "anonymous_person": {
                "anonymous_id": anonymous_id,
                "seen_count": 5,
                "familiar_score": 0.81,
                "last_seen_at_ms": 1780000000000,
                "debug_crop": "must-not-leak",
            }
        },
        "text": "看到一个经常出现但尚未命名的人",
    }


def known_identity_context() -> dict[str, Any]:
    return {
        "status": "known_person",
        "source": "cache",
        "fresh_ms": 120,
        "confidence": 0.91,
        "person": {
            "person_id": "person_000001",
            "display_name": "张三",
            "description": "店长，熟悉新品陈列和现场活动",
            "tags": ["staff", "manager"],
            "bbox_xyxy": [1, 2, 3, 4],
            "embedding": [0.1, 0.2],
            "crop_ref": "runtime/private/crop.jpg",
        },
        "track_id": 7,
        "stream_ref": "ws_1",
    }


def familiar_unknown_identity_context() -> dict[str, Any]:
    return {
        "status": "familiar_unknown",
        "source": "cache",
        "fresh_ms": 250,
        "confidence": 0.88,
        "anonymous_person": {
            "anonymous_id": "anon_seeded",
            "seen_count": 5,
            "observed_duration_ms": 4_200,
            "familiar_score": 0.88,
            "bbox_xyxy": [1, 2, 3, 4],
            "keypoints": [{"name": "nose"}],
            "embedding": [1.0, 0.0],
            "crop_ref": "runtime/private/crop.jpg",
            "stream_ref": "ws_1",
        },
        "raw_track_id": 7,
    }


def parse_botified_frame(frame: str, *, event_id: str) -> dict[str, Any]:
    assert "\n" not in frame
    assert frame.startswith(BOTIFIED_OPEN)
    assert frame.endswith(BOTIFIED_CLOSE)
    assert frame.count(BOTIFIED_OPEN) == 1
    assert frame.count(BOTIFIED_CLOSE) == 1

    inner = frame[len(BOTIFIED_OPEN) : -len(BOTIFIED_CLOSE)]
    payload = json.loads(inner)
    assert payload["id"] == f"visual:{event_id}"
    assert payload["urgency"] == "normal"
    assert payload["timeout_secs"] == 8
    assert payload["expect"] == "ack"
    assert isinstance(payload["request"], str)
    assert payload["request"].strip()
    return payload


def parse_visual_context(payload: dict[str, Any]) -> dict[str, Any]:
    marker = "visual_context="
    request = payload["request"]
    start = request.index(marker) + len(marker)
    wrapper, end = json.JSONDecoder().raw_decode(request[start:])
    assert request[start + end :].strip() == ""
    assert set(wrapper) == {"visual_context"}
    return wrapper["visual_context"]


def visual_state_with_events(
    events: list[dict[str, Any]],
    *,
    timestamp_ms: int = 1_710_000_000_000,
    scene_context: dict[str, Any] | None = None,
    tracks: list[dict[str, Any]] | None = None,
    scene_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = {
        "type": "visual_state",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 1,
        "frame_timestamp_ms": timestamp_ms,
        "image_size": [1280, 720],
        "attention": {"target_track_id": 7, "confidence": 0.91},
        "tracks": tracks
        if tracks is not None
        else [
            {
                "track_id": 7,
                "class": "person",
                "bbox_xyxy": [320.0, 120.0, 520.0, 600.0],
                "bbox_area_ratio": 0.1042,
                "center_uv": [420.0, 360.0],
                "head_uv": [421.0, 205.0],
                "lost_ms": 0,
            }
        ],
        "scene_context": scene_context
        if scene_context is not None
        else {
            "engagement_state": "available",
            "attention_available": True,
            "target_track_id": 7,
            "no_engage_reasons": [],
            "target_reacquired": None,
        },
        "semantic_events": events,
    }
    if scene_flags is not None:
        state["scene_flags"] = scene_flags
    return state


def current_snapshot_state(
    *,
    semantic_events: list[dict[str, Any]] | None = None,
    identity_context: dict[str, Any] | None = None,
    timestamp_ms: int = 1_710_000_000_000,
) -> dict[str, Any]:
    state = visual_state_with_events(
        semantic_events or [],
        timestamp_ms=timestamp_ms,
        tracks=[
            {
                "track_id": 7,
                "class": "person",
                "bbox_xyxy": [420.0, 90.0, 780.0, 690.0],
                "bbox_area_ratio": 0.24,
                "center_uv": [600.0, 390.0],
                "head_uv": [602.0, 180.0],
                "lost_ms": 0,
                "stream_ref": "ws_1",
            },
            {
                "track_id": 8,
                "class": "person",
                "bbox_xyxy": [860.0, 160.0, 1030.0, 610.0],
                "bbox_area_ratio": 0.083,
                "center_uv": [945.0, 385.0],
                "head_uv": [946.0, 230.0],
                "lost_ms": 0,
                "keypoints": [{"name": "nose"}],
            },
            {
                "track_id": 9,
                "class": "person",
                "bbox_xyxy": [20.0, 120.0, 120.0, 500.0],
                "bbox_area_ratio": 0.041,
                "center_uv": [70.0, 310.0],
                "lost_ms": 500,
            },
        ],
        scene_context={
            "engagement_state": "available",
            "attention_available": True,
            "target_track_id": 7,
            "no_engage_reasons": [],
            "target_reacquired": None,
        },
    )
    if identity_context is not None:
        state["identity_context"] = identity_context
    return state


def assert_no_current_snapshot_forbidden_fields(snapshot: dict[str, Any]) -> None:
    serialized = json.dumps(snapshot, ensure_ascii=False)
    for forbidden in (
        "track_id",
        "bbox",
        "bbox_xyxy",
        "keypoints",
        "embedding",
        "crop",
        "crop_ref",
        "stream_ref",
        "raw_track_id",
        "source_frame",
        "request_snapshot_ref",
    ):
        assert forbidden not in serialized


def test_allowed_events_constant_is_exact_botified_events():
    module = import_botified_output()

    allowed = getattr(module, "BOTIFIED_ALLOWED_EVENTS", None)
    assert isinstance(allowed, (tuple, set))
    assert set(allowed) == EXPECTED_ALLOWED_EVENTS
    assert len(allowed) == len(EXPECTED_ALLOWED_EVENTS)


@pytest.mark.parametrize("event_name", sorted(EXPECTED_ALLOWED_EVENTS))
def test_format_botified_frame_outputs_allowlisted_events(event_name):
    module = import_botified_output()
    event = semantic_event(event=f"{event_name}", event_id=f"front:{event_name}")

    frame = module.format_botified_frame(event)
    assert frame is not None

    payload = parse_botified_frame(frame, event_id=event["event_id"])
    assert event["event"] in payload["request"]
    assert event["camera"] in payload["request"]
    assert "track_id=" not in payload["request"]
    assert str(event["confidence"]) in payload["request"]
    assert event["text"] in payload["request"]


def test_attention_target_changed_is_not_output_to_botified():
    module = import_botified_output()
    visual_state = load_visual_state_tracking()
    visual_state["semantic_events"] = [
        semantic_event(event="attention_target_changed", event_id="front:evt_attn_001")
    ]

    assert module.format_botified_frame(visual_state["semantic_events"][0]) is None
    assert module.BotifiedEventMapper().frames_from_visual_state(visual_state) == []


def test_same_event_id_outputs_only_once_within_frame_and_across_frames():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(max_seen_event_ids=1024)
    first_event = semantic_event(event_id="front:evt_000456")
    duplicate_event = copy.deepcopy(first_event)
    duplicate_event["event"] = "person_appeared"
    visual_state = load_visual_state_tracking()
    visual_state["semantic_events"] = [first_event, duplicate_event]

    first_frames = mapper.frames_from_visual_state(visual_state)
    second_frames = mapper.frames_from_visual_state(
        {**visual_state, "frame_id": visual_state["frame_id"] + 1}
    )

    assert len(first_frames) == 1
    assert parse_botified_frame(first_frames[0], event_id="front:evt_000456")
    assert second_frames == []


def test_appeared_and_ordinary_left_are_suppressed_by_mapper():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    appeared = semantic_event(event_id="front:appeared", event="person_appeared")
    ordinary_left = semantic_event(
        event_id="front:left",
        event="person_left",
        evidence={
            "runtime_person_slot": 44,
            "lost_duration_ms": 350,
            "last_bbox_area_ratio": 0.05,
        },
    )

    frames = mapper.frames_from_visual_state(
        visual_state_with_events([appeared, ordinary_left])
    )

    assert frames == []


@pytest.mark.parametrize(
    ("evidence", "no_engage_reasons"),
    [
        ({"runtime_person_slot": 3, "passing_speed_class": "fast"}, []),
        ({"runtime_person_slot": 3, "passing_speed_class": "medium"}, ["passing_fast"]),
    ],
)
def test_fast_passing_is_suppressed_by_mapper(
    evidence: dict[str, Any],
    no_engage_reasons: list[str],
):
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    event = semantic_event(
        event_id="front:passing",
        event="person_passing_by",
        evidence=evidence,
    )
    state = visual_state_with_events(
        [event],
        scene_context={
            "engagement_state": "no_engage_target",
            "attention_available": False,
            "target_track_id": 7,
            "no_engage_reasons": no_engage_reasons,
            "target_reacquired": None,
        },
    )

    assert mapper.frames_from_visual_state(state) == []


def test_approaching_pending_upgrades_to_stopped_or_waving_within_window():
    module = import_botified_output()
    now_ms = 1_710_000_000_100
    mapper = module.BotifiedEventMapper(clock_ms=lambda: now_ms)
    approaching = semantic_event(
        event_id="front:approaching",
        event="person_approaching_robot",
        evidence={
            "runtime_person_slot": 3,
            "bbox_area_ratio_start": 0.02,
            "bbox_area_ratio_end": 0.08,
            "area_growth_ratio": 4.0,
            "area_delta": 0.06,
            "camera_motion_state": "stationary",
        },
    )
    stopped = semantic_event(
        event_id="front:stopped",
        event="person_stopped_near_robot",
        evidence={
            "runtime_person_slot": 3,
            "bbox_area_ratio": 0.11,
            "speed_px_s_p95": 8.0,
            "stationary_duration_ms": 1100,
            "camera_motion_state": "stationary",
        },
    )

    assert mapper.frames_from_visual_state(visual_state_with_events([approaching])) == []
    now_ms += 500
    frames = mapper.frames_from_visual_state(visual_state_with_events([stopped]))

    assert len(frames) == 1
    payload = parse_botified_frame(frames[0], event_id="front:stopped")
    assert "person_stopped_near_robot" in payload["request"]
    assert "person_approaching_robot" not in payload["request"]


def test_pending_approaching_flushes_after_window_on_next_mapper_call():
    module = import_botified_output()
    now_ms = 1_710_000_000_100
    mapper = module.BotifiedEventMapper(clock_ms=lambda: now_ms)
    approaching = semantic_event(
        event_id="front:approaching",
        event="person_approaching_robot",
        evidence={
            "runtime_person_slot": 3,
            "bbox_area_ratio_start": 0.02,
            "bbox_area_ratio_end": 0.08,
            "area_growth_ratio": 4.0,
            "area_delta": 0.06,
            "camera_motion_state": "stationary",
        },
    )

    assert mapper.frames_from_visual_state(visual_state_with_events([approaching])) == []
    now_ms += 799
    assert mapper.frames_from_visual_state(visual_state_with_events([])) == []
    now_ms += 1
    frames = mapper.frames_from_visual_state(visual_state_with_events([]))

    assert len(frames) == 1
    payload = parse_botified_frame(frames[0], event_id="front:approaching")
    assert "person_approaching_robot" in payload["request"]


def test_waving_frame_contains_parseable_visual_context_and_stable_top_level_payload():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    event = semantic_event(event_id="front:waving", event="person_waving")

    frames = mapper.frames_from_visual_state(visual_state_with_events([event]))

    assert len(frames) == 1
    payload = parse_botified_frame(frames[0], event_id="front:waving")
    assert set(payload) == {"id", "urgency", "timeout_secs", "request", "expect"}
    assert "track_id=" not in payload["request"]
    assert "target_ref=current:front:person:0" in payload["request"]
    context = parse_visual_context(payload)
    assert set(context) == {"event_target", "trigger_evidence", "current_scene"}
    assert context["event_target"]["target_ref"] == "current:front:person:0"
    assert "track_id" not in context["event_target"]
    assert context["event_target"]["runtime_person_slot"] == 3
    assert context["event_target"]["visible_now"] is True
    assert context["event_target"]["position"] in {"left", "center", "right"}
    assert context["event_target"]["size"] in {"far", "mid", "near"}
    assert (
        context["current_scene"]["attention_target"]["target_ref"]
        == "current:front:person:0"
    )
    assert "track_id" not in context["current_scene"]["attention_target"]
    assert context["current_scene"]["attention_target"]["position"] in {
        "left",
        "center",
        "right",
    }
    assert context["current_scene"]["attention_target"]["size"] in {
        "far",
        "mid",
        "near",
    }
    assert context["current_scene"]["attention_target"]["center_uv"] == [420.0, 360.0]
    assert context["current_scene"]["attention_target"]["bbox_area_ratio"] == 0.1042


def test_pending_approaching_flush_uses_latest_visual_state_for_current_scene():
    module = import_botified_output()
    now_ms = 1_710_000_000_100
    mapper = module.BotifiedEventMapper(clock_ms=lambda: now_ms)
    approaching = semantic_event(
        event_id="front:approaching_latest",
        event="person_approaching_robot",
        evidence={
            "runtime_person_slot": 3,
            "bbox_area_ratio_start": 0.02,
            "bbox_area_ratio_end": 0.08,
            "area_growth_ratio": 4.0,
            "area_delta": 0.06,
            "camera_motion_state": "stationary",
        },
    )

    old_state = visual_state_with_events(
        [approaching],
        timestamp_ms=now_ms - 100,
        scene_flags={"person_count": 9},
        scene_context={
            "engagement_state": "available",
            "attention_available": True,
            "target_track_id": 7,
            "no_engage_reasons": [],
            "target_reacquired": None,
        },
    )
    assert mapper.frames_from_visual_state(old_state) == []

    now_ms += 800
    latest_state = visual_state_with_events(
        [],
        timestamp_ms=now_ms - 20,
        scene_flags={"person_count": 2},
        tracks=[],
        scene_context={
            "engagement_state": "no_engage_target",
            "attention_available": False,
            "target_track_id": None,
            "no_engage_reasons": ["too_far"],
            "target_reacquired": None,
        },
    )
    frames = mapper.frames_from_visual_state(latest_state)

    context = parse_visual_context(
        parse_botified_frame(frames[0], event_id="front:approaching_latest")
    )
    assert context["event_target"]["visible_now"] is False
    assert context["event_target"]["position"] == "unknown"
    assert context["event_target"]["size"] == "unknown"
    assert context["current_scene"]["frame_age_ms"] == 20
    assert context["current_scene"]["person_count"] == 2
    assert context["current_scene"]["engagement_state"] == "no_engage_target"
    assert context["current_scene"]["attention_target"] is None


def test_current_scene_person_count_uses_visible_tracks_when_scene_flags_missing():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    event = semantic_event(event_id="front:waving_visible_count", event="person_waving")
    tracks = [
        {
            "track_id": 7,
            "class": "person",
            "bbox_xyxy": [320.0, 120.0, 520.0, 600.0],
            "bbox_area_ratio": 0.1042,
            "center_uv": [420.0, 360.0],
            "head_uv": [421.0, 205.0],
            "lost_ms": 0,
        },
        {
            "track_id": 8,
            "class": "person",
            "bbox_xyxy": [700.0, 180.0, 850.0, 610.0],
            "bbox_area_ratio": 0.07,
            "center_uv": [775.0, 395.0],
            "head_uv": [776.0, 240.0],
            "lost_ms": 500,
        },
        {
            "track_id": 9,
            "class": "chair",
            "bbox_xyxy": [100.0, 200.0, 180.0, 360.0],
            "bbox_area_ratio": 0.02,
            "center_uv": [140.0, 280.0],
            "lost_ms": 0,
        },
    ]

    frames = mapper.frames_from_visual_state(
        visual_state_with_events([event], tracks=tracks)
    )

    context = parse_visual_context(
        parse_botified_frame(frames[0], event_id="front:waving_visible_count")
    )
    assert context["current_scene"]["person_count"] == 1


def test_trigger_evidence_only_contains_whitelisted_projection_fields():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    event = semantic_event(
        event_id="front:waving",
        event="person_waving",
        evidence={
            "runtime_person_slot": 3,
            "wrist_x_span_px": 84.0,
            "wrist_x_span_bbox_ratio": 0.42,
            "wrist_y_relative_to_shoulder_px": 18.0,
            "wave_duration_ms": 900,
            "keypoint_min_confidence": 0.72,
            "crop_b64": "must-not-leak",
            "embedding": [0.1, 0.2, 0.3],
            "identity_name": "must-not-leak",
        },
    )

    frames = mapper.frames_from_visual_state(visual_state_with_events([event]))

    evidence = parse_visual_context(
        parse_botified_frame(frames[0], event_id="front:waving")
    )["trigger_evidence"]
    assert evidence == {
        "runtime_person_slot": 3,
        "wrist_x_span_px": 84.0,
        "wrist_x_span_bbox_ratio": 0.42,
        "wrist_y_relative_to_shoulder_px": 18.0,
        "wave_duration_ms": 900,
        "keypoint_min_confidence": 0.72,
    }


def test_current_visual_snapshot_projects_visible_people_and_active_target():
    module = import_botified_output()
    state = current_snapshot_state(
        identity_context={
            "overlay_status": "ready",
            "active_target": {"track_id": 8},
            "tracks": [
                {"track_id": 7, "identity": known_identity_context()},
                {"track_id": 8, "identity": familiar_unknown_identity_context()},
            ],
            "stream_ref": "ws_1",
        }
    )

    snapshot = module.build_current_visual_snapshot(state, now_ms=1_710_000_000_100)

    assert snapshot == {
        "type": "current_visual_snapshot",
        "camera": "front",
        "frame_age_ms": 100,
        "person_count": 2,
        "overlay_status": "ready",
        "active_target_ref": "current:front:person:1",
        "people": [
            {
                "target_ref": "current:front:person:0",
                "visible_now": True,
                "attention_target": False,
                "position": "center",
                "size": "near",
                "identity_context": {
                    "status": "known_person",
                    "source": "cache",
                    "fresh_ms": 120,
                    "confidence": 0.91,
                    "person": {
                        "person_id": "person_000001",
                        "display_name": "张三",
                        "description": "店长，熟悉新品陈列和现场活动",
                        "tags": ["staff", "manager"],
                    },
                },
            },
            {
                "target_ref": "current:front:person:1",
                "visible_now": True,
                "attention_target": True,
                "position": "right",
                "size": "mid",
                "identity_context": {
                    "status": "familiar_unknown",
                    "source": "cache",
                    "fresh_ms": 250,
                    "confidence": 0.88,
                    "anonymous_person": {
                        "anonymous_id": "anon_seeded",
                        "seen_count": 5,
                        "observed_duration_ms": 4_200,
                        "familiar_score": 0.88,
                    },
                },
            },
        ],
        "events": [],
    }
    assert_no_current_snapshot_forbidden_fields(snapshot)


def test_current_visual_snapshot_events_use_event_identity_before_track_identity():
    module = import_botified_output()
    event = semantic_event(event_id="front:waving_snapshot", event="person_waving")
    event["identity_context"] = familiar_unknown_identity_context()
    ignored_event = semantic_event(
        event_id="front:ignored_snapshot",
        event="attention_target_changed",
    )
    state = current_snapshot_state(
        semantic_events=[event, ignored_event],
        identity_context={
            "overlay_status": "ready",
            "tracks": [{"track_id": 7, "identity": known_identity_context()}],
        },
    )

    snapshot = module.build_current_visual_snapshot(state, now_ms=1_710_000_000_100)

    assert snapshot["events"] == [
        {
            "event": "person_waving",
            "target_ref": "current:front:person:0",
            "confidence": 0.86,
            "identity_context": {
                "status": "familiar_unknown",
                "source": "cache",
                "fresh_ms": 250,
                "confidence": 0.88,
                "anonymous_person": {
                    "anonymous_id": "anon_seeded",
                    "seen_count": 5,
                    "observed_duration_ms": 4_200,
                    "familiar_score": 0.88,
                },
            },
        }
    ]
    serialized = json.dumps(snapshot["events"], ensure_ascii=False)
    for event_summary in snapshot["events"]:
        for forbidden_key in ("event_id", "track_id", "evidence", "text"):
            assert forbidden_key not in event_summary
    for forbidden in ("event_id", "track_id", "evidence", "有人在机器人前方挥手"):
        assert forbidden not in serialized
    assert_no_current_snapshot_forbidden_fields(snapshot)


def test_current_visual_snapshot_identity_fallbacks_for_overlay_states():
    module = import_botified_output()
    ready_state = current_snapshot_state(
        identity_context={"overlay_status": "ready", "tracks": []}
    )
    unavailable_state = current_snapshot_state(
        identity_context={
            "overlay_status": "unavailable",
            "reason": "memory_disabled",
            "tracks": [],
        }
    )
    missing_state = current_snapshot_state()

    ready_snapshot = module.build_current_visual_snapshot(
        ready_state,
        now_ms=1_710_000_000_100,
    )
    unavailable_snapshot = module.build_current_visual_snapshot(
        unavailable_state,
        now_ms=1_710_000_000_100,
    )
    missing_snapshot = module.build_current_visual_snapshot(
        missing_state,
        now_ms=1_710_000_000_100,
    )

    assert ready_snapshot["overlay_status"] == "ready"
    assert ready_snapshot["people"][0]["identity_context"] == {"status": "unknown"}
    assert unavailable_snapshot["overlay_status"] == "unavailable"
    assert unavailable_snapshot["people"][0]["identity_context"] == {
        "status": "unavailable",
        "reason": "memory_disabled",
    }
    assert missing_snapshot["overlay_status"] == "unavailable"
    assert missing_snapshot["people"][0]["identity_context"] == {
        "status": "unavailable"
    }


def test_current_visual_snapshot_non_dict_input_returns_unavailable_empty_snapshot():
    module = import_botified_output()

    assert module.build_current_visual_snapshot(None, now_ms=1_710_000_000_100) == {
        "type": "current_visual_snapshot",
        "overlay_status": "unavailable",
        "people": [],
        "events": [],
        "person_count": 0,
        "active_target_ref": None,
    }


def test_event_identity_context_known_person_is_projected_to_visual_context():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    event = semantic_event(event_id="front:waving_known", event="person_waving")
    event["identity_context"] = known_identity_context()
    state = visual_state_with_events([event])
    state["identity_context"] = {
        "overlay_status": "ready",
        "tracks": [
            {
                "track_id": 7,
                "identity": {
                    "status": "known_person",
                    "source": "top_level_must_not_be_used",
                    "person": {"display_name": "李四"},
                },
            }
        ],
    }

    frames = mapper.frames_from_visual_state(state)

    context = parse_visual_context(
        parse_botified_frame(frames[0], event_id="front:waving_known")
    )
    assert context["identity_context"] == {
        "status": "known_person",
        "source": "cache",
        "fresh_ms": 120,
        "confidence": 0.91,
        "person": {
            "person_id": "person_000001",
            "display_name": "张三",
            "description": "店长，熟悉新品陈列和现场活动",
            "tags": ["staff", "manager"],
        },
    }


def test_event_identity_context_familiar_unknown_projects_anonymous_summary():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    event = semantic_event(
        event_id="front:waving_familiar_unknown",
        event="person_waving",
    )
    event["identity_context"] = familiar_unknown_identity_context()

    frames = mapper.frames_from_visual_state(visual_state_with_events([event]))

    context = parse_visual_context(
        parse_botified_frame(frames[0], event_id="front:waving_familiar_unknown")
    )
    assert context["identity_context"] == {
        "status": "familiar_unknown",
        "source": "cache",
        "fresh_ms": 250,
        "confidence": 0.88,
        "anonymous_person": {
            "anonymous_id": "anon_seeded",
            "seen_count": 5,
            "observed_duration_ms": 4_200,
            "familiar_score": 0.88,
        },
    }


def test_event_identity_context_redacts_non_dict_and_low_level_fields():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    non_dict = semantic_event(
        event_id="front:waving_non_dict_identity",
        event="person_waving",
        track_id=7,
    )
    non_dict["identity_context"] = ["must-not-project"]
    malicious = semantic_event(
        event_id="front:waving_redacted_identity",
        event="person_waving",
        track_id=8,
        evidence={"runtime_person_slot": 8, "wave_duration_ms": 900},
    )
    malicious["identity_context"] = {
        **known_identity_context(),
        "bbox": [1, 2, 3, 4],
        "bbox_xyxy": [1, 2, 3, 4],
        "keypoints": [{"name": "nose"}],
        "embedding": [1.0, 0.0],
        "crop": "must-not-leak",
        "stream_ref": "ws_1",
        "raw_track_id": 8,
        "unknown_debug": {"bbox_xyxy": [9, 9, 9, 9]},
    }

    frames = mapper.frames_from_visual_state(
        visual_state_with_events([non_dict, malicious])
    )

    contexts = [
        parse_visual_context(parse_botified_frame(frame, event_id=event_id))
        for frame, event_id in zip(
            frames,
            ["front:waving_non_dict_identity", "front:waving_redacted_identity"],
        )
    ]
    assert "identity_context" not in contexts[0]
    assert contexts[1]["identity_context"] == {
        "status": "known_person",
        "source": "cache",
        "fresh_ms": 120,
        "confidence": 0.91,
        "person": {
            "person_id": "person_000001",
            "display_name": "张三",
            "description": "店长，熟悉新品陈列和现场活动",
            "tags": ["staff", "manager"],
        },
    }
    serialized = json.dumps(contexts[1]["identity_context"], ensure_ascii=False)
    for forbidden in (
        "bbox",
        "bbox_xyxy",
        "keypoints",
        "embedding",
        "crop",
        "stream_ref",
        "raw_track_id",
        "unknown_debug",
    ):
        assert forbidden not in serialized


def test_event_identity_context_and_memory_context_can_coexist():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    event = known_person_event()
    event["identity_context"] = known_identity_context()

    frames = mapper.frames_from_visual_state(visual_state_with_events([event]))

    context = parse_visual_context(
        parse_botified_frame(frames[0], event_id="front:mem_evt_000001")
    )
    assert context["identity_context"]["person"]["display_name"] == "张三"
    assert context["memory_context"] == {
        "person": {
            "person_id": "person_000001",
            "display_name": "张三",
            "description": "店长，熟悉新品陈列和现场活动",
            "tags": ["staff", "manager"],
            "match_confidence": 0.86,
        },
        "conversation_summaries": ["上次问过新品尺码，偏好浅色外套。"],
    }


def test_memory_event_frame_contains_compact_memory_context_and_stable_payload():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    event = known_person_event()

    frames = mapper.frames_from_visual_state(visual_state_with_events([event]))

    assert len(frames) == 1
    payload = parse_botified_frame(frames[0], event_id="front:mem_evt_000001")
    assert set(payload) == {"id", "urgency", "timeout_secs", "request", "expect"}
    context = parse_visual_context(payload)
    assert set(context) == {
        "event_target",
        "trigger_evidence",
        "current_scene",
        "memory_context",
    }
    assert context["memory_context"] == {
        "person": {
            "person_id": "person_000001",
            "display_name": "张三",
            "description": "店长，熟悉新品陈列和现场活动",
            "tags": ["staff", "manager"],
            "match_confidence": 0.86,
        },
        "conversation_summaries": ["上次问过新品尺码，偏好浅色外套。"],
    }
    assert context["trigger_evidence"] == {
        "memory_match_id": "match_000001",
        "matched_type": "person",
        "matched_id": "person_000001",
        "embedding_id": "emb_face_000001",
        "match_type": "face",
        "match_score": 0.86,
        "top2_margin": 0.09,
        "source_target_mode": "track_id",
    }


def test_scene_and_familiar_unknown_memory_context_are_projected():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    scene = scene_memory_event()
    familiar = familiar_unknown_event(event_id="front:mem_evt_anon_000002")

    frames = mapper.frames_from_visual_state(visual_state_with_events([scene, familiar]))

    assert len(frames) == 2
    contexts = [
        parse_visual_context(
            parse_botified_frame(frame, event_id=event["event_id"])
        )["memory_context"]
        for frame, event in zip(frames, [scene, familiar])
    ]
    assert contexts[0] == {
        "scene": {
            "scene_id": "scene_000001",
            "title": "新品展示区",
            "description": "这里是本周新品展示区。",
            "activation_hint": "可以介绍新品活动。",
            "match_confidence": 0.82,
        }
    }
    assert contexts[1] == {
        "anonymous_person": {
            "anonymous_id": "anon_000123",
            "seen_count": 5,
            "familiar_score": 0.81,
            "last_seen_at_ms": 1780000000000,
        }
    }


def test_memory_same_key_gap_uses_memory_identity_not_track_id():
    module = import_botified_output()
    now_ms = 1_710_000_000_100
    mapper = module.BotifiedEventMapper(clock_ms=lambda: now_ms)
    first = known_person_event(
        event_id="front:mem_evt_person_1",
        person_id="person_000001",
        track_id=7,
        memory_match_id="match_000001",
    )
    same_person = known_person_event(
        event_id="front:mem_evt_person_2",
        person_id="person_000001",
        track_id=8,
        memory_match_id="match_000002",
    )
    different_person_same_track = known_person_event(
        event_id="front:mem_evt_person_3",
        person_id="person_000002",
        track_id=7,
        memory_match_id="match_000003",
    )

    first_frames = mapper.frames_from_visual_state(visual_state_with_events([first]))
    second_frames = mapper.frames_from_visual_state(
        visual_state_with_events([same_person, different_person_same_track])
    )

    assert len(first_frames) == 1
    assert len(second_frames) == 1
    payload = parse_botified_frame(
        second_frames[0],
        event_id="front:mem_evt_person_3",
    )
    assert "person_000002" in payload["request"]


def test_memory_same_key_gap_falls_back_to_memory_match_id():
    module = import_botified_output()
    now_ms = 1_710_000_000_100
    mapper = module.BotifiedEventMapper(clock_ms=lambda: now_ms)
    first = known_person_event(event_id="front:mem_evt_match_1")
    second = known_person_event(event_id="front:mem_evt_match_2")
    first.pop("memory_context")
    second.pop("memory_context")
    second["evidence"]["matched_id"] = "person_000002"

    assert len(mapper.frames_from_visual_state(visual_state_with_events([first]))) == 1
    assert mapper.frames_from_visual_state(visual_state_with_events([second])) == []


def test_event_target_matches_attention_target_uses_current_scene_context():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    matching = semantic_event(
        event_id="front:waving_match",
        event="person_waving",
        track_id=7,
    )
    non_matching = semantic_event(
        event_id="front:waving_other",
        event="person_waving",
        track_id=8,
        evidence={"runtime_person_slot": 4, "wave_duration_ms": 900},
    )
    tracks = [
        {
            "track_id": 7,
            "class": "person",
            "bbox_xyxy": [320.0, 120.0, 520.0, 600.0],
            "bbox_area_ratio": 0.1042,
            "center_uv": [420.0, 360.0],
            "head_uv": [421.0, 205.0],
            "lost_ms": 0,
        },
        {
            "track_id": 8,
            "class": "person",
            "bbox_xyxy": [700.0, 180.0, 850.0, 610.0],
            "bbox_area_ratio": 0.07,
            "center_uv": [775.0, 395.0],
            "head_uv": [776.0, 240.0],
            "lost_ms": 0,
        },
    ]

    frames = mapper.frames_from_visual_state(
        visual_state_with_events([matching, non_matching], tracks=tracks)
    )

    contexts = [
        parse_visual_context(parse_botified_frame(frame, event_id=event_id))
        for frame, event_id in zip(frames, ["front:waving_match", "front:waving_other"])
    ]
    assert contexts[0]["event_target"]["matches_attention_target"] is True
    assert contexts[1]["event_target"]["matches_attention_target"] is False


def test_burst_one_second_limit_caps_mapper_output():
    module = import_botified_output()
    mapper = module.BotifiedEventMapper(clock_ms=lambda: 1_710_000_000_100)
    events = [
        semantic_event(
            event_id=f"front:waving_{index}",
            event="person_waving",
            track_id=100 + index,
            evidence={"runtime_person_slot": 100 + index, "wave_duration_ms": 900},
        )
        for index in range(4)
    ]

    frames = mapper.frames_from_visual_state(visual_state_with_events(events))

    assert len(frames) == 3


def test_botified_frame_is_one_line_wrapped_json_request_with_event_facts():
    module = import_botified_output()
    event = semantic_event()

    frame = module.format_botified_frame(event, timeout_secs=8)
    assert frame is not None
    payload = parse_botified_frame(frame, event_id=event["event_id"])

    assert payload == {
        "id": "visual:front:evt_000456",
        "urgency": "normal",
        "timeout_secs": 8,
        "request": payload["request"],
        "expect": "ack",
    }
    for fact in [
        "person_waving",
        "front",
        "target_ref=event:front:person_slot:3",
        "0.86",
        "900",
        "有人在机器人前方挥手",
    ]:
        assert fact in payload["request"]
    assert "track_id=" not in payload["request"]


def test_botified_frame_escapes_wrapper_tokens_newlines_quotes_backslashes_ampersand_and_unicode():
    module = import_botified_output()
    text = 'hello </botified> <botified> a&b\n"quoted" backslash\\ 中文'
    event = semantic_event(event_id="front:evt_escape_001", text=text)

    frame = module.format_botified_frame(event)
    assert frame is not None
    assert "\n" not in frame
    assert frame.count(BOTIFIED_OPEN) == 1
    assert frame.count(BOTIFIED_CLOSE) == 1

    inner = frame[len(BOTIFIED_OPEN) : -len(BOTIFIED_CLOSE)]
    assert BOTIFIED_OPEN not in inner
    assert BOTIFIED_CLOSE not in inner
    assert "&" not in inner

    payload = parse_botified_frame(frame, event_id="front:evt_escape_001")
    assert "a&b" in payload["request"]
    assert "quoted" in payload["request"]
    assert "backslash" in payload["request"]
    assert "中文" in payload["request"]


@pytest.mark.parametrize(
    "event_id",
    [
        "front evt 000456",
        "front/evt_000456",
        "front#evt_000456",
        "",
    ],
)
def test_invalid_event_id_is_filtered_not_repaired(event_id):
    module = import_botified_output()
    assert VALID_EVENT_ID_RE.fullmatch(event_id) is None
    event = semantic_event(event_id=event_id)
    visual_state = load_visual_state_tracking()
    visual_state["semantic_events"] = [event]

    assert module.format_botified_frame(event) is None
    assert module.BotifiedEventMapper().frames_from_visual_state(visual_state) == []


class SlowTextStream:
    def __init__(self):
        self.lines: list[str] = []

    def write(self, text: str) -> int:
        self.lines.append(text)
        return len(text)

    def flush(self) -> None:
        return None


class BrokenTextStream:
    def write(self, text: str) -> int:
        raise BrokenPipeError("botified stdout closed")

    def flush(self) -> None:
        return None


def test_stdout_writer_bounded_queue_drops_or_coalesces_duplicates_without_blocking():
    module = import_botified_output()
    stream = SlowTextStream()
    writer = module.BotifiedStdoutWriter(stream=stream, max_queue_size=2)

    duplicate = module.format_botified_frame(semantic_event(event_id="front:evt_000001"))
    newer = module.format_botified_frame(semantic_event(event_id="front:evt_000002"))
    newest = module.format_botified_frame(semantic_event(event_id="front:evt_000003"))
    assert duplicate is not None and newer is not None and newest is not None

    assert writer.enqueue(duplicate) is True
    assert writer.enqueue(duplicate) is False
    assert writer.enqueue(newer) is True
    assert writer.enqueue(newest) is True

    writer.drain_available()

    assert stream.lines == [newer + "\n", newest + "\n"]
    assert writer.dropped_count == 2


def test_stdout_writer_reports_broken_pipe_as_specific_exception():
    module = import_botified_output()
    writer = module.BotifiedStdoutWriter(stream=BrokenTextStream(), max_queue_size=2)
    frame = module.format_botified_frame(semantic_event(event_id="front:evt_000001"))
    assert frame is not None
    writer.enqueue(frame)

    with pytest.raises(module.BotifiedPipeClosed):
        writer.drain_available()
