from __future__ import annotations

from pathlib import Path

from tools.visual_evidence_helpers import (
    _engagement_state,
    _event_summary,
    _evidence_summary,
    _memory_event_summary,
    _reacquire_summary,
    _scene_context_summary,
    frame_header_lines,
    render_frame_card,
    render_html_document,
)
from tools.visualize_service_replay import _progress_line


def _visual_state() -> dict:
    return {
        "frame_id": 12,
        "camera": "front",
        "tracks": [{"track_id": 7, "lost_ms": 0}],
        "attention": {"target_track_id": 7},
        "scene_context": {
            "engagement_state": "no_engage_target",
            "attention_available": False,
            "target_track_id": 7,
            "no_engage_reasons": ["too_far", "camera_motion_not_stationary"],
            "target_reacquired": {
                "runtime_person_slot": 2,
                "reacquired_from_track_id": 4,
                "reacquired_to_track_id": 7,
                "reacquire_elapsed_ms": 850,
                "reacquire_center_distance_px": 13.42,
            },
        },
        "semantic_events": [
            {
                "type": "semantic_event",
                "event": "person_waving",
                "track_id": 7,
                "evidence": {
                    "runtime_person_slot": 2,
                    "wave_duration_ms": 1200,
                    "wrist_x_span_bbox_ratio": 0.42,
                    "keypoint_min_confidence": 0.81,
                    "reacquired_from_track_id": 4,
                    "reacquired_to_track_id": 7,
                    "reacquire_elapsed_ms": 850,
                    "long_debug_text": "x" * 200,
                },
            }
        ],
    }


def _memory_events() -> list[dict]:
    return [
        {
            "type": "semantic_event",
            "event": "known_person_present",
            "track_id": 7,
            "evidence": {
                "memory_match_id": "match_000001",
                "matched_id": "person_000001",
                "match_score": 0.86,
                "top2_margin": 0.09,
                "embedding_id": "emb_face_000001",
                "vector_blob": "must-not-leak",
            },
            "memory_context": {
                "person": {
                    "person_id": "person_000001",
                    "display_name": "张三",
                    "description": "店长，熟悉新品陈列和现场活动",
                },
                "raw_notes": "must-not-leak",
            },
        },
        {
            "type": "semantic_event",
            "event": "scene_activated",
            "track_id": None,
            "evidence": {
                "memory_match_id": "match_scene_000001",
                "matched_id": "scene_000001",
                "match_score": 0.82,
                "top2_margin": 0.07,
            },
            "memory_context": {
                "scene": {
                    "scene_id": "scene_000001",
                    "title": "新品展示区",
                    "activation_hint": "可以介绍新品活动。",
                }
            },
        },
        {
            "type": "semantic_event",
            "event": "familiar_unknown_present",
            "track_id": 9,
            "evidence": {
                "memory_match_id": "match_anon_000001",
                "matched_id": "anon_000123",
                "match_score": 0.81,
                "top2_margin": 0.08,
            },
            "memory_context": {
                "anonymous_person": {
                    "anonymous_id": "anon_000123",
                    "seen_count": 5,
                }
            },
        },
    ]


def _frame_evidence(tmp_path: Path, state: dict, *, scene: str = "fixtures/scene") -> dict:
    return {
        "frame_id": 1,
        "scene": scene,
        "source_name": "frame_001.jpg",
        "image_path": tmp_path / "frames" / "frame_001.jpg",
        "state_path": tmp_path / "states" / "frame_001.json",
        "state": state,
    }


def test_visual_debug_helpers_summarize_memory_events() -> None:
    known_person, scene, familiar_unknown = _memory_events()

    assert _memory_event_summary(known_person) == (
        "matched_id=person_000001 name=张三 match_score=0.86 "
        "top2_margin=0.09 memory_match_id=match_000001"
    )
    assert _event_summary(known_person) == (
        "known_person_present track=7 memory=matched_id=person_000001 name=张三 "
        "match_score=0.86 top2_margin=0.09 memory_match_id=match_000001"
    )

    assert _memory_event_summary(scene) == (
        "matched_id=scene_000001 title=新品展示区 match_score=0.82 "
        "top2_margin=0.07 memory_match_id=match_scene_000001"
    )
    assert _event_summary(scene) == (
        "scene_activated track=- memory=matched_id=scene_000001 title=新品展示区 "
        "match_score=0.82 top2_margin=0.07 memory_match_id=match_scene_000001"
    )

    assert _memory_event_summary(familiar_unknown) == (
        "matched_id=anon_000123 match_score=0.81 top2_margin=0.08 "
        "memory_match_id=match_anon_000001"
    )
    assert "seen_count" not in _event_summary(familiar_unknown)
    assert "vector_blob" not in _event_summary(known_person)


def test_visual_debug_html_includes_memory_event_summary(tmp_path: Path) -> None:
    state = _visual_state()
    state["semantic_events"] = _memory_events()
    frame = _frame_evidence(tmp_path, state)

    rendered = render_html_document(
        root=tmp_path,
        server="ws://127.0.0.1:8765/v1/stream",
        scene=Path("fixtures/scene"),
        frames=[frame],
        jsonl_path=tmp_path / "visual_state.jsonl",
    )

    assert (
        "known_person_present track=7 memory=matched_id=person_000001 name=张三 "
        "match_score=0.86 top2_margin=0.09 memory_match_id=match_000001"
    ) in rendered
    assert (
        "scene_activated track=- memory=matched_id=scene_000001 title=新品展示区 "
        "match_score=0.82 top2_margin=0.07 memory_match_id=match_scene_000001"
    ) in rendered
    assert (
        "familiar_unknown_present track=9 memory=matched_id=anon_000123 "
        "match_score=0.81 top2_margin=0.08 memory_match_id=match_anon_000001"
    ) in rendered


def test_visual_debug_memory_summary_tolerates_missing_fields() -> None:
    assert _memory_event_summary({}) == "-"
    assert _event_summary({"event": "known_person_present"}) == (
        "known_person_present track=- memory=-"
    )
    assert _memory_event_summary(
        {
            "event": "known_person_present",
            "evidence": {"matched_id": "person_1"},
            "memory_context": {"person": {"display_name": "Alex"}},
        }
    ) == "matched_id=person_1 name=Alex"


def test_visual_debug_helpers_summarize_scene_reacquire_and_evidence() -> None:
    state = _visual_state()
    event = state["semantic_events"][0]

    assert (
        _scene_context_summary(state["scene_context"])
        == "engagement=no_engage_target reasons=too_far,camera_motion_not_stationary"
    )
    assert _engagement_state(state["scene_context"]) == "no_engage_target"
    assert _reacquire_summary(state["scene_context"]) == "reacq 4->7 elapsed_ms=850"

    evidence = _evidence_summary(event)
    assert "runtime_person_slot=2" in evidence
    assert "reacq=4->7" in evidence
    assert "reacquire_elapsed_ms=850" in evidence
    assert "wave_duration_ms=1200" in evidence
    assert "long_debug_text" not in evidence

    event_summary = _event_summary(event)
    assert "person_waving track=7" in event_summary
    assert (
        "evidence=runtime_person_slot=2 reacq=4->7 "
        "reacquire_elapsed_ms=850 wave_duration_ms=1200"
    ) in event_summary


def test_visual_debug_html_and_progress_include_short_summaries(tmp_path: Path) -> None:
    state = _visual_state()
    frame = _frame_evidence(tmp_path, state)

    rendered = render_frame_card(tmp_path, frame)

    assert "frame=1 frame_001.jpg" in rendered
    assert "scene=fixtures/scene" in rendered
    assert "scene_context=engagement=no_engage_target reasons=too_far,camera_motion_not_stationary" in rendered
    assert "reacq=reacq 4-&gt;7 elapsed_ms=850" in rendered
    assert (
        "person_waving track=7 evidence=runtime_person_slot=2 reacq=4-&gt;7 "
        "reacquire_elapsed_ms=850 wave_duration_ms=1200"
    ) in rendered
    assert "<details><summary>visual_state</summary>" in rendered

    progress = _progress_line(1, 3, Path("frame_001.jpg"), state)
    assert progress == (
        "[1/3] frame_001.jpg: tracks=1 attention=7 "
        "engagement=no_engage_target events=1"
    )


def test_visual_debug_helpers_tolerate_missing_fields() -> None:
    assert _scene_context_summary(None) == "scene_context=none"
    assert _scene_context_summary({}) == "scene_context=none"
    assert _engagement_state({}) == "-"
    assert _reacquire_summary({}) == "-"
    assert _evidence_summary({"event": "person_waving"}) == "-"
    assert _evidence_summary({"evidence": "not-a-dict"}) == "-"
    assert _event_summary({"event": "person_left"}) == "person_left track=- evidence=-"

    progress = _progress_line(1, 1, Path("frame.jpg"), {"semantic_events": "bad"})
    assert progress == "[1/1] frame.jpg: tracks=0 attention=- engagement=- events=0"


def test_visual_debug_frame_header_lines_include_wrapper_metadata() -> None:
    state = {"camera": "front", "frame_id": 12}

    lines = frame_header_lines(state, scene="lobby", frame_id=99)

    assert "frame=99" in lines
    assert "camera=front" in lines
    assert "scene=lobby" in lines
    assert "scene_context=none" in lines


def test_visual_debug_frame_card_shows_none_for_missing_context_and_events(tmp_path: Path) -> None:
    state = {"camera": "front", "tracks": [], "semantic_events": []}

    rendered = render_frame_card(
        tmp_path,
        _frame_evidence(tmp_path, state, scene="lobby"),
        anchor="frame-1",
    )

    assert 'id="frame-1"' in rendered
    assert "scene=lobby" in rendered
    assert "scene_context=none" in rendered
    assert "events=none" in rendered
