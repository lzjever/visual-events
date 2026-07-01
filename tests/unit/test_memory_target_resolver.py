from __future__ import annotations

import pytest

from visual_events_server.attention import AttentionResult
from visual_events_server.memory.target_resolver import (
    TargetCandidate,
    TargetPreview,
    TargetRequest,
    TargetResolveError,
    TargetResolver,
)
from visual_events_server.inference.base import PoseKeypoint
from visual_events_server.tracking import TrackSnapshot


def track(
    track_id: int,
    *,
    bbox_xyxy: tuple[float, float, float, float],
    hits: int = 2,
    lost_ms: int = 0,
    keypoints: tuple[PoseKeypoint, ...] = (),
) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=track_id,
        first_seen_ms=700,
        last_seen_ms=1000 - lost_ms,
        frame_timestamp_ms=1000,
        bbox_xyxy=bbox_xyxy,
        confidence=0.9,
        pose_confidence=0.8,
        head_uv=((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, bbox_xyxy[1] + 40.0),
        velocity_uv_s=(0.0, 0.0),
        lost_ms=lost_ms,
        hits=hits,
        misses=0,
        keypoints=keypoints,
    )


def attention(track_id: int) -> AttentionResult:
    return AttentionResult(
        target_track_id=track_id,
        target_uv=(300.0, 180.0),
        reason="largest_stable_person",
        confidence=0.91,
        largest_person_stable=True,
    )


def kp(name: str, x: float, y: float, confidence: float = 0.9) -> PoseKeypoint:
    return PoseKeypoint(name=name, x=x, y=y, confidence=confidence)


def pointing_right_keypoints(*, confidence: float = 0.9) -> tuple[PoseKeypoint, ...]:
    return (
        kp("left_shoulder", 180.0, 220.0, confidence),
        kp("left_elbow", 300.0, 250.0, confidence),
        kp("left_wrist", 380.0, 260.0, confidence),
    )


def right_arm_pointing_right_keypoints(
    *,
    confidence: float = 0.9,
) -> tuple[PoseKeypoint, ...]:
    return (
        kp("right_shoulder", 180.0, 220.0, confidence),
        kp("right_elbow", 300.0, 250.0, confidence),
        kp("right_wrist", 380.0, 260.0, confidence),
    )


def test_resolves_attention_target_to_matching_track_bbox() -> None:
    resolved = TargetResolver().resolve(
        TargetRequest(mode="attention_target"),
        image_width=640,
        image_height=480,
        tracks=[track(7, bbox_xyxy=(100.0, 80.0, 380.0, 420.0))],
        attention=attention(7),
    )

    assert resolved.source_target_mode == "attention_target"
    assert resolved.target_type == "person"
    assert resolved.track_id == 7
    assert resolved.bbox_xyxy == (100.0, 80.0, 380.0, 420.0)


def test_pose_pointing_resolves_third_person_candidate() -> None:
    preview = TargetResolver().preview_pose_pointing_person(
        introducer_track_id=1,
        image_width=800,
        image_height=600,
        tracks=[
            track(
                1,
                bbox_xyxy=(100.0, 100.0, 300.0, 500.0),
                keypoints=pointing_right_keypoints(),
            ),
            track(2, bbox_xyxy=(500.0, 170.0, 650.0, 420.0)),
            track(3, bbox_xyxy=(60.0, 150.0, 180.0, 420.0)),
        ],
    )

    assert preview.status == "resolved"
    assert preview.ambiguity_type == ""
    assert [candidate.track_id for candidate in preview.candidates] == [2]
    assert preview.candidates[0].reason == "pose_pointing_to_person"
    scoring = preview.candidates[0].evidence["pose_pointing_scoring"]
    assert scoring["arm_side"] == "left"
    assert scoring["keypoint_confidences"] == {
        "left_shoulder": 0.9,
        "left_elbow": 0.9,
        "left_wrist": 0.9,
    }
    assert scoring["arm_vector"] == [200.0, 40.0]
    assert scoring["ambiguous_score_margin"] == 0.08
    assert scoring["checks"]["keypoints_ok"] is True
    assert scoring["checks"]["margin_ok"] is True
    candidate_scores = scoring["candidate_scores"]
    assert candidate_scores[0]["track_id"] == 2
    assert candidate_scores[0]["bbox_xyxy"] == [500.0, 170.0, 650.0, 420.0]
    assert {
        "score",
        "arm_side",
        "bbox_xyxy",
        "perpendicular_distance",
        "ray_intersects_bbox",
    } <= set(candidate_scores[0])
    assert preview.evidence["pose_pointing_scoring"] == scoring
    visual = preview.evidence["pose_visual_evidence"]
    assert visual["coordinate_space"] == "source_frame"
    assert visual["introducer_track_id"] == 1
    assert visual["introducer_bbox_xyxy"] == [100.0, 100.0, 300.0, 500.0]
    assert visual["target_track_id"] == 2
    assert visual["target_bbox_xyxy"] == [500.0, 170.0, 650.0, 420.0]
    assert visual["arm_side"] == "left"
    assert visual["shoulder_xy"] == [180.0, 220.0]
    assert visual["elbow_xy"] == [300.0, 250.0]
    assert visual["wrist_xy"] == [380.0, 260.0]
    assert visual["ray_start_xy"] == [380.0, 260.0]
    assert visual["ray_end_xy"][0] > visual["ray_start_xy"][0]
    assert visual["candidate_scores"][0]["track_id"] == 2
    assert visual["candidate_scores"][0]["bbox_xyxy"] == [
        500.0,
        170.0,
        650.0,
        420.0,
    ]
    assert preview.candidates[0].evidence["pose_visual_evidence"] == visual


def test_pose_pointing_missing_keypoints_is_pose_unclear() -> None:
    preview = TargetResolver().preview_pose_pointing_person(
        introducer_track_id=1,
        image_width=800,
        image_height=600,
        tracks=[
            track(1, bbox_xyxy=(100.0, 100.0, 300.0, 500.0)),
            track(2, bbox_xyxy=(500.0, 170.0, 650.0, 420.0)),
        ],
    )

    assert preview.status == "ambiguous"
    assert preview.ambiguity_type == "pose_unclear"
    assert preview.candidates == []
    visual = preview.evidence["pose_visual_evidence"]
    assert visual["coordinate_space"] == "source_frame"
    assert visual["introducer_track_id"] == 1
    assert visual["introducer_bbox_xyxy"] == [100.0, 100.0, 300.0, 500.0]
    assert visual["ambiguity_type"] == "pose_unclear"
    assert visual["candidate_scores"] == []
    assert "target_track_id" not in visual
    assert "target_bbox_xyxy" not in visual


def test_pose_pointing_close_candidates_are_ambiguous() -> None:
    preview = TargetResolver().preview_pose_pointing_person(
        introducer_track_id=1,
        image_width=900,
        image_height=600,
        tracks=[
            track(
                1,
                bbox_xyxy=(100.0, 100.0, 300.0, 500.0),
                keypoints=pointing_right_keypoints(),
            ),
            track(2, bbox_xyxy=(500.0, 170.0, 620.0, 410.0)),
            track(3, bbox_xyxy=(500.0, 190.0, 620.0, 430.0)),
        ],
    )

    assert preview.status == "ambiguous"
    assert preview.ambiguity_type == "multiple_candidates"
    assert [candidate.reason for candidate in preview.candidates[:2]] == [
        "pose_pointing_to_person",
        "pose_pointing_to_person",
    ]
    scoring = preview.evidence["pose_pointing_scoring"]
    assert scoring["checks"]["keypoints_ok"] is True
    assert scoring["checks"]["margin_ok"] is False
    assert scoring["score_margin"] <= scoring["ambiguous_score_margin"]
    assert {item["track_id"] for item in scoring["candidate_scores"][:2]} == {2, 3}
    assert all(
        {
            "track_id",
            "score",
            "arm_side",
            "bbox_xyxy",
            "perpendicular_distance",
            "ray_intersects_bbox",
        }
        <= set(item)
        for item in scoring["candidate_scores"][:2]
    )
    visual = preview.evidence["pose_visual_evidence"]
    assert visual["introducer_track_id"] == 1
    assert visual["introducer_bbox_xyxy"] == [100.0, 100.0, 300.0, 500.0]
    assert visual["arm_side"] == "left"
    assert visual["ambiguity_type"] == "multiple_candidates"
    assert "target_track_id" not in visual
    assert "target_bbox_xyxy" not in visual
    assert {item["track_id"] for item in visual["candidate_scores"][:2]} == {2, 3}
    assert all("bbox_xyxy" in item for item in visual["candidate_scores"][:2])


def test_pose_pointing_multiple_ray_hits_resolves_when_top_score_leads() -> None:
    preview = TargetResolver().preview_pose_pointing_person(
        introducer_track_id=1,
        image_width=900,
        image_height=600,
        tracks=[
            track(
                1,
                bbox_xyxy=(100.0, 100.0, 300.0, 500.0),
                keypoints=pointing_right_keypoints(),
            ),
            track(2, bbox_xyxy=(500.0, 170.0, 650.0, 420.0)),
            track(3, bbox_xyxy=(820.0, 280.0, 900.0, 480.0)),
        ],
    )

    assert preview.status == "resolved"
    assert preview.ambiguity_type == ""
    assert [candidate.track_id for candidate in preview.candidates] == [2]
    scoring = preview.evidence["pose_pointing_scoring"]
    assert scoring["checks"]["multiple_ray_hits"] is True
    candidate_scores = scoring["candidate_scores"]
    assert [item["track_id"] for item in candidate_scores[:2]] == [2, 3]
    assert all(item["ray_intersects_bbox"] for item in candidate_scores[:2])
    assert (
        candidate_scores[0]["score"] - candidate_scores[1]["score"]
        > scoring["ambiguous_score_margin"]
    )
    visual = preview.evidence["pose_visual_evidence"]
    assert visual["target_track_id"] == 2


def test_pose_pointing_same_ray_near_hit_beats_far_hit() -> None:
    preview = TargetResolver().preview_pose_pointing_person(
        introducer_track_id=1,
        image_width=1400,
        image_height=700,
        tracks=[
            track(
                1,
                bbox_xyxy=(100.0, 100.0, 300.0, 500.0),
                keypoints=pointing_right_keypoints(),
            ),
            track(2, bbox_xyxy=(405.0, 220.0, 455.0, 320.0)),
            track(3, bbox_xyxy=(1080.0, 340.0, 1180.0, 540.0)),
        ],
    )

    assert preview.status == "resolved"
    assert preview.ambiguity_type == ""
    assert [candidate.track_id for candidate in preview.candidates] == [2]
    scoring = preview.evidence["pose_pointing_scoring"]
    assert scoring["checks"]["multiple_ray_hits"] is True
    assert scoring["score_margin"] > scoring["ambiguous_score_margin"]


def test_pose_pointing_near_off_ray_center_beats_far_ray_hit() -> None:
    preview = TargetResolver().preview_pose_pointing_person(
        introducer_track_id=1,
        image_width=1400,
        image_height=700,
        tracks=[
            track(
                1,
                bbox_xyxy=(100.0, 100.0, 300.0, 500.0),
                keypoints=pointing_right_keypoints(),
            ),
            track(2, bbox_xyxy=(500.0, 100.0, 650.0, 300.0)),
            track(3, bbox_xyxy=(820.0, 280.0, 900.0, 480.0)),
        ],
    )

    assert preview.status == "resolved"
    assert preview.ambiguity_type == ""
    assert [candidate.track_id for candidate in preview.candidates] == [2]
    scoring = preview.evidence["pose_pointing_scoring"]
    assert scoring["checks"]["multiple_ray_hits"] is True
    candidate_scores = scoring["candidate_scores"]
    assert [item["track_id"] for item in candidate_scores[:2]] == [2, 3]
    assert all(item["ray_intersects_bbox"] for item in candidate_scores[:2])
    assert candidate_scores[0]["perpendicular_distance"] > 60.0
    assert (
        candidate_scores[0]["score"] - candidate_scores[1]["score"]
        > scoring["ambiguous_score_margin"]
    )


def test_pose_pointing_without_forward_hit_is_target_unclear() -> None:
    preview = TargetResolver().preview_pose_pointing_person(
        introducer_track_id=1,
        image_width=800,
        image_height=600,
        tracks=[
            track(
                1,
                bbox_xyxy=(100.0, 100.0, 300.0, 500.0),
                keypoints=pointing_right_keypoints(),
            ),
            track(2, bbox_xyxy=(520.0, 450.0, 660.0, 580.0)),
        ],
    )

    assert preview.status == "not_found"
    assert preview.ambiguity_type == "target_unclear"


def test_pose_pointing_ray_hit_with_off_ray_target_center_resolves() -> None:
    preview = TargetResolver().preview_pose_pointing_person(
        introducer_track_id=1,
        image_width=900,
        image_height=600,
        tracks=[
            track(
                1,
                bbox_xyxy=(100.0, 100.0, 300.0, 500.0),
                keypoints=right_arm_pointing_right_keypoints(),
            ),
            track(2, bbox_xyxy=(500.0, 100.0, 650.0, 300.0)),
        ],
    )

    assert preview.status == "resolved"
    assert preview.ambiguity_type == ""
    assert [candidate.track_id for candidate in preview.candidates] == [2]
    scoring = preview.evidence["pose_pointing_scoring"]
    assert scoring["arm_side"] == "right"
    assert "multiple_ray_hits" not in scoring["checks"]
    assert scoring["candidate_scores"][0]["score"] > 0.0
    assert scoring["candidate_scores"][0]["ray_intersects_bbox"] is True
    assert scoring["candidate_scores"][0]["perpendicular_distance"] > 60.0


def test_resolves_point_to_visible_track_containing_point() -> None:
    resolved = TargetResolver().resolve(
        TargetRequest(mode="point_uv", point_uv=(320.0, 220.0)),
        image_width=640,
        image_height=480,
        tracks=[
            track(1, bbox_xyxy=(50.0, 50.0, 190.0, 460.0)),
            track(2, bbox_xyxy=(250.0, 120.0, 390.0, 360.0)),
        ],
        attention=None,
    )

    assert resolved.source_target_mode == "point_uv"
    assert resolved.target_type == "person"
    assert resolved.track_id == 2
    assert resolved.bbox_xyxy == (250.0, 120.0, 390.0, 360.0)


def test_previews_overlapping_point_hits_as_ambiguous_person_candidates() -> None:
    preview = TargetResolver().preview(
        TargetRequest(mode="point_uv", point_uv=(320.0, 220.0)),
        image_width=640,
        image_height=480,
        tracks=[
            track(1, bbox_xyxy=(50.0, 50.0, 590.0, 460.0)),
            track(2, bbox_xyxy=(250.0, 120.0, 390.0, 360.0)),
        ],
        attention=None,
    )

    assert isinstance(preview, TargetPreview)
    assert preview.status == "ambiguous"
    assert [candidate.track_id for candidate in preview.candidates] == [2, 1]
    assert all(isinstance(candidate, TargetCandidate) for candidate in preview.candidates)
    assert preview.candidates[0].target_type == "person"
    assert preview.candidates[0].bbox_xyxy == (250.0, 120.0, 390.0, 360.0)
    assert preview.candidates[0].confidence > preview.candidates[1].confidence
    assert preview.candidates[0].reason == "point_inside_bbox"


def test_previews_single_point_bbox_hit_as_resolved() -> None:
    preview = TargetResolver().resolve_candidates(
        TargetRequest(mode="point_uv", point_uv=(320.0, 220.0)),
        image_width=640,
        image_height=480,
        tracks=[
            track(1, bbox_xyxy=(50.0, 50.0, 190.0, 460.0)),
            track(2, bbox_xyxy=(250.0, 120.0, 390.0, 360.0)),
        ],
        attention=None,
    )

    assert preview.status == "resolved"
    assert [candidate.track_id for candidate in preview.candidates] == [2]
    assert preview.candidates[0].reason == "point_inside_bbox"


def test_previews_point_between_two_near_people_as_ambiguous() -> None:
    preview = TargetResolver().preview(
        TargetRequest(mode="point_uv", point_uv=(300.0, 180.0)),
        image_width=640,
        image_height=480,
        tracks=[
            track(7, bbox_xyxy=(160.0, 100.0, 260.0, 360.0)),
            track(9, bbox_xyxy=(340.0, 100.0, 440.0, 360.0)),
        ],
        attention=None,
    )

    assert preview.status == "ambiguous"
    assert {candidate.track_id for candidate in preview.candidates} == {7, 9}
    assert {candidate.reason for candidate in preview.candidates} == {"nearest_bbox_to_point"}


def test_resolve_rejects_ambiguous_point_instead_of_selecting_track() -> None:
    with pytest.raises(TargetResolveError) as exc:
        TargetResolver().resolve(
            TargetRequest(mode="point_uv", point_uv=(300.0, 180.0)),
            image_width=640,
            image_height=480,
            tracks=[
                track(7, bbox_xyxy=(160.0, 100.0, 260.0, 360.0)),
                track(9, bbox_xyxy=(340.0, 100.0, 440.0, 360.0)),
            ],
            attention=None,
        )

    assert exc.value.code == "target_ambiguous"


def test_attention_prior_does_not_override_explicit_point_hit() -> None:
    preview = TargetResolver().preview(
        TargetRequest(mode="point_uv", point_uv=(120.0, 180.0)),
        image_width=640,
        image_height=480,
        tracks=[
            track(1, bbox_xyxy=(80.0, 90.0, 190.0, 360.0)),
            track(2, bbox_xyxy=(300.0, 90.0, 430.0, 360.0)),
        ],
        attention=attention(2),
    )

    assert preview.status == "resolved"
    assert preview.candidates[0].track_id == 1
    assert preview.candidates[0].reason == "point_inside_bbox"


def test_preview_point_without_visible_people_returns_not_found_region_candidate() -> None:
    preview = TargetResolver().preview(
        TargetRequest(mode="point_uv", point_uv=(320.0, 220.0)),
        image_width=640,
        image_height=480,
        tracks=[track(3, bbox_xyxy=(100.0, 80.0, 380.0, 420.0), lost_ms=1)],
        attention=None,
    )

    assert preview.status == "not_found"
    assert len(preview.candidates) == 1
    assert preview.candidates[0].target_type == "region"
    assert preview.candidates[0].track_id is None
    assert preview.candidates[0].reason == "point_region_fallback"


def test_resolves_scene_to_full_image_region() -> None:
    resolved = TargetResolver().resolve(
        TargetRequest(mode="scene"),
        image_width=640,
        image_height=480,
        tracks=[],
        attention=None,
    )

    assert resolved.source_target_mode == "scene"
    assert resolved.target_type == "scene"
    assert resolved.track_id is None
    assert resolved.bbox_xyxy == (0.0, 0.0, 640.0, 480.0)


def test_rejects_missing_or_lost_track_without_returning_guess() -> None:
    with pytest.raises(TargetResolveError) as exc:
        TargetResolver().resolve(
            TargetRequest(mode="track_id", track_id=3),
            image_width=640,
            image_height=480,
            tracks=[track(3, bbox_xyxy=(100.0, 80.0, 380.0, 420.0), lost_ms=1)],
            attention=None,
        )

    assert exc.value.code == "target_not_visible"


def test_rejects_bbox_that_is_too_small_after_clipping() -> None:
    with pytest.raises(TargetResolveError) as exc:
        TargetResolver(min_box_size_px=16.0).resolve(
            TargetRequest(mode="bbox", bbox_xyxy=(-10.0, -10.0, 8.0, 8.0)),
            image_width=640,
            image_height=480,
            tracks=[],
            attention=None,
        )

    assert exc.value.code == "target_too_small"
