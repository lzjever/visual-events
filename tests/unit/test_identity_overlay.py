from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from visual_events_server.memory.events import MemoryMatch
from visual_events_server.memory.identity_overlay import (
    IdentityOverlay,
    unavailable_identity_context,
)


def person_track(track_id: int = 7) -> dict:
    return {
        "track_id": track_id,
        "class": "person",
        "bbox_xyxy": [300.0, 100.0, 700.0, 650.0],
        "keypoints": [{"name": "nose", "x": 500.0, "y": 150.0}],
        "embedding": [1.0, 0.0],
        "crop_ref": "runtime/private/crop.jpg",
        "lost_ms": 0,
    }


def visual_state(*, track_id: int = 7) -> dict:
    return {
        "camera": "front",
        "tracks": [person_track(track_id)],
        "attention": {"target_track_id": track_id},
    }


def match(
    *,
    matched_type: str = "person",
    matched_id: str = "person_1",
    score: float = 0.91,
) -> MemoryMatch:
    return MemoryMatch(
        memory_match_id="match_1",
        matched_type=matched_type,
        matched_id=matched_id,
        embedding_id="emb_1",
        match_type="face",
        match_score=score,
        top2_margin=0.2,
        embedding_model="fake-face",
        embedding_version="v1",
    )


def test_known_projection_is_stream_scoped_redacted_and_expires() -> None:
    now = 1_000
    overlay = IdentityOverlay(ttl_ms=500, clock_ms=lambda: now)
    overlay.put_known_person(
        connection_id="ws_a",
        camera="front",
        track_id=7,
        person_profile={
            "person_id": "person_1",
            "display_name": "张三",
            "description": "店长",
            "tags": ["staff"],
        },
        match=match(),
    )

    projected = overlay.project(
        connection_id="ws_a",
        camera="front",
        visual_state=visual_state(),
    )
    isolated = overlay.project(
        connection_id="ws_b",
        camera="front",
        visual_state=visual_state(),
    )

    assert projected["overlay_status"] == "ready"
    assert projected["active_target"] == {"track_id": 7}
    assert projected["tracks"] == [
        {
            "track_id": 7,
            "identity": {
                "status": "known_person",
                "source": "background_recall",
                "fresh_ms": 0,
                "confidence": 0.91,
                "person": {
                    "person_id": "person_1",
                    "display_name": "张三",
                    "description": "店长",
                    "tags": ["staff"],
                },
            },
        }
    ]
    assert isolated["tracks"][0]["identity"]["status"] == "pending"
    assert "bbox_xyxy" not in projected["tracks"][0]
    assert "keypoints" not in projected["tracks"][0]
    assert "embedding" not in str(projected)
    assert "crop" not in str(projected)

    now = 1_501

    expired = overlay.project(
        connection_id="ws_a",
        camera="front",
        visual_state=visual_state(),
    )
    assert expired["tracks"][0]["identity"]["status"] == "pending"
    overlay.purge()
    assert overlay.active_count == 0


def test_familiar_unknown_unknown_and_unavailable_public_shapes() -> None:
    overlay = IdentityOverlay(ttl_ms=1_000, clock_ms=lambda: 2_000)
    overlay.put_familiar_unknown(
        connection_id="ws_a",
        camera="front",
        track_id=7,
        anonymous_profile={
            "anonymous_id": "anon_1",
            "seen_count": 4,
            "observed_duration_ms": 3_500,
            "familiar_score": 0.88,
        },
        match=match(
            matched_type="anonymous_person",
            matched_id="anon_1",
            score=0.88,
        ),
    )
    overlay.put_unknown(
        connection_id="ws_a",
        camera="front",
        track_id=8,
        reason="not_familiar",
    )
    overlay.put_unavailable(
        connection_id="ws_a",
        camera="front",
        track_id=10,
        reason="no_usable_face",
    )
    state = {
        "camera": "front",
        "tracks": [
            person_track(7),
            person_track(8),
            {**person_track(9), "class": "bag"},
            person_track(10),
        ],
    }

    projected = overlay.project(
        connection_id="ws_a",
        camera="front",
        visual_state=state,
    )

    assert projected["tracks"] == [
        {
            "track_id": 7,
            "identity": {
                "status": "familiar_unknown",
                "source": "background_recall",
                "fresh_ms": 0,
                "confidence": 0.88,
                "anonymous_person": {
                    "anonymous_id": "anon_1",
                    "seen_count": 4,
                    "observed_duration_ms": 3_500,
                    "familiar_score": 0.88,
                },
            },
        },
        {
            "track_id": 8,
            "identity": {
                "status": "unknown",
                "source": "background_recall",
                "fresh_ms": 0,
                "reason": "not_familiar",
            },
        },
        {
            "track_id": 10,
            "identity": {
                "status": "unavailable",
                "source": "background_recall",
                "fresh_ms": 0,
                "reason": "no_usable_face",
            },
        },
    ]
    assert unavailable_identity_context("memory_disabled") == {
        "overlay_status": "unavailable",
        "reason": "memory_disabled",
        "tracks": [],
    }


def test_rejects_invalid_overlay_inputs() -> None:
    with pytest.raises(ValueError, match="ttl_ms"):
        IdentityOverlay(ttl_ms=0, clock_ms=lambda: 0)


def test_overlay_allows_concurrent_put_and_project_without_runtime_errors() -> None:
    now = 1_000
    overlay = IdentityOverlay(ttl_ms=1_000, clock_ms=lambda: now)

    def put_loop() -> None:
        for index in range(200):
            overlay.put_known_person(
                connection_id="ws_a",
                camera="front",
                track_id=7,
                person_profile={
                    "person_id": f"person_{index}",
                    "display_name": "张三",
                    "description": "",
                    "tags": [],
                },
                match=match(),
            )

    def project_loop() -> None:
        for _ in range(200):
            context = overlay.project(
                connection_id="ws_a",
                camera="front",
                visual_state=visual_state(),
            )
            assert context["overlay_status"] == "ready"

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(put_loop),
            executor.submit(put_loop),
            executor.submit(project_loop),
            executor.submit(project_loop),
        ]
        for future in futures:
            future.result()
