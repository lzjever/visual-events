from __future__ import annotations

import json
import math
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VISUAL_STATE_TRACKING_SAMPLE = (
    REPO_ROOT / "common" / "schema" / "samples" / "visual_state_tracking.json"
)
DDS_SAMPLE_DIR = REPO_ROOT / "common" / "schema" / "dds" / "samples"

GAZE_TARGET_FIELDS = (
    "schema_version",
    "camera",
    "frame_id",
    "frame_timestamp_ms",
    "publish_timestamp_ms",
    "valid",
    "state",
    "target_track_id",
    "target_u",
    "target_v",
    "target_norm_x",
    "target_norm_y",
    "image_width",
    "image_height",
    "confidence",
    "reason",
    "stale_after_ms",
)


def import_target_mapper():
    try:
        import visual_events_cli.target_mapper as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.target_mapper module: {exc}")
    return module


def load_visual_state_tracking() -> dict[str, Any]:
    return json.loads(VISUAL_STATE_TRACKING_SAMPLE.read_text(encoding="utf-8"))


def payload_dict(module: Any, payload: Any) -> dict[str, Any]:
    payload_type = getattr(module, "GazeTargetPayload", None)
    assert payload_type is not None
    assert is_dataclass(payload_type)
    assert isinstance(payload, payload_type)
    assert callable(getattr(payload, "to_dict", None))

    data = payload.to_dict()
    assert isinstance(data, dict)
    assert tuple(data) == GAZE_TARGET_FIELDS
    return data


def assert_lost_zero_target(data: dict[str, Any]) -> None:
    assert data["valid"] is False
    assert data["state"] == "lost"
    assert data["target_track_id"] == -1
    assert data["target_u"] == 0.0
    assert data["target_v"] == 0.0
    assert data["target_norm_x"] == 0.0
    assert data["target_norm_y"] == 0.0
    assert data["confidence"] == 0.0
    assert data["reason"] == "lost"


def test_maps_sample_visual_state_tracking_to_gaze_target_payload():
    module = import_target_mapper()
    visual_state = load_visual_state_tracking()
    publish_timestamp_ms = visual_state["frame_timestamp_ms"] + 82

    data = payload_dict(
        module,
        module.map_visual_state_to_gaze_target(
            visual_state,
            publish_timestamp_ms=publish_timestamp_ms,
            stale_after_ms=250,
        ),
    )

    assert data == {
        "schema_version": 1,
        "camera": "front",
        "frame_id": 1024,
        "frame_timestamp_ms": 1710000000000,
        "publish_timestamp_ms": publish_timestamp_ms,
        "valid": True,
        "state": "tracking",
        "target_track_id": 7,
        "target_u": 421.0,
        "target_v": 205.0,
        "target_norm_x": pytest.approx(421.0 / 1280.0 - 0.5),
        "target_norm_y": pytest.approx(205.0 / 720.0 - 0.5),
        "image_width": 1280,
        "image_height": 720,
        "confidence": 0.86,
        "reason": "largest_stable_person",
        "stale_after_ms": 250,
    }


def test_fresh_visual_state_without_attention_maps_to_lost_invalid_payload():
    module = import_target_mapper()
    visual_state = load_visual_state_tracking()
    visual_state["attention"] = None

    data = payload_dict(
        module,
        module.map_visual_state_to_gaze_target(
            visual_state,
            publish_timestamp_ms=visual_state["frame_timestamp_ms"] + 10,
        ),
    )

    assert data["camera"] == "front"
    assert data["frame_id"] == 1024
    assert data["frame_timestamp_ms"] == 1710000000000
    assert data["publish_timestamp_ms"] == 1710000000010
    assert data["image_width"] == 1280
    assert data["image_height"] == 720
    assert data["stale_after_ms"] == 250
    assert_lost_zero_target(data)


def test_old_frame_timestamp_does_not_make_mapper_emit_stale():
    module = import_target_mapper()
    visual_state = load_visual_state_tracking()
    publish_timestamp_ms = visual_state["frame_timestamp_ms"] + 10_000

    data = payload_dict(
        module,
        module.map_visual_state_to_gaze_target(
            visual_state,
            publish_timestamp_ms=publish_timestamp_ms,
            stale_after_ms=250,
        ),
    )

    assert data["valid"] is True
    assert data["state"] == "tracking"
    assert data["publish_timestamp_ms"] == publish_timestamp_ms


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda state: state["attention"].update({"target_track_id": 999}),
            id="target-track-id-not-in-tracks",
        ),
        pytest.param(
            lambda state: state["attention"].update({"target_uv": [-0.01, 205.0]}),
            id="negative-target-u",
        ),
        pytest.param(
            lambda state: state["attention"].update({"target_uv": [421.0, 720.01]}),
            id="target-v-beyond-height",
        ),
        pytest.param(
            lambda state: state["attention"].update({"target_uv": [math.nan, 205.0]}),
            id="nan-target-u",
        ),
        pytest.param(
            lambda state: state["attention"].update({"target_uv": [421.0, math.inf]}),
            id="inf-target-v",
        ),
        pytest.param(
            lambda state: state.update({"image_size": [0, 720]}),
            id="zero-image-width",
        ),
        pytest.param(
            lambda state: state.update({"image_size": [1280]}),
            id="malformed-image-size",
        ),
    ],
)
def test_invalid_attention_or_image_size_maps_to_lost_invalid_payload(mutation):
    module = import_target_mapper()
    visual_state = load_visual_state_tracking()
    mutation(visual_state)

    data = payload_dict(
        module,
        module.map_visual_state_to_gaze_target(
            visual_state,
            publish_timestamp_ms=visual_state["frame_timestamp_ms"] + 10,
        ),
    )

    assert_lost_zero_target(data)


@pytest.mark.parametrize(
    "target_uv, expected_norm",
    [
        ((0.0, 0.0), (-0.5, -0.5)),
        ((1280.0, 720.0), (0.5, 0.5)),
    ],
)
def test_target_uv_normalization_includes_image_boundaries(target_uv, expected_norm):
    module = import_target_mapper()
    visual_state = load_visual_state_tracking()
    visual_state["attention"]["target_uv"] = [target_uv[0], target_uv[1]]

    data = payload_dict(
        module,
        module.map_visual_state_to_gaze_target(
            visual_state,
            publish_timestamp_ms=visual_state["frame_timestamp_ms"] + 10,
        ),
    )

    assert data["valid"] is True
    assert data["target_u"] == target_uv[0]
    assert data["target_v"] == target_uv[1]
    assert data["target_norm_x"] == pytest.approx(expected_norm[0])
    assert data["target_norm_y"] == pytest.approx(expected_norm[1])


def test_disabled_mapper_state_takes_priority_over_legal_attention():
    module = import_target_mapper()
    visual_state = load_visual_state_tracking()

    data = payload_dict(
        module,
        module.map_visual_state_to_gaze_target(
            visual_state,
            publish_timestamp_ms=visual_state["frame_timestamp_ms"] + 10,
            enabled=False,
        ),
    )

    assert data["valid"] is False
    assert data["state"] == "disabled"
    assert data["target_track_id"] == -1
    assert data["target_u"] == 0.0
    assert data["target_v"] == 0.0
    assert data["target_norm_x"] == 0.0
    assert data["target_norm_y"] == 0.0
    assert data["confidence"] == 0.0
    assert data["reason"] == "disabled"


@pytest.mark.parametrize(
    "state,sample_name",
    [
        ("stale", "gaze_target_stale.json"),
        ("disabled", "gaze_target_disabled.json"),
    ],
)
def test_make_invalid_gaze_target_matches_dds_sample_zero_semantics(state, sample_name):
    module = import_target_mapper()
    expected = json.loads((DDS_SAMPLE_DIR / sample_name).read_text(encoding="utf-8"))

    data = payload_dict(
        module,
        module.make_invalid_gaze_target(
            state,
            camera=expected["camera"],
            frame_id=expected["frame_id"],
            frame_timestamp_ms=expected["frame_timestamp_ms"],
            image_size=(expected["image_width"], expected["image_height"]),
            publish_timestamp_ms=expected["publish_timestamp_ms"],
            stale_after_ms=expected["stale_after_ms"],
        ),
    )

    assert data == expected


def test_stale_gaze_target_is_only_explicit_invalid_state():
    module = import_target_mapper()
    visual_state = load_visual_state_tracking()

    data = payload_dict(
        module,
        module.make_invalid_gaze_target(
            "stale",
            camera=visual_state["camera"],
            frame_id=visual_state["frame_id"],
            frame_timestamp_ms=visual_state["frame_timestamp_ms"],
            image_size=tuple(visual_state["image_size"]),
            publish_timestamp_ms=visual_state["frame_timestamp_ms"] + 10_000,
            stale_after_ms=250,
        ),
    )

    assert data["valid"] is False
    assert data["state"] == "stale"
    assert data["reason"] == "stale"


def test_make_invalid_gaze_target_rejects_unknown_state():
    module = import_target_mapper()

    try:
        module.make_invalid_gaze_target(
            "sleeping",
            camera="front",
            publish_timestamp_ms=1710000000140,
        )
    except ValueError:
        return
    except Exception as exc:
        assert exc.__class__.__module__ == module.__name__
        return

    pytest.fail("make_invalid_gaze_target must reject unknown invalid states")
