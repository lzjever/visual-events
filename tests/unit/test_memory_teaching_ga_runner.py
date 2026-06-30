from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from visual_events_server.memory.api_contract import (
    ResolveTargetRequest,
    TeachPersonRequest,
    TeachSceneRequest,
)
from visual_events_server.protocol import FrameMessage

from tools import run_memory_e2e as memory_e2e
from tools import run_memory_teaching_ga_e2e as module


def _make_scene(
    data_dir: Path,
    name: str,
    *,
    frames: int = 1,
    des_text: str | None = None,
    jpeg_bytes: bytes = b"jpeg",
) -> Path:
    scene_dir = data_dir / name
    scene_dir.mkdir(parents=True)
    for index in range(frames):
        (scene_dir / f"img_{index:03d}.jpeg").write_bytes(jpeg_bytes)
    if des_text is not None:
        (scene_dir / "des.txt").write_text(des_text, encoding="utf-8")
    return scene_dir


def _write_manifest(data_dir: Path, scene_names: list[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "scene_count": len(scene_names),
        "scenes": [{"scene_name": name} for name in scene_names],
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )


def _valid_jpeg_bytes(color: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (1280, 720), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _assert_image_verifies(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()


def _records_by_scene(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["scene"]: record for record in records}


def _pose_pointing_scoring(*, score_margin: float = 0.5) -> dict[str, Any]:
    return {
        "arm_side": "left",
        "keypoint_confidences": {
            "left_shoulder": 0.91,
            "left_elbow": 0.92,
            "left_wrist": 0.93,
        },
        "arm_vector": [200.0, 35.0],
        "candidate_scores": [
            {
                "track_id": 8,
                "score": 0.95,
                "arm_side": "left",
                "perpendicular_distance": 12.0,
                "ray_intersects_bbox": True,
            }
        ],
        "score_margin": score_margin,
        "ambiguous_score_margin": 0.08,
        "checks": {"keypoints_ok": True, "margin_ok": True},
    }


def test_discovers_all_jpeg_scene_dirs_without_manifest_as_authority(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "val-data"
    _make_scene(data_dir, "alpha")
    _make_scene(data_dir, "beta", frames=2)
    _make_scene(data_dir, "gamma")
    _write_manifest(data_dir, ["alpha", "stale_manifest_scene"])

    scenes = module.discover_scene_dirs(data_dir)
    manifest = module.manifest_risk_report(data_dir, scenes)

    assert [scene.name for scene in scenes] == ["alpha", "beta", "gamma"]
    assert manifest["matches_actual_scene_dirs"] is False
    assert manifest["manifest_scene_count"] == 2
    assert manifest["actual_scene_count"] == 3
    assert manifest["missing_from_manifest"] == ["beta", "gamma"]
    assert manifest["manifest_only_scenes"] == ["stale_manifest_scene"]
    assert manifest["risks"]


def test_maps_des_txt_to_stable_agent_payloads_without_low_level_fields(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "val-data"
    _make_scene(
        data_dir,
        "pic_teach_me",
        des_text="请你记住我，我是小李飞刀",
    )
    _make_scene(
        data_dir,
        "pic_teach_person",
        des_text="这是彭刚，请你记住",
    )
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
    )
    _make_scene(
        data_dir,
        "pic_teach_item_phone",
        des_text="这是手机，请你记住",
    )

    records = module.build_teach_payload_records(data_dir, camera="front")
    by_scene = _records_by_scene(records)

    assert by_scene["pic_teach_me"]["endpoint"] == "/v1/memory/teach/person"
    assert by_scene["pic_teach_me"]["payload"] == {
        "camera": "front",
        "stream_ref": memory_e2e.PAYLOAD_FIXTURE_STREAM_REF,
        "target": {
            "kind": "person",
            "intent": "self_introduction",
            "referent_text": "我",
        },
        "profile": {"display_name": "小李飞刀"},
    }
    assert by_scene["pic_teach_person"]["payload"] == {
        "camera": "front",
        "stream_ref": memory_e2e.PAYLOAD_FIXTURE_STREAM_REF,
        "target": {
            "kind": "person",
            "intent": "third_person_introduction",
            "referent_text": "这位/彭刚",
        },
        "profile": {"display_name": "彭刚"},
    }
    assert by_scene["pic_teach_scene_galbot"]["payload"] == {
        "camera": "front",
        "stream_ref": memory_e2e.PAYLOAD_FIXTURE_STREAM_REF,
        "target": {
            "kind": "scene",
            "intent": "teach_scene",
            "referent_text": "这里/银河通用办公室",
        },
        "memory": {"title": "银河通用办公室"},
    }
    assert by_scene["pic_teach_item_phone"]["endpoint"] == "/v1/memory/resolve-target"
    assert by_scene["pic_teach_item_phone"]["payload"] == {
        "camera": "front",
        "stream_ref": memory_e2e.PAYLOAD_FIXTURE_STREAM_REF,
        "target": {
            "kind": "object",
            "intent": "teach_object",
            "referent_text": "手机",
        },
    }
    assert by_scene["pic_teach_item_phone"]["expected"] == {
        "negative_only": True,
        "status": "not_found",
        "error_code": "unsupported_target_kind",
        "writes_memory": False,
    }

    for record in records:
        assert module.find_forbidden_agent_payload_fields(record["payload"]) == []


def test_generated_payloads_parse_with_public_memory_api_contract(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "val-data"
    _make_scene(data_dir, "pic_teach_me", des_text="请你记住我，我是小李飞刀")
    _make_scene(data_dir, "pic_teach_person", des_text="这是彭刚，请你记住")
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
    )
    _make_scene(data_dir, "pic_teach_item_phone", des_text="这是手机，请你记住")

    records = module.build_teach_payload_records(data_dir, camera="front")

    for record in records:
        if record["endpoint"] == "/v1/memory/teach/person":
            TeachPersonRequest.model_validate(record["payload"])
        elif record["endpoint"] == "/v1/memory/teach/scene":
            TeachSceneRequest.model_validate(record["payload"])
        elif record["endpoint"] == "/v1/memory/resolve-target":
            ResolveTargetRequest.model_validate(record["payload"])
        else:
            raise AssertionError(f"unexpected endpoint: {record['endpoint']}")


def test_teach_person_helper_merges_anonymous_required_response() -> None:
    posts: list[dict[str, Any]] = []

    class FakeResponse:
        def __init__(self, status_code: int, body: dict[str, Any]) -> None:
            self.status_code = status_code
            self._body = body

        def json(self) -> dict[str, Any]:
            return self._body

    class FakeClient:
        def post(self, endpoint: str, json: dict[str, Any]) -> FakeResponse:
            posts.append({"endpoint": endpoint, "payload": json})
            if endpoint == "/v1/memory/teach/person":
                return FakeResponse(
                    409,
                    {
                        "detail": {
                            "code": "anonymous_merge_required",
                            "error_code": "anonymous_merge_required",
                            "outcome": "merge_anonymous_required",
                            "anonymous_id": "anon_123",
                            "evidence": {"resolver_target_ref": "front:track:8"},
                        }
                    },
                )
            assert endpoint == "/v1/memory/merge-anonymous-person"
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "anonymous_id": "anon_123",
                    "person_id": "person_from_merge",
                    "copied_embedding_count": 1,
                    "merge_id": "merge_123",
                },
            )

    records: list[dict[str, Any]] = []
    result = module._post_teach_person_with_optional_anonymous_merge(
        runner=SimpleNamespace(client=FakeClient()),
        api_response_records=records,
        payload_index="pic_teach_person:teach",
        scene="pic_teach_person",
        endpoint="/v1/memory/teach/person",
        payload={
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "third_person_introduction",
                "referent_text": "这位/彭刚",
            },
            "profile": {"display_name": "彭刚"},
        },
        operation="teach_person_third_person",
    )

    assert [post["endpoint"] for post in posts] == [
        "/v1/memory/teach/person",
        "/v1/memory/merge-anonymous-person",
    ]
    assert posts[1]["payload"] == {
        "anonymous_id": "anon_123",
        "profile": {"display_name": "彭刚"},
    }
    assert result["status_code"] == 200
    assert result["body"]["person_id"] == "person_from_merge"
    assert result["body"]["teach_person_outcome"] == "merge_anonymous_required"
    assert result["body"]["evidence"] == {"resolver_target_ref": "front:track:8"}
    assert result["body"]["teach_person"]["detail"]["anonymous_id"] == "anon_123"
    assert result["body"]["merge_anonymous_person"]["person_id"] == "person_from_merge"
    assert [record["operation"] for record in records] == [
        "teach_person_third_person",
        "teach_person_third_person_merge_anonymous_person",
    ]


def test_teach_person_helper_keeps_non_anonymous_409_failed() -> None:
    posts: list[dict[str, Any]] = []

    class FakeResponse:
        status_code = 409

        def json(self) -> dict[str, Any]:
            return {
                "detail": {
                    "code": "person_teach_conflict",
                    "outcome": "conflict",
                    "matched_person_id": "person_existing",
                }
            }

    class FakeClient:
        def post(self, endpoint: str, json: dict[str, Any]) -> FakeResponse:
            posts.append({"endpoint": endpoint, "payload": json})
            return FakeResponse()

    records: list[dict[str, Any]] = []
    result = module._post_teach_person_with_optional_anonymous_merge(
        runner=SimpleNamespace(client=FakeClient()),
        api_response_records=records,
        payload_index="pic_teach_me:teach",
        scene="pic_teach_me",
        endpoint="/v1/memory/teach/person",
        payload={
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
            "profile": {"display_name": "小李飞刀"},
        },
        operation="teach_person_self",
    )

    assert [post["endpoint"] for post in posts] == ["/v1/memory/teach/person"]
    assert result["status_code"] == 409
    assert result["body"]["detail"]["code"] == "person_teach_conflict"
    assert records == [
        {
            "payload_index": "pic_teach_me:teach",
            "scene": "pic_teach_me",
            "endpoint": "/v1/memory/teach/person",
            "operation": "teach_person_self",
            "dry_run": False,
            "status_code": 409,
            "payload": {
                "camera": "front",
                "target": {
                    "kind": "person",
                    "intent": "self_introduction",
                    "referent_text": "我",
                },
                "profile": {"display_name": "小李飞刀"},
            },
            "response": {
                "detail": {
                    "code": "person_teach_conflict",
                    "outcome": "conflict",
                    "matched_person_id": "person_existing",
                }
            },
        }
    ]


def test_teach_person_report_fields_promote_person_visual_evidence() -> None:
    person_visual_evidence = {
        "source_frame_ref": "front:42:1000",
        "source_bbox_xyxy": [10, 20, 120, 180],
        "embedding_crop_path": "runtime/memory/artifacts/person.jpg",
        "face_detection": {
            "coordinate_space": "crop",
            "face_bbox_xyxy": [20, 30, 80, 90],
            "score": 0.97,
            "source": "local_embedding_scrfd",
        },
    }

    fields = module._teach_person_report_fields(
        {
            "status_code": 200,
            "body": {
                "ok": True,
                "person_id": "person_123",
                "evidence": {
                    "crop_hash": "crop-hash",
                    "person_visual_evidence": person_visual_evidence,
                },
            },
        }
    )

    assert fields == {"person_visual_evidence": person_visual_evidence}


def test_memory_e2e_runner_generates_public_rest_payloads_without_low_level_fields() -> None:
    payloads = [
        memory_e2e.self_introduction_payload(
            camera="front",
            stream_ref=memory_e2e.PAYLOAD_FIXTURE_STREAM_REF,
            display_name="小李飞刀",
            description="public self introduction",
            tags=["memory-e2e"],
        ),
        memory_e2e.third_person_introduction_payload(
            camera="front",
            stream_ref=memory_e2e.PAYLOAD_FIXTURE_STREAM_REF,
            display_name="彭刚",
            referent_text="这位/彭刚",
        ),
        memory_e2e.teach_scene_payload(
            camera="front",
            stream_ref=memory_e2e.PAYLOAD_FIXTURE_STREAM_REF,
            title="银河通用办公室",
            description="public whole-scene teaching",
            activation_hint="office",
        ),
        memory_e2e.object_resolve_payload(
            camera="front",
            stream_ref=memory_e2e.PAYLOAD_FIXTURE_STREAM_REF,
            referent_text="手机",
        ),
    ]

    TeachPersonRequest.model_validate(payloads[0])
    TeachPersonRequest.model_validate(payloads[1])
    TeachSceneRequest.model_validate(payloads[2])
    ResolveTargetRequest.model_validate(payloads[3])

    assert payloads[0]["target"] == {
        "kind": "person",
        "intent": "self_introduction",
        "referent_text": "我",
    }
    assert payloads[1]["target"] == {
        "kind": "person",
        "intent": "third_person_introduction",
        "referent_text": "这位/彭刚",
    }
    assert payloads[2]["target"] == {
        "kind": "scene",
        "intent": "teach_scene",
        "referent_text": "这里",
    }
    assert payloads[3]["target"] == {
        "kind": "object",
        "intent": "teach_object",
        "referent_text": "手机",
    }
    for payload in payloads:
        assert module.find_forbidden_agent_payload_fields(payload) == []


def test_memory_e2e_processor_keeps_keypoints_in_snapshot_side_channel_only() -> None:
    processor = memory_e2e.MemoryScenarioProcessor()
    processor.mode = "third_person"
    frame = FrameMessage(
        camera="front",
        frame_id=1,
        timestamp_ms=1_000,
        width=1280,
        height=720,
        jpeg_bytes=b"jpeg",
        head_motion_state="stationary",
    )

    visual_state = asyncio.run(processor.process_frame(frame))
    snapshot = processor.take_memory_frame_snapshot()

    assert visual_state["scene_context"]["engagement_state"] == "available"
    assert all("keypoints" not in track for track in visual_state["tracks"])
    assert snapshot is not None
    assert snapshot.attention is not None
    assert snapshot.attention.largest_person_stable is True
    assert snapshot.tracks[0].track_id == memory_e2e.PRIMARY_TRACK_ID
    assert snapshot.tracks[0].keypoints
    assert snapshot.tracks[1].track_id == memory_e2e.AMBIGUOUS_TRACK_ID


def test_dry_run_writes_minimal_artifact_skeleton_and_evidence_index(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "val-data"
    for name in ["pci_stand", "pic_hello"]:
        _make_scene(data_dir, name, frames=2)
    _make_scene(data_dir, "pic_teach_me", des_text="请你记住我，我是小李飞刀")
    _make_scene(data_dir, "pic_teach_person", des_text="这是彭刚，请你记住")
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
    )
    _make_scene(data_dir, "pic_teach_item_phone", des_text="这是手机，请你记住")
    _write_manifest(data_dir, ["pci_stand"])
    out = tmp_path / "artifacts" / "memory-teaching-ga"

    exit_code = module.main(
        ["--data-dir", str(data_dir), "--out", str(out), "--dry-run"]
    )

    assert exit_code == 0
    required_files = [
        "report.json",
        "timeline.jsonl",
        "teach_payloads.json",
        "api_responses.jsonl",
        "botified_frames.jsonl",
        "visual-evidence/index.html",
    ]
    for relative_path in required_files:
        assert (out / relative_path).is_file()

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["mode"] == "dry-run"
    assert report["scene_count"] == 6
    assert report["manifest"]["matches_actual_scene_dirs"] is False
    assert report["warnings"]
    assert report["visual_evidence_index"]
    assert all(
        evidence["kind"] == "html_index"
        for evidence in report["visual_evidence_index"]
    )
    for evidence in report["visual_evidence_index"]:
        assert (out / evidence["path"]).is_file()

    payloads = json.loads((out / "teach_payloads.json").read_text(encoding="utf-8"))
    assert payloads["schema_version"] == 1
    assert len(payloads["payloads"]) == 4
    for record in payloads["payloads"]:
        assert module.find_forbidden_agent_payload_fields(record["payload"]) == []

    responses = [
        json.loads(line)
        for line in (out / "api_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    object_response = next(
        response
        for response in responses
        if response["scene"] == "pic_teach_item_phone"
    )
    assert object_response["dry_run"] is True
    assert object_response["response"]["status"] == "not_found"
    assert object_response["response"]["error_code"] == "unsupported_target_kind"


def test_actual_fake_runner_replays_scenes_and_writes_real_api_artifacts(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "val-data"
    scene_jpegs = {
        "pci_stand": _valid_jpeg_bytes((128, 128, 128)),
        "pic_hello": _valid_jpeg_bytes((48, 128, 196)),
        "pic_teach_me": _valid_jpeg_bytes((196, 64, 64)),
        "pic_teach_person": _valid_jpeg_bytes((64, 180, 96)),
        "pic_teach_scene_galbot": _valid_jpeg_bytes((220, 190, 72)),
        "pic_teach_item_phone": _valid_jpeg_bytes((140, 80, 196)),
    }
    for name in ["pci_stand", "pic_hello"]:
        _make_scene(data_dir, name, frames=2, jpeg_bytes=scene_jpegs[name])
    _make_scene(
        data_dir,
        "pic_teach_me",
        des_text="请你记住我，我是小李飞刀",
        jpeg_bytes=scene_jpegs["pic_teach_me"],
    )
    _make_scene(
        data_dir,
        "pic_teach_person",
        des_text="这是彭刚，请你记住",
        jpeg_bytes=scene_jpegs["pic_teach_person"],
    )
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
        jpeg_bytes=scene_jpegs["pic_teach_scene_galbot"],
    )
    _make_scene(
        data_dir,
        "pic_teach_item_phone",
        des_text="这是手机，请你记住",
        jpeg_bytes=scene_jpegs["pic_teach_item_phone"],
    )
    out = tmp_path / "artifacts" / "memory-teaching-ga"

    exit_code = module.main(["--data-dir", str(data_dir), "--out", str(out)])

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["mode"] == "actual"
    assert report["backend"] == "fake"
    assert report["scene_count"] == 6
    assert report["replayed_scene_count"] == 6
    expected_scene_names = sorted(scene_jpegs)
    assert report["post_teach_scene_replay"]["runner_case"] == (
        "ga-post-teach-scene-replay"
    )
    assert "self_introduction" not in report
    assert "teach_scene" not in report
    assert report["post_teach_scene_replay"]["replayed_scene_count"] == 6
    assert report["post_teach_scene_replay"]["replayed_scene_names"] == (
        expected_scene_names
    )
    assert all(
        fields == [] for fields in report["forbidden_agent_payload_fields"].values()
    )
    object_no_write = report["object_no_write"]
    assert object_no_write["assertions"]["no_memory_write"] is True
    store_delta = object_no_write["store_delta"]
    assert store_delta["before"] == store_delta["after"]
    assert all(value == 0 for value in store_delta["delta"].values())
    assert {
        "embedding_provenance",
        "conversation_summaries",
        "external_user_links",
        "memory_match_records",
        "profile_merge_history",
        "negative_identity_matches",
        "person_embedding_vectors",
        "scene_embedding_vectors",
        "anonymous_embedding_vectors",
    } <= set(store_delta["delta"])
    assert object_no_write["store_delta_source"] == {
        "universe": "MemoryStore.memory_table_counts",
        "allowed_diagnostic_whitelist": [],
    }
    visual_items = report["visual_evidence_index"]
    overlay_items = [
        item for item in visual_items if item["kind"] == "image_overlay"
    ]
    overlays_by_assertion = {
        item["assertion_id"]: item for item in overlay_items
    }
    assert {
        "self_introduction_known_person",
        "third_person_pose_pointing",
        "teach_scene_scene_activated",
        "object_unsupported_no_write",
    } <= set(overlays_by_assertion)
    for item in overlay_items:
        assert item["scene"]
        assert item["report_section"]
        image_path = out / item["path"]
        assert image_path.is_file()
        _assert_image_verifies(image_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["all_scenes_replayed"]["passed"] is True
    assert checks["actual_api_responses"]["passed"] is True
    assert checks["cli_projection_botified_frames"]["passed"] is True
    assert checks["self_introduction_known_person_present"]["passed"] is True
    assert checks["third_person_known_person_present"]["passed"] is True
    assert checks["teach_scene_scene_activated"]["passed"] is True
    assert checks["post_teach_all_scenes_memory_behavior"]["passed"] is True
    assert checks["object_resolve_unsupported_no_write"]["passed"] is True
    assert checks["bounded_multi_person_recognition"]["passed"] is True

    bounded_recognition = report["bounded_multi_person_recognition"]
    assert bounded_recognition == checks["bounded_multi_person_recognition"]["details"]
    assert bounded_recognition["attention_target_only"] is False
    assert bounded_recognition["tracks_seen"] == 2
    assert bounded_recognition["tracks_eligible"] == 2
    assert bounded_recognition["tracks_candidates"] == 2
    assert bounded_recognition["candidate_track_ids"] == [7, 8]
    assert bounded_recognition["tracks_queried"] == 2
    assert bounded_recognition["tracks_queried"] <= bounded_recognition[
        "max_tracks_per_tick"
    ]
    assert bounded_recognition["attention_target_track_id"] == 7
    assert bounded_recognition["queried_track_ids"] == [7, 8]
    assert bounded_recognition["recognition_runs_in_executor"] is True

    post_teach = report["post_teach_scene_replay"]
    assert post_teach == checks["post_teach_all_scenes_memory_behavior"]["details"]
    assert post_teach["assertions"] == {
        "all_required_teaching_scenes_present": True,
        "all_scenes_replayed": True,
        "self_positive_scene_confirmed": True,
        "third_person_positive_scene_confirmed": True,
        "scene_positive_scene_confirmed": True,
        "non_self_scenes_no_taught_self_confirmed": True,
        "non_third_person_scenes_no_taught_third_person_confirmed": True,
        "non_scene_scenes_no_taught_scene_confirmed": True,
    }
    assert post_teach["self_person_id"]
    assert post_teach["third_person_id"]
    assert post_teach["scene_id"]
    post_scenes = {scene["scene"]: scene for scene in post_teach["scenes"]}
    assert sorted(post_scenes) == expected_scene_names
    assert post_scenes["pic_teach_me"]["flags"][
        "taught_self_known_person_present"
    ] is True
    assert post_scenes["pic_teach_person"]["flags"][
        "taught_third_person_known_person_present"
    ] is True
    assert post_scenes["pic_teach_scene_galbot"]["flags"][
        "taught_scene_activated"
    ] is True
    for scene_name, scene_result in post_scenes.items():
        flags = scene_result["flags"]
        if scene_name != "pic_teach_me":
            assert flags["taught_self_known_person_present"] is False
        if scene_name != "pic_teach_person":
            assert flags["taught_third_person_known_person_present"] is False
        if scene_name != "pic_teach_scene_galbot":
            assert flags["taught_scene_activated"] is False

    third_person = report["third_person_introduction"]
    assert third_person == checks["third_person_known_person_present"]["details"]
    assert third_person["resolver_target_ref"] == "front:track:8"
    assert third_person["introducer_ref"] == "front:track:7"
    assert third_person["stored_person_id"] == third_person["person_id"]
    assert third_person["stored_embedding_source_track_ref"] == "front:track:8"
    assert third_person["stored_crop_hash"]
    assert third_person["stored_crop_path_or_artifact_ref"]
    assert "teach_person" not in third_person
    assert third_person["pose_pointing_scoring"]["checks"][
        "keypoints_ok"
    ] is True
    assert third_person["pose_pointing_scoring"]["checks"]["margin_ok"] is True
    assert third_person["debug_test_channel_enabled"] is False
    assert third_person["fixture_inputs_consumed"] == []
    assert third_person["debug_fixture_used_for_target_resolution"] is False
    assert third_person["bounded_multi_person_recognition"] == bounded_recognition
    third_person_crop_path = Path(third_person["stored_crop_path_or_artifact_ref"])
    assert third_person_crop_path.is_file()
    assert out / "runtime" / "memory" / "artifacts" in third_person_crop_path.parents
    assert (
        hashlib.sha256(third_person_crop_path.read_bytes()).hexdigest()
        == third_person["stored_crop_hash"]
    )
    stability_window = third_person["resolve_target"]["evidence"]["stability_window"]
    assert stability_window["active_snapshot_count"] >= 2
    assert stability_window["active_target_track_id"] == 7
    third_overlay = overlays_by_assertion["third_person_pose_pointing"]
    third_evidence = third_person["resolve_target"]["evidence"]
    assert third_overlay["resolver_target_ref"] == third_person["resolver_target_ref"]
    assert third_overlay["introducer_ref"] == third_person["introducer_ref"]
    assert third_overlay["stored_embedding_source_track_ref"] == (
        third_person["stored_embedding_source_track_ref"]
    )
    assert third_overlay["crop_hash"] == third_person["stored_crop_hash"]
    assert third_overlay["crop_path_or_artifact_ref"] == (
        third_person["stored_crop_path_or_artifact_ref"]
    )
    assert third_overlay["request_snapshot_ref"] == third_evidence[
        "request_snapshot_ref"
    ]
    assert third_overlay["source_frame_ref"] == third_evidence["source_frame_ref"]
    assert third_overlay["pose_stability_window"]["selected_target_track_id"] == (
        third_evidence["pose_stability_window"]["selected_target_track_id"]
    )
    assert third_overlay["candidate_score"] == third_person[
        "pose_pointing_scoring"
    ]["candidate_scores"][0]["score"]
    assert third_overlay["target_bbox_xyxy"] == third_person["resolve_target"][
        "candidates"
    ][0]["bbox_xyxy"]
    object_overlay = overlays_by_assertion["object_unsupported_no_write"]
    assert object_overlay["error_code"] == (
        object_no_write["resolve_target"]["error_code"]
    )
    assert object_overlay["error_code"] == "unsupported_target_kind"
    assert object_overlay["status"] == object_no_write["resolve_target"]["status"]
    assert object_overlay["store_delta_summary"]["before_equals_after"] is True
    assert object_overlay["store_delta_summary"]["delta_all_zero"] is True
    assert all(
        value == 0
        for value in object_overlay["store_delta_summary"]["delta"].values()
    )
    assert third_person["assertions"][
        "stored_embedding_source_is_target"
    ] is True
    assert third_person["assertions"][
        "stored_embedding_source_not_introducer"
    ] is True
    assert third_person["assertions"][
        "b_positive_known_person_present"
    ] is True
    assert third_person["assertions"][
        "a_only_no_known_person_for_stored_person"
    ] is True
    assert third_person["assertions"]["pose_pointing_scoring_present"] is True
    assert third_person["assertions"]["pose_pointing_checks_passed"] is True
    assert third_person["b_positive_replay"]["known_person_present"] is True
    assert third_person["b_positive_replay"][
        "stored_person_known_person_present"
    ] is True
    assert third_person["b_positive_replay"]["stored_person_track_id"] == 8
    assert any(
        event["event"] == "known_person_present"
        and event["track_id"] == 8
        and event["memory_context"]["person"]["person_id"]
        == third_person["stored_person_id"]
        for event in third_person["b_positive_replay"]["events"]
    )
    assert third_person["a_only_negative_replay"][
        "known_person_present"
    ] is False
    assert third_person["a_only_negative_replay"][
        "stored_person_known_person_present"
    ] is False
    assert all(
        event["event"] != "known_person_present"
        for event in third_person["a_only_negative_replay"]["events"]
    )

    payloads = json.loads((out / "teach_payloads.json").read_text(encoding="utf-8"))
    assert payloads["mode"] == "actual"
    for record in payloads["payloads"]:
        assert module.find_forbidden_agent_payload_fields(record["payload"]) == []

    responses = [
        json.loads(line)
        for line in (out / "api_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert responses
    assert all(response["dry_run"] is False for response in responses)
    assert all(
        response["response"].get("status") != "stubbed" for response in responses
    )
    object_response = next(
        response
        for response in responses
        if response["scene"] == "pic_teach_item_phone"
    )
    assert object_response["response"]["status"] == "not_found"
    assert object_response["response"]["error_code"] == "unsupported_target_kind"

    botified_frames = [
        json.loads(line)
        for line in (out / "botified_frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert botified_frames
    assert all(
        frame["source"] == "cli_frame_pump_stdout" for frame in botified_frames
    )
    assert all(
        frame["botified_frame"].startswith("<botified>")
        and frame["botified_frame"].endswith("</botified>")
        for frame in botified_frames
    )
    assert all("semantic_event" not in frame for frame in botified_frames)
    assert any(
        "visual_context=" in frame["payload"]["request"] for frame in botified_frames
    )
    event_names = {frame["event"] for frame in botified_frames}
    assert {"known_person_present", "scene_activated"} <= event_names


def test_actual_fake_runner_report_includes_supporting_contracts(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "val-data"
    scene_jpegs = {
        "pic_teach_me": _valid_jpeg_bytes((196, 64, 64)),
        "pic_teach_person": _valid_jpeg_bytes((64, 180, 96)),
        "pic_teach_scene_galbot": _valid_jpeg_bytes((220, 190, 72)),
        "pic_teach_item_phone": _valid_jpeg_bytes((140, 80, 196)),
    }
    _make_scene(
        data_dir,
        "pic_teach_me",
        des_text="请你记住我，我是小李飞刀",
        jpeg_bytes=scene_jpegs["pic_teach_me"],
    )
    _make_scene(
        data_dir,
        "pic_teach_person",
        des_text="这是彭刚，请你记住",
        jpeg_bytes=scene_jpegs["pic_teach_person"],
    )
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
        jpeg_bytes=scene_jpegs["pic_teach_scene_galbot"],
    )
    _make_scene(
        data_dir,
        "pic_teach_item_phone",
        des_text="这是手机，请你记住",
        jpeg_bytes=scene_jpegs["pic_teach_item_phone"],
    )
    out = tmp_path / "artifacts" / "memory-teaching-ga"

    exit_code = module.main(["--data-dir", str(data_dir), "--out", str(out)])

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    checks = {check["name"]: check for check in report["checks"]}
    supporting = report["supporting_contracts"]

    assert supporting["passed"] is True
    assert checks["supporting_contracts"]["passed"] is True
    assert checks["supporting_contracts"]["details"] == supporting
    assert {
        "conversation_summary_context",
        "external_user_link",
        "familiar_unknown",
        "merge_anonymous_person",
        "correct_identity",
        "resolve_target_states",
    } <= set(supporting)
    summary_context = supporting["conversation_summary_context"]
    assert summary_context["event_conversation_summaries"]
    assert summary_context["lookup_conversation_summaries"]
    assert "conversation_summaries" not in summary_context
    assert (
        supporting["external_user_link"]["lookup"]["person"]["person_id"]
        == summary_context["person_id"]
    )
    assert supporting["external_user_link"]["lookup_conversation_summaries"]
    assert supporting["familiar_unknown"]["present"] is True
    assert supporting["merge_anonymous_person"]["old_anonymous_suppressed"] is True
    assert supporting["merge_anonymous_person"]["known_replay_present"] is True
    assert supporting["correct_identity"]["wrong_person_not_returned"] is True
    resolve_states = supporting["resolve_target_states"]
    assert set(resolve_states) == {"resolved", "ambiguous", "not_found"}
    assert resolve_states["resolved"]["status"] == "resolved"
    assert resolve_states["ambiguous"]["status"] == "ambiguous"
    assert resolve_states["ambiguous"]["no_memory_write"] is True
    assert resolve_states["not_found"]["status"] == "not_found"
    assert resolve_states["not_found"]["no_memory_write"] is True


def test_bounded_recognition_projection_reads_service_report_without_recomputing() -> None:
    service_report = {
        "camera": "front",
        "frame_id": 44,
        "frame_timestamp_ms": 12_345,
        "source_frame_ref": "front:44:12345",
        "tracks_seen": 99,
        "tracks_eligible": 88,
        "tracks_candidates": 2,
        "candidate_track_ids": [7, 8],
        "tracks_queried": 2,
        "tracks_skipped_reason": {"max_tracks_per_tick": 86},
        "queried_track_ids": [7, 8],
        "attention_target_track_id": 7,
        "attention_target_only": False,
        "max_tracks_per_tick": 4,
        "query_interval_ms": 1_000,
        "event_cooldown_ms": 1_000,
        "recognition_runs_in_executor": True,
        "eligibility_policy": "service-owned-policy",
    }

    class FakeMemoryService:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def latest_recognition_report(self, camera: str | None = None) -> dict[str, Any]:
            self.calls.append(camera)
            return dict(service_report)

    memory_service = FakeMemoryService()
    runner = SimpleNamespace(
        client=SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(memory_service=memory_service),
            ),
        ),
        session_factory=SimpleNamespace(
            last_snapshot=SimpleNamespace(
                tracks=[{"track_id": 7}],
                attention={"target_track_id": 7},
            ),
        ),
    )

    projected = module._latest_bounded_recognition_report_from_runner(
        runner,
        camera="front",
    )
    check = module._bounded_multi_person_recognition_check(
        projected,
        require_non_attention_query=True,
    )

    assert memory_service.calls == ["front"]
    assert projected == service_report
    assert check == {
        "name": "bounded_multi_person_recognition",
        "passed": True,
        "details": service_report,
    }


def test_actual_fake_runner_writes_failed_report_when_teaching_scene_missing(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "val-data"
    jpeg_bytes = _valid_jpeg_bytes()
    _make_scene(data_dir, "pci_stand", jpeg_bytes=jpeg_bytes)
    _make_scene(
        data_dir,
        "pic_teach_me",
        des_text="请你记住我，我是小李飞刀",
        jpeg_bytes=jpeg_bytes,
    )
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
        jpeg_bytes=jpeg_bytes,
    )
    _make_scene(
        data_dir,
        "pic_teach_item_phone",
        des_text="这是手机，请你记住",
        jpeg_bytes=jpeg_bytes,
    )
    out = tmp_path / "artifacts" / "memory-teaching-ga"

    exit_code = module.main(["--data-dir", str(data_dir), "--out", str(out)])

    assert exit_code != 0
    report_path = out / "report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["mode"] == "actual"
    assert report["backend"] == "fake"

    checks = {check["name"]: check for check in report["checks"]}
    payload_check = checks["expected_teach_des_payloads"]
    assert payload_check["passed"] is False
    assert payload_check["details"]["missing"] == ["pic_teach_person"]
    assert checks["third_person_known_person_present"]["passed"] is False
    assert checks["third_person_known_person_present"]["details"]["error"] == (
        "required_teaching_scene_missing"
    )
    assert checks["post_teach_all_scenes_memory_behavior"]["passed"] is False
    assert checks["post_teach_all_scenes_memory_behavior"]["details"]["reason"] == (
        "required_teaching_scene_missing"
    )
    assert checks["post_teach_all_scenes_memory_behavior"]["details"]["missing"] == [
        "pic_teach_person"
    ]
    assert report["post_teach_scene_replay"]["reason"] == (
        "required_teaching_scene_missing"
    )

    for relative_path in [
        "timeline.jsonl",
        "teach_payloads.json",
        "api_responses.jsonl",
        "botified_frames.jsonl",
        "visual-evidence/index.html",
    ]:
        assert (out / relative_path).is_file()


def test_post_teach_scene_replay_uses_one_runner_case(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "val-data"
    scene_jpegs = {
        "other_scene": _valid_jpeg_bytes((32, 32, 32)),
        "pic_teach_me": _valid_jpeg_bytes((196, 64, 64)),
        "pic_teach_person": _valid_jpeg_bytes((64, 180, 96)),
        "pic_teach_scene_galbot": _valid_jpeg_bytes((220, 190, 72)),
    }
    _make_scene(data_dir, "other_scene", jpeg_bytes=scene_jpegs["other_scene"])
    _make_scene(
        data_dir,
        "pic_teach_me",
        des_text="请你记住我，我是小李飞刀",
        jpeg_bytes=scene_jpegs["pic_teach_me"],
    )
    _make_scene(
        data_dir,
        "pic_teach_person",
        des_text="这是彭刚，请你记住",
        jpeg_bytes=scene_jpegs["pic_teach_person"],
    )
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
        jpeg_bytes=scene_jpegs["pic_teach_scene_galbot"],
    )
    scenes = module.discover_scene_dirs(data_dir)
    records = module._build_teach_payload_records_from_scenes(
        scenes,
        camera="front",
    )
    created_cases: list[str] = []

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            created_cases.append(kwargs["case"])
            self.case = kwargs["case"]
            self.source_frame = kwargs["source_frame"]
            self.processor = SimpleNamespace(mode="single")
            self.frame_id = 0

        def open_stream(self):
            return self

        def __enter__(self):
            return object()

        def __exit__(self, *_args: Any) -> None:
            return None

        def send(self, _websocket: Any, *, timestamp_ms: int, states_file: Any, phase: str):
            self.frame_id += 1
            states_file.write(
                json.dumps(
                    {
                        "case": self.case,
                        "phase": phase,
                        "timestamp_ms": timestamp_ms,
                    }
                )
                + "\n"
            )
            return {"semantic_events": []}

        def start_query_and_drain(
            self,
            websocket: Any,
            *,
            query_timestamp_ms: int,
            states_file: Any,
            phase: str,
        ) -> list[dict[str, Any]]:
            self.send(
                websocket,
                timestamp_ms=query_timestamp_ms,
                states_file=states_file,
                phase=f"{phase}:query",
            )
            self.send(
                websocket,
                timestamp_ms=query_timestamp_ms + 1,
                states_file=states_file,
                phase=f"{phase}:drain",
            )
            scene_name = self.source_frame.path.parent.name
            if scene_name == "pic_teach_me":
                return [
                    {
                        "event": "known_person_present",
                        "track_id": 7,
                        "memory_context": {"person": {"person_id": "person_self"}},
                    }
                ]
            if scene_name == "pic_teach_person":
                return [
                    {
                        "event": "known_person_present",
                        "track_id": 8,
                        "memory_context": {"person": {"person_id": "person_third"}},
                    }
                ]
            if scene_name == "pic_teach_scene_galbot":
                return [
                    {
                        "event": "scene_activated",
                        "track_id": None,
                        "memory_context": {"scene": {"scene_id": "scene_ga"}},
                    }
                ]
            return [
                {
                    "event": "familiar_unknown_present",
                    "track_id": 7,
                    "memory_context": {"anonymous": {"anonymous_id": "anon_1"}},
                }
            ]

    def fake_post_and_record_api_response(**kwargs: Any) -> dict[str, Any]:
        scene = kwargs["scene"]
        body_by_scene = {
            "pic_teach_me": {"ok": True, "person_id": "person_self"},
            "pic_teach_person": {"ok": True, "person_id": "person_third"},
            "pic_teach_scene_galbot": {"ok": True, "scene_id": "scene_ga"},
        }
        body = body_by_scene[scene]
        kwargs["api_response_records"].append(
            {
                "scene": scene,
                "operation": kwargs["operation"],
                "payload": kwargs["payload"],
                "response": body,
                "status_code": 200,
                "dry_run": False,
            }
        )
        return {"status_code": 200, "body": body}

    monkeypatch.setattr(module, "_actual_runner", lambda **kwargs: FakeRunner(**kwargs))
    monkeypatch.setattr(
        module,
        "_post_and_record_api_response",
        fake_post_and_record_api_response,
    )

    states_path = tmp_path / "visual_states.jsonl"
    with states_path.open("w", encoding="utf-8") as states_file:
        result = module._run_actual_post_teach_scene_replay(
            scenes=scenes,
            payload_records_by_scene={record["scene"]: record for record in records},
            out=tmp_path,
            camera="front",
            states_file=states_file,
            api_response_records=[],
            botified_frame_records=[],
        )

    assert created_cases == ["ga-post-teach-scene-replay"]
    assert result["runner_case"] == "ga-post-teach-scene-replay"
    assert result["passed"] is True
    assert result["replayed_scene_names"] == sorted(scene_jpegs)
    phases = [
        json.loads(line)["phase"]
        for line in states_path.read_text(encoding="utf-8").splitlines()
    ]
    assert "post-teach-third-person-seed:stable-1" in phases
    assert "post-teach-third-person-seed:stable-2" in phases


def test_local_smoke_requires_explicit_real_local_backends(tmp_path: Path) -> None:
    data_dir = tmp_path / "val-data"
    out = tmp_path / "artifacts" / "memory-teaching-ga-local-smoke"

    exit_code = module.main(
        ["--data-dir", str(data_dir), "--out", str(out), "--local-smoke"]
    )

    assert exit_code != 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["mode"] == "local-smoke"
    assert report["backend"] == "local"
    assert report["status"] == "failed"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["local_smoke_explicit_real_backends"]["passed"] is False
    missing = checks["local_smoke_explicit_real_backends"]["details"]["missing"]
    assert "--embedding-backend local" in missing
    assert "--person-model-path" in missing
    assert "--scene-model-path" in missing
    assert "--inference-backend ultralytics" in missing
    assert "--pose-model-path" in missing


def test_local_runner_frame_bound_posts_include_latest_stream_ref(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "img_000.jpeg"
    frame_path.write_bytes(_valid_jpeg_bytes())
    source_frame = memory_e2e.SourceFrame(
        path=frame_path,
        jpeg_bytes=frame_path.read_bytes(),
        width=1280,
        height=720,
    )
    runner = module.LocalMemorySmokeRunner(
        case="local-stream-ref-payload",
        out=tmp_path,
        camera="front",
        config=module.ServerConfig(
            runtime_dir=tmp_path / "runtime",
            memory=module.MemoryConfig(
                enabled=True,
                db_path=tmp_path / "memory.sqlite3",
                embedding=module.MemoryEmbeddingConfig(backend="fake"),
            ),
        ),
    )
    api_response_records: list[dict[str, Any]] = []
    states_path = tmp_path / "visual_states.jsonl"

    with states_path.open("w", encoding="utf-8") as states_file:
        with runner.open_stream() as websocket:
            runner.send(
                websocket,
                source_frame,
                timestamp_ms=1_000,
                states_file=states_file,
                phase="stream-ref-seed",
            )
            response = module._post_and_record_api_response(
                runner=runner,
                api_response_records=api_response_records,
                payload_index="local:resolve-object",
                scene="pic_teach_item_phone",
                endpoint="/v1/memory/resolve-target",
                payload={
                    "camera": "front",
                    "target": {
                        "kind": "object",
                        "intent": "teach_object",
                        "referent_text": "手机",
                    },
                },
                operation="local_resolve_object_stream_ref_contract",
            )

    assert response["status_code"] == 200
    assert api_response_records
    payload = api_response_records[0]["payload"]
    assert payload["stream_ref"] == runner.latest_stream_ref
    assert payload["stream_ref"].startswith("ws_")
    assert payload["stream_ref"] != memory_e2e.PAYLOAD_FIXTURE_STREAM_REF


def test_local_smoke_report_fails_insufficient_third_person_without_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "val-data"
    jpeg_bytes = _valid_jpeg_bytes()
    _make_scene(
        data_dir,
        "pic_teach_me",
        des_text="请你记住我，我是小李飞刀",
        jpeg_bytes=jpeg_bytes,
    )
    _make_scene(
        data_dir,
        "pic_teach_person",
        des_text="这是彭刚，请你记住",
        jpeg_bytes=jpeg_bytes,
    )
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
        jpeg_bytes=jpeg_bytes,
    )
    person_model = tmp_path / "runtime" / "models" / "face-buffalo-s"
    scene_model = tmp_path / "runtime" / "models" / "scene-mobileclip2-s0"
    pose_model = tmp_path / "runtime" / "models" / "yolov8n-pose.pt"
    person_model.mkdir(parents=True)
    scene_model.mkdir(parents=True)
    pose_model.parent.mkdir(parents=True, exist_ok=True)
    pose_model.write_bytes(b"pose")
    out = tmp_path / "artifacts" / "memory-teaching-ga-local-smoke"

    def fake_execute(**_kwargs):
        return {
            "self_smoke": {
                "status": "passed",
                "passed": True,
                "selected_window": {"scene": "pic_teach_me"},
            },
            "scene_smoke": {
                "status": "passed",
                "passed": True,
                "selected_window": {"scene": "pic_teach_scene_galbot"},
            },
            "third_person_probe": {
                "status": "insufficient_sample",
                "passed": False,
                "reason": "pose_unclear",
                "observations": [{"track_count": 1, "keypoint_tracks": 1}],
            },
            "api_response_records": [],
            "botified_frame_records": [],
        }

    monkeypatch.setattr(module, "_execute_local_smoke", fake_execute)

    exit_code = module.main(
        [
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--camera",
            "front",
            "--local-smoke",
            "--embedding-backend",
            "local",
            "--person-model-path",
            str(person_model),
            "--scene-model-path",
            str(scene_model),
            "--inference-backend",
            "ultralytics",
            "--pose-model-path",
            str(pose_model),
        ]
    )

    assert exit_code != 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["status"] == "failed"
    assert report["mode"] == "local-smoke"
    assert report["real_model_evidence"] is True
    assert report["self_smoke"]["status"] == "passed"
    assert report["scene_smoke"]["status"] == "passed"
    assert report["third_person_probe"]["status"] == "insufficient_sample"
    assert report["third_person_probe"]["debug_test_channel_enabled"] is False
    assert report["third_person_probe"]["fixture_inputs_consumed"] == []
    assert (
        report["third_person_probe"]["debug_fixture_used_for_target_resolution"]
        is False
    )
    assert all(
        fields == [] for fields in report["forbidden_agent_payload_fields"].values()
    )
    payloads = json.loads((out / "teach_payloads.json").read_text(encoding="utf-8"))
    assert payloads["mode"] == "local-smoke"
    for record in payloads["payloads"]:
        assert module.find_forbidden_agent_payload_fields(record["payload"]) == []

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["self_local_smoke"]["passed"] is True
    assert checks["scene_local_smoke"]["passed"] is True
    assert checks["third_person_local_probe"]["passed"] is False
    assert checks["third_person_local_probe"]["details"]["status"] == "insufficient_sample"
    assert checks["third_person_local_probe"]["details"][
        "debug_test_channel_enabled"
    ] is False
    assert checks["third_person_local_probe"]["details"]["fixture_inputs_consumed"] == []
    assert checks["third_person_local_probe"]["details"][
        "debug_fixture_used_for_target_resolution"
    ] is False


def test_local_smoke_runner_writes_visual_evidence_overlays_join_report_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "val-data"
    _make_scene(
        data_dir,
        "pic_teach_me",
        des_text="请你记住我，我是小李飞刀",
        jpeg_bytes=_valid_jpeg_bytes((196, 64, 64)),
    )
    _make_scene(
        data_dir,
        "pic_teach_person",
        des_text="这是彭刚，请你记住",
        jpeg_bytes=_valid_jpeg_bytes((64, 180, 96)),
    )
    _make_scene(
        data_dir,
        "pic_teach_scene_galbot",
        des_text="这是银河通用的办公室，请你记住",
        jpeg_bytes=_valid_jpeg_bytes((220, 190, 72)),
    )
    person_model = tmp_path / "runtime" / "models" / "face-buffalo-s"
    scene_model = tmp_path / "runtime" / "models" / "scene-mobileclip2-s0"
    pose_model = tmp_path / "runtime" / "models" / "yolov8n-pose.pt"
    person_model.mkdir(parents=True)
    scene_model.mkdir(parents=True)
    pose_model.parent.mkdir(parents=True, exist_ok=True)
    pose_model.write_bytes(b"pose")
    out = tmp_path / "artifacts" / "memory-teaching-ga-local-smoke"

    self_result = {
        "status": "passed",
        "passed": True,
        "person_id": "person_self",
        "teach_crop_hash": "self_crop_hash",
        "teach_crop_path_or_artifact_ref": "runtime/memory/artifacts/self.jpg",
        "selected_window": {
            "scene": "pic_teach_me",
            "frame": str(data_dir / "pic_teach_me" / "img_000.jpeg"),
        },
        "events": [
            {
                "event": "known_person_present",
                "event_id": "evt_self",
                "evidence": {"memory_match_id": "match_self"},
            }
        ],
    }
    third_person_result = {
        "status": "passed",
        "passed": True,
        "resolver_target_ref": "front:track:8",
        "introducer_ref": "front:track:7",
        "stored_embedding_source_track_ref": "front:track:8",
        "stored_crop_hash": "third_crop_hash",
        "stored_crop_path_or_artifact_ref": "runtime/memory/artifacts/third.jpg",
        "selected_window": {
            "scene": "pic_teach_person",
            "frame": str(data_dir / "pic_teach_person" / "img_000.jpeg"),
        },
        "pose_pointing_scoring": {
            "candidate_scores": [{"track_id": 8, "score": 0.93}]
        },
        "resolve_target": {
            "status": "ok",
            "evidence": {
                "request_snapshot_ref": "memory_frame:front:42:1000",
                "source_frame_ref": "front:42:1000",
                "pose_stability_window": {
                    "size": 3,
                    "fresh_snapshot_count": 3,
                    "required_pose_snapshot_count": 2,
                    "selected_target_track_id": 8,
                    "selected_arm_side": "left",
                    "selected_count": 2,
                    "failure_reason": None,
                },
            },
            "candidates": [
                {"track_id": 8, "bbox_xyxy": [144, 80, 360, 650]},
            ],
        },
        "bounded_multi_person_recognition": {
            "attention_target_only": False,
            "tracks_seen": 2,
            "tracks_eligible": 2,
            "tracks_candidates": 2,
            "candidate_track_ids": [7, 8],
            "tracks_queried": 2,
            "queried_track_ids": [7, 8],
            "tracks_skipped_reason": {},
            "attention_target_track_id": 7,
            "max_tracks_per_tick": 4,
            "query_interval_ms": 250,
            "event_cooldown_ms": 2000,
            "recognition_runs_in_executor": True,
        },
    }
    scene_result = {
        "status": "passed",
        "passed": True,
        "scene_id": "scene_galbot",
        "teach_crop_hash": "scene_crop_hash",
        "teach_crop_path_or_artifact_ref": "runtime/memory/artifacts/scene.jpg",
        "selected_window": {
            "scene": "pic_teach_scene_galbot",
            "frame": str(data_dir / "pic_teach_scene_galbot" / "img_000.jpeg"),
        },
        "events": [
            {
                "event": "scene_activated",
                "event_id": "evt_scene",
                "evidence": {"memory_match_id": "match_scene"},
            }
        ],
    }

    def fake_execute(**_kwargs):
        return {
            "self_smoke": self_result,
            "scene_smoke": scene_result,
            "third_person_probe": third_person_result,
            "api_response_records": [],
            "botified_frame_records": [],
        }

    monkeypatch.setattr(module, "_execute_local_smoke", fake_execute)

    exit_code = module.main(
        [
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--camera",
            "front",
            "--local-smoke",
            "--embedding-backend",
            "local",
            "--person-model-path",
            str(person_model),
            "--scene-model-path",
            str(scene_model),
            "--inference-backend",
            "ultralytics",
            "--pose-model-path",
            str(pose_model),
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    visual_evidence_index = report["visual_evidence_index"]
    overlays_by_assertion = {
        item["assertion_id"]: item
        for item in visual_evidence_index
        if item["kind"] == "image_overlay"
    }
    assert set(overlays_by_assertion) == {
        "self_introduction_known_person",
        "third_person_pose_pointing",
        "teach_scene_scene_activated",
    }
    assert (
        overlays_by_assertion["self_introduction_known_person"]["report_section"]
        == "self_smoke"
    )
    third_overlay = overlays_by_assertion["third_person_pose_pointing"]
    assert third_overlay["report_section"] == "third_person_probe"
    assert third_overlay["request_snapshot_ref"] == "memory_frame:front:42:1000"
    assert third_overlay["source_frame_ref"] == "front:42:1000"
    assert third_overlay["candidate_score"] == 0.93
    assert third_overlay["target_bbox_xyxy"] == [144.0, 80.0, 360.0, 650.0]
    assert third_overlay["pose_stability_window"]["selected_count"] == 2
    assert (
        overlays_by_assertion["teach_scene_scene_activated"]["report_section"]
        == "scene_smoke"
    )
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["artifact_skeleton"]["passed"] is True
    for item in overlays_by_assertion.values():
        image_path = out / item["path"]
        assert image_path.is_file()
        _assert_image_verifies(image_path)

    index_html = (out / "visual-evidence" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "third_person_probe" in index_html
    assert "third-person-pose-pointing.jpg" in index_html
    assert "object_unsupported_no_write" not in index_html


def test_local_third_person_probe_passes_after_resolve_teach_and_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame_path = tmp_path / "img_000.jpeg"
    frame_path.write_bytes(_valid_jpeg_bytes())
    scene = module.SceneDir(
        name="pic_teach_person",
        path=tmp_path,
        jpeg_paths=(frame_path,),
        des_text="这是彭刚，请你记住",
    )
    source_frame = memory_e2e.SourceFrame(
        path=frame_path,
        jpeg_bytes=frame_path.read_bytes(),
        width=1280,
        height=720,
    )
    record = {
        "scene": "pic_teach_person",
        "endpoint": "/v1/memory/teach/person",
        "payload": {
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "third_person_introduction",
                "referent_text": "这位/彭刚",
            },
            "profile": {"display_name": "彭刚"},
        },
    }
    posted_operations: list[str] = []
    replay_calls: list[dict[str, Any]] = []
    recognition_report = {
        "camera": "front",
        "frame_id": 12,
        "frame_timestamp_ms": 4_200,
        "source_frame_ref": "front:12:4200",
        "tracks_seen": 2,
        "tracks_eligible": 2,
        "tracks_candidates": 2,
        "candidate_track_ids": [7, 8],
        "tracks_queried": 2,
        "tracks_skipped_reason": {},
        "queried_track_ids": [7, 8],
        "attention_target_track_id": 7,
        "attention_target_only": False,
        "max_tracks_per_tick": 4,
        "query_interval_ms": 1_000,
        "event_cooldown_ms": 1_000,
        "recognition_runs_in_executor": True,
        "eligibility_policy": "class_name == 'person' and lost_ms == 0 and hits > 0",
    }

    class FakeMemoryService:
        def latest_recognition_report(
            self,
            camera: str | None = None,
        ) -> dict[str, Any]:
            assert camera == "front"
            return dict(recognition_report)

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.session_factory = SimpleNamespace(last_snapshot=None)
            self.client = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(memory_service=FakeMemoryService()),
                ),
            )

        def open_stream(self):
            return self

        def __enter__(self):
            return object()

        def __exit__(self, *_args: Any) -> None:
            return None

        def send(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "frame_id": 1,
                "tracks": [
                    {"track_id": 7, "class": "person", "lost_ms": 0},
                    {"track_id": 8, "class": "person", "lost_ms": 0},
                ],
                "attention": {"target_track_id": 7},
                "scene_context": {"engagement_state": "available"},
            }

    def fake_post_and_record_api_response(**kwargs: Any) -> dict[str, Any]:
        operation = kwargs["operation"]
        posted_operations.append(operation)
        if operation == "local_resolve_third_person_target":
            return {
                "status_code": 200,
                "body": {
                    "ok": True,
                    "status": "resolved",
                    "candidates": [
                        {"track_id": 8, "reason": "pose_pointing_to_person"}
                    ],
                    "evidence": {
                        "resolution_reason": "pose_pointing_to_person",
                        "resolver_target_ref": "front:track:8",
                        "introducer_ref": "front:track:7",
                        "pose_pointing_scoring": _pose_pointing_scoring(
                            score_margin=0.4
                        ),
                    },
                },
            }
        if operation == "local_teach_person_third_person":
            return {
                "status_code": 200,
                "body": {
                    "ok": True,
                    "person_id": "person_123",
                    "evidence": {
                        "resolution_reason": "pose_pointing_to_person",
                        "source_track_ref": "front:track:8",
                        "resolver_target_ref": "front:track:8",
                        "introducer_ref": "front:track:7",
                        "crop_hash": "crop-hash-123",
                        "crop_path_or_artifact_ref": (
                            "runtime/memory/artifacts/person_123.jpg"
                        ),
                        "person_visual_evidence": {
                            "source_frame_ref": "front:1:1000",
                            "source_bbox_xyxy": [30, 40, 140, 180],
                            "embedding_crop_path": (
                                "runtime/memory/artifacts/person_123.jpg"
                            ),
                            "face_detection": {
                                "coordinate_space": "crop",
                                "face_bbox_xyxy": [10, 12, 80, 90],
                                "score": 0.96,
                                "source": "local_embedding_scrfd",
                            },
                        },
                        "pose_pointing_scoring": _pose_pointing_scoring(
                            score_margin=0.5
                        ),
                    },
                },
            }
        raise AssertionError(f"unexpected operation: {operation}")

    def fake_replay(*_args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        replay_calls.append(kwargs)
        return [
            {
                "event": "known_person_present",
                "track_id": 8,
                "memory_context": {"person": {"person_id": "person_123"}},
            }
        ]

    monkeypatch.setattr(module, "LocalMemorySmokeRunner", FakeRunner)
    monkeypatch.setattr(
        module,
        "_local_third_person_source_frames",
        lambda _scene: [source_frame],
    )
    monkeypatch.setattr(
        module,
        "_post_and_record_api_response",
        fake_post_and_record_api_response,
    )
    monkeypatch.setattr(module, "_send_stable_query_and_drain_local", fake_replay)

    states_path = tmp_path / "visual_states.jsonl"
    with states_path.open("w", encoding="utf-8") as states_file:
        result = module._run_local_third_person_probe(
            out=tmp_path,
            scene=scene,
            record=record,
            camera="front",
            config=SimpleNamespace(),
            states_file=states_file,
            api_response_records=[],
        )

    assert posted_operations == [
        "local_resolve_third_person_target",
        "local_teach_person_third_person",
    ]
    assert replay_calls
    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["assertions"] == {
        "resolve_target_resolved": True,
        "resolve_target_pose_pointing": True,
        "teach_person_ok": True,
        "stored_embedding_source_is_target": True,
        "stored_embedding_source_not_introducer": True,
        "known_person_present": True,
        "known_person_context": True,
        "target_not_introducer": True,
        "pose_pointing_scoring_present": True,
        "pose_pointing_checks_passed": True,
    }
    assert result["pose_pointing_scoring"] == _pose_pointing_scoring(
        score_margin=0.5
    )
    assert result["person_id"] == "person_123"
    assert result["person_visual_evidence"]["face_detection"]["score"] == 0.96
    assert result["stored_embedding_source_track_ref"] == "front:track:8"
    assert result["stored_crop_hash"] == "crop-hash-123"
    assert (
        result["stored_crop_path_or_artifact_ref"]
        == "runtime/memory/artifacts/person_123.jpg"
    )
    assert result["debug_test_channel_enabled"] is False
    assert result["fixture_inputs_consumed"] == []
    assert result["debug_fixture_used_for_target_resolution"] is False
    assert result["bounded_multi_person_recognition"] == recognition_report
    assert result["resolve_target"]["evidence"]["introducer_ref"] == "front:track:7"
    assert result["selected_window"]["scene"] == "pic_teach_person"


def test_local_third_person_probe_scans_until_late_pose_pointing_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame_paths: list[Path] = []
    for index in range(18):
        frame_path = tmp_path / f"img_{index:03d}.jpeg"
        frame_path.write_bytes(_valid_jpeg_bytes())
        frame_paths.append(frame_path)
    scene = module.SceneDir(
        name="pic_teach_person",
        path=tmp_path,
        jpeg_paths=tuple(frame_paths),
        des_text="这是彭刚，请你记住",
    )
    record = {
        "scene": "pic_teach_person",
        "endpoint": "/v1/memory/teach/person",
        "payload": {
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "third_person_introduction",
                "referent_text": "这位/彭刚",
            },
            "profile": {"display_name": "彭刚"},
        },
    }
    posted_operations: list[str] = []
    resolve_indices: list[int] = []
    replay_frames: list[Path] = []

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.session_factory = SimpleNamespace(last_snapshot=None)
            self.client = object()

        def open_stream(self):
            return self

        def __enter__(self):
            return object()

        def __exit__(self, *_args: Any) -> None:
            return None

        def send(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "frame_id": 1,
                "tracks": [
                    {"track_id": 7, "class": "person", "lost_ms": 0},
                    {"track_id": 8, "class": "person", "lost_ms": 0},
                ],
                "attention": {"target_track_id": 7},
                "scene_context": {"engagement_state": "available"},
            }

    def fake_post_and_record_api_response(**kwargs: Any) -> dict[str, Any]:
        operation = kwargs["operation"]
        posted_operations.append(operation)
        if operation == "local_resolve_third_person_target":
            index = int(str(kwargs["payload_index"]).rsplit(":", 1)[-1])
            resolve_indices.append(index)
            if index < 16:
                return {
                    "status_code": 200,
                    "body": {
                        "ok": False,
                        "status": "target_unclear",
                        "reason": "target_unclear",
                    },
                }
            return {
                "status_code": 200,
                "body": {
                    "ok": True,
                    "status": "resolved",
                    "candidates": [
                        {"track_id": 8, "reason": "pose_pointing_to_person"}
                    ],
                    "evidence": {
                        "resolution_reason": "pose_pointing_to_person",
                        "resolver_target_ref": "front:track:8",
                        "introducer_ref": "front:track:7",
                        "pose_pointing_scoring": _pose_pointing_scoring(
                            score_margin=0.4
                        ),
                    },
                },
            }
        if operation == "local_teach_person_third_person":
            return {
                "status_code": 200,
                "body": {
                    "ok": True,
                    "person_id": "person_late",
                    "evidence": {
                        "resolution_reason": "pose_pointing_to_person",
                        "source_track_ref": "front:track:8",
                        "resolver_target_ref": "front:track:8",
                        "introducer_ref": "front:track:7",
                        "crop_hash": "crop-hash-late",
                        "crop_path_or_artifact_ref": (
                            "runtime/memory/artifacts/person_late.jpg"
                        ),
                        "pose_pointing_scoring": _pose_pointing_scoring(
                            score_margin=0.5
                        ),
                    },
                },
            }
        raise AssertionError(f"unexpected operation: {operation}")

    def fake_replay(*args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        source_frame = args[2]
        replay_frames.append(source_frame.path)
        return [
            {
                "event": "known_person_present",
                "track_id": 8,
                "memory_context": {"person": {"person_id": "person_late"}},
            }
        ]

    monkeypatch.setattr(module, "LocalMemorySmokeRunner", FakeRunner)
    monkeypatch.setattr(
        module,
        "_post_and_record_api_response",
        fake_post_and_record_api_response,
    )
    monkeypatch.setattr(module, "_send_stable_query_and_drain_local", fake_replay)

    states_path = tmp_path / "visual_states.jsonl"
    with states_path.open("w", encoding="utf-8") as states_file:
        result = module._run_local_third_person_probe(
            out=tmp_path,
            scene=scene,
            record=record,
            camera="front",
            config=SimpleNamespace(),
            states_file=states_file,
            api_response_records=[],
        )

    assert resolve_indices == list(range(17))
    assert posted_operations == [
        *["local_resolve_third_person_target"] * 17,
        "local_teach_person_third_person",
    ]
    assert replay_frames == [frame_paths[16]]
    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["person_id"] == "person_late"
    assert result["selected_window"] == {
        "scene": "pic_teach_person",
        "frame": str(frame_paths[16]),
        "frame_index": 16,
        "mode": "fixed_val_data_frame",
    }


def test_local_third_person_probe_does_not_teach_or_replay_without_resolved_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame_paths: list[Path] = []
    for index in range(18):
        frame_path = tmp_path / f"img_{index:03d}.jpeg"
        frame_path.write_bytes(_valid_jpeg_bytes())
        frame_paths.append(frame_path)
    scene = module.SceneDir(
        name="pic_teach_person",
        path=tmp_path,
        jpeg_paths=tuple(frame_paths),
        des_text="这是彭刚，请你记住",
    )
    record = {
        "scene": "pic_teach_person",
        "endpoint": "/v1/memory/teach/person",
        "payload": {
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "third_person_introduction",
                "referent_text": "这位/彭刚",
            },
            "profile": {"display_name": "彭刚"},
        },
    }
    resolve_indices: list[int] = []

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.session_factory = SimpleNamespace(last_snapshot=None)
            self.client = object()

        def open_stream(self):
            return self

        def __enter__(self):
            return object()

        def __exit__(self, *_args: Any) -> None:
            return None

        def send(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "frame_id": 1,
                "tracks": [{"track_id": 7, "class": "person", "lost_ms": 0}],
                "attention": {"target_track_id": 7},
                "scene_context": {"engagement_state": "available"},
            }

    def fake_post_and_record_api_response(**kwargs: Any) -> dict[str, Any]:
        operation = kwargs["operation"]
        if operation == "local_teach_person_third_person":
            raise AssertionError("teach_person should not be called")
        assert operation == "local_resolve_third_person_target"
        resolve_indices.append(int(str(kwargs["payload_index"]).rsplit(":", 1)[-1]))
        return {
            "status_code": 200,
            "body": {
                "ok": False,
                "status": "target_unclear",
                "reason": "target_unclear",
            },
        }

    def fake_replay(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("replay should not be called")

    monkeypatch.setattr(module, "LocalMemorySmokeRunner", FakeRunner)
    monkeypatch.setattr(
        module,
        "_post_and_record_api_response",
        fake_post_and_record_api_response,
    )
    monkeypatch.setattr(module, "_send_stable_query_and_drain_local", fake_replay)

    states_path = tmp_path / "visual_states.jsonl"
    with states_path.open("w", encoding="utf-8") as states_file:
        result = module._run_local_third_person_probe(
            out=tmp_path,
            scene=scene,
            record=record,
            camera="front",
            config=SimpleNamespace(),
            states_file=states_file,
            api_response_records=[],
        )

    assert resolve_indices == list(range(18))
    assert result["status"] == "insufficient_sample"
    assert result["passed"] is False
    assert result["reason"] == "target_unclear"
    assert result["selected_window"] is None
    assert result["events"] == []
    assert result["debug_test_channel_enabled"] is False
    assert result["fixture_inputs_consumed"] == []
    assert result["debug_fixture_used_for_target_resolution"] is False


def test_local_third_person_probe_failed_result_keeps_debug_fixture_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame_path = tmp_path / "img_000.jpeg"
    frame_path.write_bytes(_valid_jpeg_bytes())
    scene = module.SceneDir(
        name="pic_teach_person",
        path=tmp_path,
        jpeg_paths=(frame_path,),
        des_text="这是彭刚，请你记住",
    )
    source_frame = memory_e2e.SourceFrame(
        path=frame_path,
        jpeg_bytes=frame_path.read_bytes(),
        width=1280,
        height=720,
    )
    record = {
        "scene": "pic_teach_person",
        "endpoint": "/v1/memory/teach/person",
        "payload": {
            "camera": "front",
            "target": {
                "kind": "person",
                "intent": "third_person_introduction",
                "referent_text": "这位/彭刚",
            },
            "profile": {"display_name": "彭刚"},
        },
    }

    class FakeRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.session_factory = SimpleNamespace(last_snapshot=None)
            self.client = object()

        def open_stream(self):
            return self

        def __enter__(self):
            return object()

        def __exit__(self, *_args: Any) -> None:
            return None

        def send(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "frame_id": 1,
                "tracks": [
                    {"track_id": 7, "class": "person", "lost_ms": 0},
                    {"track_id": 8, "class": "person", "lost_ms": 0},
                ],
                "attention": {"target_track_id": 7},
                "scene_context": {"engagement_state": "available"},
            }

    def fake_post_and_record_api_response(**kwargs: Any) -> dict[str, Any]:
        if kwargs["operation"] == "local_teach_person_third_person":
            raise AssertionError("teach_person should not be called")
        return {
            "status_code": 200,
            "body": {
                "ok": True,
                "status": "resolved",
                "candidates": [{"track_id": 8, "reason": "active_interaction_target"}],
                "evidence": {
                    "resolution_reason": "active_interaction_target",
                    "resolver_target_ref": "front:track:8",
                    "introducer_ref": "front:track:7",
                },
            },
        }

    def fake_replay(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("replay should not be called")

    monkeypatch.setattr(module, "LocalMemorySmokeRunner", FakeRunner)
    monkeypatch.setattr(
        module,
        "_local_smoke_source_frames",
        lambda _scene: [source_frame],
    )
    monkeypatch.setattr(
        module,
        "_post_and_record_api_response",
        fake_post_and_record_api_response,
    )
    monkeypatch.setattr(module, "_send_stable_query_and_drain_local", fake_replay)

    states_path = tmp_path / "visual_states.jsonl"
    with states_path.open("w", encoding="utf-8") as states_file:
        result = module._run_local_third_person_probe(
            out=tmp_path,
            scene=scene,
            record=record,
            camera="front",
            config=SimpleNamespace(),
            states_file=states_file,
            api_response_records=[],
        )

    assert result["status"] == "failed"
    assert result["passed"] is False
    assert result["reason"] == "resolved_without_pose_pointing_evidence"
    assert result["debug_test_channel_enabled"] is False
    assert result["fixture_inputs_consumed"] == []
    assert result["debug_fixture_used_for_target_resolution"] is False
