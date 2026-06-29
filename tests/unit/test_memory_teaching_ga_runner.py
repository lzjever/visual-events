from __future__ import annotations

import asyncio
from io import BytesIO
import json
from pathlib import Path
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


def _valid_jpeg_bytes() -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (1280, 720), color=(128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _records_by_scene(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["scene"]: record for record in records}


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
        "target": {
            "kind": "person",
            "intent": "self_introduction",
            "referent_text": "我",
        },
        "profile": {"display_name": "小李飞刀"},
    }
    assert by_scene["pic_teach_person"]["payload"] == {
        "camera": "front",
        "target": {
            "kind": "person",
            "intent": "third_person_introduction",
            "referent_text": "这位/彭刚",
        },
        "profile": {"display_name": "彭刚"},
    }
    assert by_scene["pic_teach_scene_galbot"]["payload"] == {
        "camera": "front",
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


def test_memory_e2e_runner_generates_public_rest_payloads_without_low_level_fields() -> None:
    payloads = [
        memory_e2e.self_introduction_payload(
            camera="front",
            display_name="小李飞刀",
            description="public self introduction",
            tags=["memory-e2e"],
        ),
        memory_e2e.third_person_introduction_payload(
            camera="front",
            display_name="彭刚",
            referent_text="这位/彭刚",
        ),
        memory_e2e.teach_scene_payload(
            camera="front",
            title="银河通用办公室",
            description="public whole-scene teaching",
            activation_hint="office",
        ),
        memory_e2e.object_resolve_payload(camera="front", referent_text="手机"),
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
    jpeg_bytes = _valid_jpeg_bytes()
    for name in ["pci_stand", "pic_hello"]:
        _make_scene(data_dir, name, frames=2, jpeg_bytes=jpeg_bytes)
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
    _make_scene(
        data_dir,
        "pic_teach_item_phone",
        des_text="这是手机，请你记住",
        jpeg_bytes=jpeg_bytes,
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
    assert all(
        fields == [] for fields in report["forbidden_agent_payload_fields"].values()
    )
    assert report["object_no_write"]["assertions"]["no_memory_write"] is True

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["all_scenes_replayed"]["passed"] is True
    assert checks["actual_api_responses"]["passed"] is True
    assert checks["self_introduction_known_person_present"]["passed"] is True
    assert checks["third_person_known_person_present"]["passed"] is True
    assert checks["teach_scene_scene_activated"]["passed"] is True
    assert checks["object_resolve_unsupported_no_write"]["passed"] is True

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
    event_names = {
        frame["semantic_event"]["event"]
        for frame in botified_frames
        if frame.get("semantic_event")
    }
    assert {"known_person_present", "scene_activated"} <= event_names


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

    for relative_path in [
        "timeline.jsonl",
        "teach_payloads.json",
        "api_responses.jsonl",
        "botified_frames.jsonl",
        "visual-evidence/index.html",
    ]:
        assert (out / relative_path).is_file()


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


def test_local_smoke_report_distinguishes_pass_fail_insufficient_without_models(
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

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["mode"] == "local-smoke"
    assert report["real_model_evidence"] is True
    assert report["self_smoke"]["status"] == "passed"
    assert report["scene_smoke"]["status"] == "passed"
    assert report["third_person_probe"]["status"] == "insufficient_sample"
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
    assert checks["third_person_local_probe"]["passed"] is True
    assert checks["third_person_local_probe"]["details"]["status"] == "insufficient_sample"
