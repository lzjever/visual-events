from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools import generate_memory_teaching_evidence as module


def _valid_jpeg_bytes(color: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (320, 180), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _assert_image_verifies(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()


def _pixel_rgb(path: Path, xy: tuple[int, int]) -> tuple[int, int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").getpixel(xy)


def _make_scene(root: Path, name: str, color: tuple[int, int, int]) -> Path:
    scene_dir = root / name
    scene_dir.mkdir(parents=True)
    (scene_dir / "img_000.jpeg").write_bytes(_valid_jpeg_bytes(color))
    return scene_dir / "img_000.jpeg"


def _write_artifact(
    artifact: Path,
    *,
    report_overrides: dict[str, Any] | None = None,
    include_identity_sidecars: bool = False,
) -> dict[str, Path]:
    artifact.mkdir(parents=True, exist_ok=True)
    data_dir = artifact.parent / "val-data"
    frame_paths = {
        "pic_teach_me": _make_scene(data_dir, "pic_teach_me", (196, 64, 64)),
        "pic_teach_person": _make_scene(data_dir, "pic_teach_person", (64, 180, 96)),
        "pic_teach_scene_galbot": _make_scene(
            data_dir,
            "pic_teach_scene_galbot",
            (220, 190, 72),
        ),
        "pic_teach_item_phone": _make_scene(
            data_dir,
            "pic_teach_item_phone",
            (140, 80, 196),
        ),
    }
    crop_dir = artifact / "runtime" / "memory" / "artifacts"
    crop_dir.mkdir(parents=True, exist_ok=True)
    (crop_dir / "self.jpg").write_bytes(_valid_jpeg_bytes((210, 80, 80)))
    (crop_dir / "third.jpg").write_bytes(_valid_jpeg_bytes((80, 190, 110)))

    self_result = {
        "status": "passed",
        "passed": True,
        "person_id": "person_self",
        "teach_crop_hash": "self_crop_hash",
        "teach_crop_path_or_artifact_ref": "runtime/memory/artifacts/self.jpg",
        "selected_window": {
            "scene": "pic_teach_me",
            "frame": str(frame_paths["pic_teach_me"]),
        },
        "person_visual_evidence": {
            "source_bbox_xyxy": [34, 24, 120, 160],
            "crop_box_xyxy": [34, 24, 120, 160],
            "embedding_crop_path": "runtime/memory/artifacts/self.jpg",
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
        "person_id": "person_third",
        "resolver_target_ref": "front:track:8",
        "introducer_ref": "front:track:7",
        "stored_embedding_source_track_ref": "front:track:8",
        "stored_crop_hash": "third_crop_hash",
        "stored_crop_path_or_artifact_ref": "runtime/memory/artifacts/third.jpg",
        "selected_window": {
            "scene": "pic_teach_person",
            "frame": str(frame_paths["pic_teach_person"]),
        },
        "pose_pointing_scoring": {
            "candidate_scores": [
                {
                    "track_id": 8,
                    "score": 0.93,
                    "ray_intersects_bbox": True,
                    "perpendicular_distance": 12.5,
                }
            ]
        },
        "resolve_target": {
            "status": "resolved",
            "evidence": {
                "request_snapshot_ref": "memory_frame:front:42:1000",
                "source_frame_ref": "front:42:1000",
                "pose_visual_evidence": {
                    "introducer_ref": "front:track:7",
                    "introducer_bbox_xyxy": [20, 30, 130, 170],
                    "target_ref": "front:track:8",
                    "target_bbox_xyxy": [180, 35, 300, 175],
                    "arm_side": "left",
                    "shoulder_xy": [82, 65],
                    "elbow_xy": [130, 78],
                    "wrist_xy": [170, 88],
                    "ray_start_xy": [170, 88],
                    "ray_end_xy": [304, 124],
                    "candidate_scores": [
                        {
                            "track_id": 8,
                            "score": 0.93,
                            "ray_intersects_bbox": True,
                            "perpendicular_distance": 12.5,
                        }
                    ],
                    "pose_stability_window": {"selected_count": 2},
                },
            },
            "candidates": [
                {"track_id": 8, "bbox_xyxy": [180, 35, 300, 175]},
            ],
        },
    }
    scene_result = {
        "status": "passed",
        "passed": True,
        "scene_id": "scene_galbot",
        "teach_crop_hash": "scene_crop_hash",
        "selected_window": {
            "scene": "pic_teach_scene_galbot",
            "frame": str(frame_paths["pic_teach_scene_galbot"]),
        },
        "events": [
            {
                "event": "scene_activated",
                "event_id": "evt_scene",
                "evidence": {"memory_match_id": "match_scene"},
            }
        ],
    }
    object_result = {
        "status": "passed",
        "passed": True,
        "selected_window": {
            "scene": "pic_teach_item_phone",
            "frame": str(frame_paths["pic_teach_item_phone"]),
        },
        "resolve_target": {
            "status": "not_found",
            "error_code": "unsupported_target_kind",
        },
        "store_delta": {
            "before": {"person_embedding_vectors": 2},
            "after": {"person_embedding_vectors": 2},
            "delta": {"person_embedding_vectors": 0},
        },
    }
    report = {
        "ok": True,
        "status": "passed",
        "gate": "memory_teaching_ga_runner_local_smoke",
        "mode": "local-smoke",
        "backend": "local",
        "real_model_evidence": True,
        "scene_count": len(frame_paths),
        "scenes": [
            {
                "name": name,
                "path": str(path.parent),
                "frame_count": 1,
                "first_frame": str(path),
            }
            for name, path in frame_paths.items()
        ],
        "self_smoke": self_result,
        "third_person_probe": third_person_result,
        "scene_smoke": scene_result,
        "object_no_write": object_result,
        "post_teach_scene_replay": {
            "passed": True,
            "replayed_scene_count": len(frame_paths),
            "scenes": [{"scene": name, "flags": {}} for name in sorted(frame_paths)],
        },
        "checks": [],
        "artifacts": {"report_json": "report.json"},
    }
    if report_overrides:
        report.update(report_overrides)
    (artifact / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact / "teach_payloads.json").write_text(
        json.dumps({"schema_version": 1, "payloads": []}),
        encoding="utf-8",
    )
    if include_identity_sidecars:
        _write_identity_sidecars(artifact)
    return frame_paths


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_identity_sidecars(artifact: Path) -> None:
    identity = {
        "status": "known_person",
        "source": "cache",
        "person": {
            "person_id": "person_identity",
            "display_name": "张三",
            "description": "店长",
            "tags": ["staff"],
        },
    }
    visual_state = {
        "type": "visual_state",
        "camera": "front",
        "frame_id": 1,
        "frame_timestamp_ms": 1_000,
        "identity_context": {
            "overlay_status": "ready",
            "active_target": {"track_id": 7},
            "tracks": [{"track_id": 7, "identity": identity}],
        },
        "semantic_events": [
            {
                "type": "semantic_event",
                "event": "person_waving",
                "track_id": 7,
                "identity_context": identity,
            }
        ],
    }
    _write_jsonl(artifact / "visual_states.jsonl", [{"response": visual_state}])
    _write_jsonl(
        artifact / "api_responses.jsonl",
        [
            {
                "endpoint": "/v1/memory/identify-current",
                "operation": "supporting_identify_current",
                "status_code": 200,
                "response": {
                    "ok": True,
                    "status": "identified",
                    "people": [
                        {
                            "target_ref": "current:front:active_target",
                            "identity_context": identity,
                        }
                    ],
                },
            }
        ],
    )
    _write_jsonl(
        artifact / "botified_frames.jsonl",
        [
            {
                "source": "cli_frame_pump_stdout",
                "event": "person_waving",
                "payload": {
                    "request": (
                        'event=person_waving visual_context={"visual_context":'
                        '{"identity_context":{"status":"known_person","source":"cache",'
                        '"person":{"display_name":"张三","person_id":"person_identity"}}}}'
                    )
                },
            }
        ],
    )
    (artifact / "current_visual_snapshot.json").write_text(
        json.dumps(
            {
                "type": "current_visual_snapshot",
                "camera": "front",
                "active_target_ref": "current:front:person:0",
                "people": [
                    {
                        "target_ref": "current:front:person:0",
                        "identity_context": identity,
                    }
                ],
                "events": [
                    {
                        "event": "person_waving",
                        "target_ref": "current:front:person:0",
                        "identity_context": identity,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _rewrite_report_frame_paths_to_cwd_relative(
    artifact: Path,
    frame_paths: dict[str, Path],
) -> None:
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for scene in report["scenes"]:
        scene_name = scene["name"]
        scene["path"] = f"val-data/{scene_name}"
        scene["first_frame"] = f"val-data/{scene_name}/img_000.jpeg"
    frame_by_section = {
        "self_smoke": "pic_teach_me",
        "third_person_probe": "pic_teach_person",
        "scene_smoke": "pic_teach_scene_galbot",
        "object_no_write": "pic_teach_item_phone",
    }
    for section, scene_name in frame_by_section.items():
        report[section]["selected_window"]["frame"] = (
            f"val-data/{scene_name}/{frame_paths[scene_name].name}"
        )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_offline_renderer_generates_html_json_images_and_crop_previews(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    out = tmp_path / "evidence"

    exit_code = module.main(
        ["--artifact", str(artifact), "--out", str(out)],
    )

    assert exit_code == 0
    for relative_path in [
        "index.html",
        "source-artifact.json",
        "visual_evidence_index.json",
        "visual-evidence/index.html",
        "visual-evidence/self-introduction-known-person.jpg",
        "visual-evidence/third-person-pose-pointing.jpg",
        "visual-evidence/teach-scene-scene-activated.jpg",
        "visual-evidence/object-unsupported-no-write.jpg",
    ]:
        assert (out / relative_path).is_file()

    for relative_path in [
        "visual-evidence/self-introduction-known-person.jpg",
        "visual-evidence/third-person-pose-pointing.jpg",
        "visual-evidence/teach-scene-scene-activated.jpg",
        "visual-evidence/object-unsupported-no-write.jpg",
        "visual-evidence/crops/self.jpg",
        "visual-evidence/crops/third.jpg",
    ]:
        _assert_image_verifies(out / relative_path)

    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    by_assertion = {item["assertion_id"]: item for item in index}
    assert by_assertion["self_introduction_known_person"]["status"] == "present"
    assert by_assertion["self_introduction_known_person"][
        "face_detection"
    ] == "not_recorded"
    assert by_assertion["third_person_pose_pointing"]["candidate_score"] == 0.93
    assert by_assertion["third_person_pose_pointing"][
        "perpendicular_distance"
    ] == 12.5
    assert by_assertion["third_person_pose_pointing"]["ray_intersects_bbox"] is True
    assert by_assertion["post_teach_full_scene_replay"]["status"] == "present"

    root_html = (out / "index.html").read_text(encoding="utf-8")
    visual_html = (out / "visual-evidence" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "source gate: passed" in root_html
    assert "real_model_evidence" in root_html
    assert "third-person-pose-pointing.jpg" in visual_html
    assert "face_detection" in visual_html


def test_offline_renderer_prefers_payload_source_image_over_scene_first_frame(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    scene_dir = frame_paths["pic_teach_scene_galbot"].parent
    source_image = scene_dir / "teach.jpeg"
    source_text = scene_dir / "teach.transcript"
    source_image.write_bytes(_valid_jpeg_bytes((12, 34, 210)))
    source_text.write_text("这是银河通用的办公室，请你记住", encoding="utf-8")

    payload_record = {
        "scene": "pic_teach_scene_galbot",
        "source_text_path": str(source_text),
        "source_image_path": str(source_image),
        "transcript_text": "这是银河通用的办公室，请你记住",
        "endpoint": "/v1/memory/teach/scene",
        "payload": {
            "camera": "front",
            "target": {"kind": "scene", "intent": "teach_scene"},
        },
        "expected": {"writes_memory": True, "memory_type": "scene"},
    }
    (artifact / "teach_payloads.json").write_text(
        json.dumps({"schema_version": 1, "payloads": [payload_record]}),
        encoding="utf-8",
    )

    out = tmp_path / "evidence"
    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code == 0
    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    scene_item = {
        item["assertion_id"]: item for item in index
    }["teach_scene_scene_activated"]
    assert scene_item["source_frame"] == str(source_image)
    assert scene_item["selected_frame"] == str(
        frame_paths["pic_teach_scene_galbot"]
    )
    assert scene_item["source_image_path"] == str(source_image)
    assert scene_item["source_text_path"] == str(source_text)
    assert scene_item["transcript_source"] == str(source_text)
    bottom_pixel = _pixel_rgb(
        out / "visual-evidence" / "teach-scene-scene-activated.jpg",
        (20, 170),
    )
    assert bottom_pixel[2] > 150
    assert bottom_pixel[0] < 80

    visual_html = (out / "visual-evidence" / "index.html").read_text(
        encoding="utf-8"
    )
    assert str(source_text) in visual_html
    assert "source_text_path" in visual_html


def test_offline_renderer_summarizes_identity_sidecars(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, include_identity_sidecars=True)
    out = tmp_path / "evidence"

    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code == 0
    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    by_assertion = {item["assertion_id"]: item for item in index}
    for assertion_id in [
        "visual_state_identity_overlay",
        "event_identity_context",
        "identify_current_response",
        "cli_current_visual_snapshot",
    ]:
        assert by_assertion[assertion_id]["status"] == "present"
        assert by_assertion[assertion_id]["kind"] == "identity_summary"

    assert by_assertion["visual_state_identity_overlay"]["display_name"] == "张三"
    assert by_assertion["visual_state_identity_overlay"]["person_id"] == "person_identity"
    assert by_assertion["event_identity_context"]["identity_status"] == "known_person"
    assert by_assertion["event_identity_context"]["identity_source"] == "cache"
    assert by_assertion["event_identity_context"]["source_label"] == "botified_frames.jsonl"
    assert by_assertion["event_identity_context"]["source_path"] == "botified_frames.jsonl"
    assert by_assertion["identify_current_response"]["identify_status"] == "identified"
    assert by_assertion["cli_current_visual_snapshot"]["target_ref"] == (
        "current:front:person:0"
    )
    assert (
        by_assertion["cli_current_visual_snapshot"]["forbidden_fields_absent"] is True
    )
    serialized_index = json.dumps(index, ensure_ascii=False)
    assert "张三" in serialized_index
    assert "current:front:person:0" in serialized_index
    html = (out / "visual-evidence" / "index.html").read_text(encoding="utf-8")
    assert "visual_state_identity_overlay" in html
    assert "identified" in html


def test_offline_renderer_prefers_richer_identity_overlay_state(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    out = tmp_path / "evidence"
    pending_state = {
        "type": "visual_state",
        "camera": "front",
        "frame_id": 1,
        "identity_context": {
            "overlay_status": "ready",
            "tracks": [
                {
                    "track_id": 7,
                    "identity": {"status": "pending", "source": "none"},
                }
            ],
        },
    }
    known_state = {
        "type": "visual_state",
        "camera": "front",
        "frame_id": 2,
        "identity_context": {
            "overlay_status": "ready",
            "tracks": [
                {
                    "track_id": 7,
                    "identity": {
                        "status": "known_person",
                        "source": "cache",
                        "person": {
                            "person_id": "person_identity",
                            "display_name": "张三",
                        },
                    },
                }
            ],
        },
    }
    _write_jsonl(
        artifact / "visual_states.jsonl",
        [{"visual_state": pending_state}, {"visual_state": known_state}],
    )

    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code == 0
    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    by_assertion = {item["assertion_id"]: item for item in index}
    overlay = by_assertion["visual_state_identity_overlay"]
    assert overlay["status"] == "present"
    assert overlay["identity_status"] == "known_person"
    assert overlay["display_name"] == "张三"
    assert by_assertion["event_identity_context"]["status"] == "not_present"


def test_offline_renderer_labels_visual_state_event_identity_fallback_source(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    out = tmp_path / "evidence"
    identity = {
        "status": "known_person",
        "source": "cache",
        "person": {
            "person_id": "person_identity",
            "display_name": "张三",
        },
    }
    visual_state = {
        "type": "visual_state",
        "camera": "front",
        "semantic_events": [
            {
                "type": "semantic_event",
                "event": "person_waving",
                "track_id": 7,
                "identity_context": identity,
            }
        ],
    }
    _write_jsonl(artifact / "visual_states.jsonl", [{"visual_state": visual_state}])

    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code == 0
    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    item = {
        item["assertion_id"]: item for item in index
    }["event_identity_context"]
    assert item["status"] == "present"
    assert item["source_label"] == "visual_states.jsonl"
    assert item["source_path"] == "visual_states.jsonl"
    assert item["report_section"] == "visual_states.jsonl.semantic_events"


def test_offline_renderer_marks_identity_sidecars_not_present_when_missing(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    out = tmp_path / "evidence"

    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code == 0
    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    by_assertion = {item["assertion_id"]: item for item in index}
    assert by_assertion["visual_state_identity_overlay"]["status"] == "not_present"
    assert by_assertion["event_identity_context"]["status"] == "not_present"
    assert by_assertion["identify_current_response"]["status"] == "not_present"
    assert by_assertion["cli_current_visual_snapshot"]["status"] == "not_present"
    assert (out / "index.html").is_file()


def test_offline_renderer_resolves_cwd_relative_val_data_source_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    _rewrite_report_frame_paths_to_cwd_relative(artifact, frame_paths)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "evidence"

    exit_code = module.main(
        ["--artifact", "artifact", "--out", str(out)],
    )

    assert exit_code == 0
    for relative_path in [
        "visual-evidence/self-introduction-known-person.jpg",
        "visual-evidence/third-person-pose-pointing.jpg",
        "visual-evidence/teach-scene-scene-activated.jpg",
    ]:
        _assert_image_verifies(out / relative_path)
    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    by_assertion = {item["assertion_id"]: item for item in index}
    assert by_assertion["self_introduction_known_person"]["source_frame"] == str(
        tmp_path / "val-data" / "pic_teach_me" / "img_000.jpeg"
    )
    root_html = (out / "index.html").read_text(encoding="utf-8")
    assert "artifact-relative first, then current working directory" in root_html


def test_offline_renderer_uses_promoted_person_visual_evidence_for_face_bbox(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    face_detection = {
        "coordinate_space": "crop",
        "face_bbox_xyxy": [20, 24, 86, 110],
        "score": 0.98,
        "source": "local_embedding_scrfd",
    }
    report["self_smoke"]["person_visual_evidence"][
        "crop_box_coordinate_space"
    ] = "source_frame"
    report["self_smoke"]["person_visual_evidence"]["face_detection"] = face_detection
    report["third_person_probe"]["person_visual_evidence"] = {
        "source_frame_ref": "front:42:1000",
        "source_bbox_xyxy": [180, 35, 300, 175],
        "embedding_crop_path": "runtime/memory/artifacts/third.jpg",
        "face_detection": {
            "coordinate_space": "source_frame",
            "face_bbox_xyxy": [190, 45, 240, 100],
            "score": 0.97,
            "source": "local_embedding_scrfd",
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    exit_code = module.main(
        ["--artifact", str(artifact), "--out", str(tmp_path / "evidence")],
    )

    assert exit_code == 0
    index = json.loads(
        (tmp_path / "evidence" / "visual_evidence_index.json").read_text(
            encoding="utf-8"
        )
    )
    by_assertion = {item["assertion_id"]: item for item in index}
    assert by_assertion["self_introduction_known_person"][
        "face_detection"
    ] == "recorded"
    assert by_assertion["self_introduction_known_person"][
        "face_bbox_xyxy"
    ] == [54.0, 48.0, 120.0, 134.0]
    assert by_assertion["third_person_pose_pointing"][
        "face_detection"
    ] == "recorded"
    assert by_assertion["third_person_pose_pointing"][
        "face_bbox_xyxy"
    ] == [190.0, 45.0, 240.0, 100.0]
    assert by_assertion["third_person_pose_pointing"][
        "person_visual_evidence"
    ] == "present"
    _assert_image_verifies(
        tmp_path
        / "evidence"
        / "visual-evidence"
        / "self-introduction-known-person.jpg"
    )
    _assert_image_verifies(
        tmp_path
        / "evidence"
        / "visual-evidence"
        / "third-person-pose-pointing.jpg"
    )


def test_source_report_failure_exits_zero_by_default_and_nonzero_when_strict(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(
        artifact,
        report_overrides={
            "ok": False,
            "status": "failed",
            "self_smoke": {"status": "failed", "passed": False, "reason": "bad"},
            "third_person_probe": {
                "status": "failed",
                "passed": False,
                "reason": "bad",
            },
            "scene_smoke": {"status": "failed", "passed": False, "reason": "bad"},
            "object_no_write": {"status": "not_present", "passed": False},
            "post_teach_scene_replay": {"status": "not_present", "passed": False},
        },
    )

    assert module.main(["--artifact", str(artifact), "--out", str(tmp_path / "out")]) == 0
    assert (
        module.main(
            [
                "--artifact",
                str(artifact),
                "--out",
                str(tmp_path / "strict-out"),
                "--strict-source-ok",
            ]
        )
        != 0
    )
    html = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert "source gate: failed" in html
    assert "renderer status: ok" in html


def test_out_inside_data_dir_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    data_dir = tmp_path / "val-data"

    exit_code = module.main(
        [
            "--artifact",
            str(artifact),
            "--data-dir",
            str(data_dir),
            "--out",
            str(data_dir / "evidence"),
        ]
    )

    assert exit_code != 0
    assert not (data_dir / "evidence" / "index.html").exists()


def test_missing_report_fails_without_creating_success_index(tmp_path: Path) -> None:
    artifact = tmp_path / "missing-report-artifact"
    artifact.mkdir()
    out = tmp_path / "evidence"

    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code != 0
    assert not (out / "index.html").exists()


def test_run_local_smoke_delegates_to_existing_runner_then_renders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_local_smoke(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        _write_artifact(kwargs["out"])
        return json.loads((kwargs["out"] / "report.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(module.runner, "run_local_smoke", fake_run_local_smoke)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    out = tmp_path / "evidence"

    exit_code = module.main(
        [
            "--run-local-smoke",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--camera",
            "front",
            "--embedding-backend",
            "local",
            "--person-model-path",
            str(tmp_path / "person-model"),
            "--scene-model-path",
            str(tmp_path / "scene-model"),
            "--inference-backend",
            "ultralytics",
            "--pose-model-path",
            str(tmp_path / "pose.pt"),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["out"] == out / "runner-artifact"
    assert calls[0]["data_dir"] == data_dir
    assert calls[0]["embedding_backend"] == "local"
    assert calls[0]["inference_backend"] == "ultralytics"
    assert (out / "runner-artifact" / "report.json").is_file()
    assert (out / "index.html").is_file()
    source = json.loads((out / "source-artifact.json").read_text(encoding="utf-8"))
    assert source["artifact_path"] == str((out / "runner-artifact").resolve())
