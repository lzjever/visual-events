from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools import generate_memory_teaching_evidence as module
from tools import memory_teaching_evidence


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
                "endpoint": "/v1/memory/teach/person",
                "operation": "supporting_teach_auto_merge_anonymous",
                "status_code": 200,
                "response": {
                    "ok": True,
                    "outcome": "merged_anonymous_person",
                    "person_id": "person_auto_merge",
                    "merged_anonymous_id": "anon_identity",
                    "copied_embedding_count": 2,
                    "identity_context": {
                        "status": "known_person",
                        "source": "teach_auto_merge",
                        "person": {
                            "person_id": "person_auto_merge",
                            "display_name": "张三",
                            "description": "店长",
                        },
                    },
                },
            },
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
    assert "Actual posted payload summary" in visual_html
    assert "<code>api_responses.jsonl</code>: not_present" in visual_html


def test_public_renderer_uses_demo_rows_without_internal_debug(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    out = tmp_path / "public"

    summary = memory_teaching_evidence.render_memory_teaching_evidence(
        artifact=artifact,
        out=out,
        public_demo=True,
    )

    root_html = (out / "index.html").read_text(encoding="utf-8")
    visual_html = (out / "visual-evidence" / "index.html").read_text(
        encoding="utf-8"
    )
    public_text = "\n".join([root_html, visual_html])
    public_index = summary["visual_evidence_index"]

    assert [item["id"] for item in public_index] == [
        "self_introduction",
        "scene_teaching",
        "pointing_teaching",
    ]
    assert not (out / "visual_evidence_index.json").exists()
    assert not (out / "visual-evidence" / "crops").exists()
    public_file_names = [path.name for path in out.rglob("*") if path.is_file()]
    assert all("person_" not in name for name in public_file_names)
    assert all(re.search(r"[0-9a-f]{12,}", name) is None for name in public_file_names)
    assert "记住自我介绍" in public_text
    assert "第三人称指向示教" in public_text
    assert "记住场景" in public_text
    for forbidden_text in [
        "Debug JSON",
        "assertion_id",
        "person_id",
        "event_id",
        "crop_hash",
        "bbox_xyxy",
        "request_snapshot_ref",
        "resolver_target_ref",
        "memory_match_id",
        "embedding_id",
        "crop_path_or_artifact_ref",
        "Source Report",
        "source gate",
        "local-smoke",
        "unsupported",
        "not_present",
    ]:
        assert forbidden_text not in public_text


def test_public_renderer_shows_familiar_unknown_representative_frame(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["familiar_unknown"] = {
        "status": "passed",
        "passed": True,
        "selected_window": {
            "scene": memory_teaching_evidence.FAMILIAR_UNKNOWN_SCENE,
            "frame": str(frame_paths["pic_teach_person"]),
        },
        "seen_count": 3,
        "observed_duration_ms": 4200,
        "familiar_score": 0.87,
        "person_visual_evidence": {
            "source_bbox_xyxy": [40, 28, 150, 170],
            "source_bbox_coordinate_space": "source_frame",
            "face_detection": {
                "coordinate_space": "source_frame",
                "face_bbox_xyxy": [70, 42, 118, 96],
            },
        },
        "events": [
            {
                "event": "familiar_unknown_present",
                "event_id": "evt_familiar",
                "evidence": {"memory_match_id": "match_familiar"},
                "memory_context": {
                    "anonymous_person": {
                        "anonymous_id": "anon_familiar",
                        "seen_count": 3,
                    }
                },
            }
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out = tmp_path / "public"

    summary = memory_teaching_evidence.render_memory_teaching_evidence(
        artifact=artifact,
        out=out,
        public_demo=True,
    )

    public_index = summary["visual_evidence_index"]
    familiar_item = {
        item["id"]: item for item in public_index
    }["familiar_unknown"]
    assert familiar_item["status"] == "passed"
    assert familiar_item["path"] == "visual-evidence/familiar-unknown-present.jpg"
    assert familiar_item["image"] == "visual-evidence/familiar-unknown-present.jpg"
    assert familiar_item["selected_evidence_frame"] == str(
        frame_paths["pic_teach_person"]
    )
    assert familiar_item["metrics"] == {
        "seen_count": 3,
        "observed_duration_ms": 4200,
        "familiar_score": 0.87,
        "face_detection": "recorded",
        "person_box_available": True,
        "face_box_available": True,
    }
    _assert_image_verifies(out / familiar_item["image"])
    visual_html = (out / "visual-evidence" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "匿名熟客出现" in visual_html
    assert "familiar-unknown-present.jpg" in visual_html
    assert "seen_count" in visual_html
    assert '<img class="thumb"' in visual_html


def test_public_renderer_prefers_familiar_direct_evidence_over_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["familiar_unknown"] = {
        "status": "passed",
        "passed": True,
        "selected_window": {
            "scene": memory_teaching_evidence.FAMILIAR_UNKNOWN_SCENE,
            "frame": str(frame_paths["pic_teach_person"]),
        },
        "seen_count": 11,
        "observed_duration_ms": 10_000,
        "familiar_score": 1.0,
        "person_visual_evidence": {
            "source_bbox_xyxy": [796, 112, 939, 590],
            "source_bbox_coordinate_space": "source_frame",
        },
        "events": [
            {
                "event": "familiar_unknown_present",
                "event_id": "evt_familiar",
                "track_id": 9,
            }
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(
        artifact / "visual_states.jsonl",
        [
            {
                "source_frame": str(frame_paths["pic_teach_person"]),
                "visual_state": {
                    "semantic_events": [
                        {
                            "event": "familiar_unknown_present",
                            "event_id": "evt_familiar",
                            "track_id": 9,
                        }
                    ],
                    "tracks": [
                        {"track_id": 9, "bbox_xyxy": [20, 20, 300, 170]},
                    ],
                },
            }
        ],
    )
    overlay_calls: list[dict[str, Any]] = []

    def fake_write_image_overlay(**kwargs: Any) -> bool:
        overlay_calls.append(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(_valid_jpeg_bytes())
        return True

    monkeypatch.setattr(
        memory_teaching_evidence,
        "write_image_overlay",
        fake_write_image_overlay,
    )

    summary = memory_teaching_evidence.render_memory_teaching_evidence(
        artifact=artifact,
        out=tmp_path / "public",
        public_demo=True,
    )

    familiar_item = {
        item["id"]: item for item in summary["visual_evidence_index"]
    }["familiar_unknown"]
    assert familiar_item["metrics"]["person_box_available"] is True
    familiar_call = next(
        call
        for call in overlay_calls
        if call["output_path"].name == "familiar-unknown-present.jpg"
    )
    assert familiar_call["boxes"][0]["bbox_xyxy"] == [796.0, 112.0, 939.0, 590.0]


def test_public_renderer_does_not_cross_scene_fallback_for_familiar_bbox(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["familiar_unknown"] = {
        "status": "passed",
        "passed": True,
        "selected_window": {
            "scene": memory_teaching_evidence.FAMILIAR_UNKNOWN_SCENE,
            "frame": str(frame_paths["pic_teach_person"]),
        },
        "seen_count": 11,
        "observed_duration_ms": 10_000,
        "familiar_score": 1.0,
        "events": [
            {
                "event": "familiar_unknown_present",
                "event_id": "evt_familiar",
                "track_id": 9,
            }
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(
        artifact / "visual_states.jsonl",
        [
            {
                "source_frame": str(frame_paths["pic_teach_me"]),
                "visual_state": {
                    "semantic_events": [
                        {
                            "event": "familiar_unknown_present",
                            "event_id": "evt_familiar",
                            "track_id": 9,
                        }
                    ],
                    "tracks": [
                        {"track_id": 9, "bbox_xyxy": [20, 20, 300, 170]},
                    ],
                },
            }
        ],
    )
    overlay_calls: list[dict[str, Any]] = []

    def fake_write_image_overlay(**kwargs: Any) -> bool:
        overlay_calls.append(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(_valid_jpeg_bytes())
        return True

    monkeypatch.setattr(
        memory_teaching_evidence,
        "write_image_overlay",
        fake_write_image_overlay,
    )

    summary = memory_teaching_evidence.render_memory_teaching_evidence(
        artifact=artifact,
        out=tmp_path / "public",
        public_demo=True,
    )

    familiar_item = {
        item["id"]: item for item in summary["visual_evidence_index"]
    }["familiar_unknown"]
    assert familiar_item["metrics"]["person_box_available"] is False
    familiar_call = next(
        call
        for call in overlay_calls
        if call["output_path"].name == "familiar-unknown-present.jpg"
    )
    assert familiar_call["boxes"] == []
    assert "person box not available" in familiar_call["lines"]


def test_public_renderer_generates_third_person_failure_evidence_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    alternate = frame_paths["pic_teach_person"].parent / "img_001.jpeg"
    alternate.write_bytes(_valid_jpeg_bytes((90, 160, 210)))
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    report["third_person_probe"] = {
        "status": "failed",
        "passed": False,
        "reason": "multiple_candidates",
        "user_message": "这位是王工",
        "source_image_path": str(frame_paths["pic_teach_person"]),
        "anchor_frame": str(frame_paths["pic_teach_person"]),
        "candidate_frames": [str(frame_paths["pic_teach_person"]), str(alternate)],
        "selected_window": {
            "scene": "pic_teach_person",
            "frame": str(alternate),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact / "api_responses.jsonl").write_text(
        json.dumps(
            {
                "scene": "pic_teach_person",
                "endpoint": "/v1/memory/resolve-target",
                "operation": "local_resolve_third_person_target",
                "response": {
                    "status": "ambiguous",
                    "evidence": {
                        "request_snapshot_ref": "memory_frame:front:9:1800",
                        "source_frame_ref": "front:9:1800",
                        "pose_visual_evidence": {
                            "introducer_ref": "front:track:7",
                            "introducer_bbox_xyxy": [20, 20, 80, 120],
                            "arm_side": "right",
                            "shoulder_xy": [62, 50],
                            "elbow_xy": [90, 62],
                            "wrist_xy": [112, 72],
                            "ray_start_xy": [112, 72],
                            "ray_end_xy": [210, 94],
                            "candidate_scores": [
                                {
                                    "track_id": 8,
                                    "bbox_xyxy": [120, 20, 180, 120],
                                    "score": 0.61,
                                    "ray_intersects_bbox": True,
                                },
                                {
                                    "track_id": 9,
                                    "bbox_xyxy": [205, 24, 270, 126],
                                    "score": 0.58,
                                    "ray_intersects_bbox": True,
                                },
                            ],
                        },
                    },
                    "candidates": [
                        {"track_id": 8, "bbox_xyxy": [120, 20, 180, 120]},
                        {"track_id": 9, "bbox_xyxy": [205, 24, 270, 126]},
                    ],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "public"
    overlay_calls: list[dict[str, Any]] = []

    def fake_write_image_overlay(**kwargs: Any) -> bool:
        overlay_calls.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_valid_jpeg_bytes())
        return True

    monkeypatch.setattr(
        memory_teaching_evidence,
        "write_image_overlay",
        fake_write_image_overlay,
    )

    summary = memory_teaching_evidence.render_memory_teaching_evidence(
        artifact=artifact,
        out=out,
        public_demo=True,
    )

    public_index = summary["visual_evidence_index"]
    pointing_item = {
        item["id"]: item for item in public_index
    }["pointing_teaching"]
    assert pointing_item["status"] == "failed"
    assert pointing_item["image"] == (
        "visual-evidence/third-person-pose-pointing-failure.jpg"
    )
    assert "未绑定" in pointing_item["summary"]
    assert pointing_item["metrics"]["candidate_count"] == 2
    assert pointing_item["metrics"]["candidate_box_available"] is True
    assert pointing_item["metrics"]["ray_evidence_available"] is True
    _assert_image_verifies(out / pointing_item["image"])
    pointing_call = next(
        call
        for call in overlay_calls
        if call["output_path"].name == "third-person-pose-pointing-failure.jpg"
    )
    labels = [
        box["label"]
        for box in pointing_call["boxes"]
        if isinstance(box, dict) and "label" in box
    ]
    assert "introducer" in labels
    assert any("candidate 1 score 0.61" in label for label in labels)
    assert any("candidate 2 score 0.58" in label for label in labels)
    assert pointing_call["arrows"]
    visual_html = (out / "visual-evidence" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "第三人称指向示教" in visual_html
    assert "third-person-pose-pointing-failure.jpg" in visual_html
    assert "目标不明确" in visual_html
    for forbidden_text in ("track_id", "bbox_xyxy", "keypoints", "embedding"):
        assert forbidden_text not in visual_html


def test_offline_renderer_marks_familiar_unknown_missing_frame_without_image(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    missing_frame = tmp_path / "missing-familiar.jpeg"
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["familiar_unknown"] = {
        "status": "passed",
        "passed": True,
        "selected_window": {
            "scene": memory_teaching_evidence.FAMILIAR_UNKNOWN_SCENE,
            "frame": str(missing_frame),
        },
        "seen_count": 3,
        "events": [{"event": "familiar_unknown_present"}],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out = tmp_path / "evidence"

    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code == 0
    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    familiar_item = {
        item["assertion_id"]: item for item in index
    }["familiar_unknown_present"]
    assert familiar_item["status"] == "missing_source_frame"
    assert familiar_item["path"] is None
    assert familiar_item["selected_evidence_frame"] == str(missing_frame)
    assert not (out / "visual-evidence" / "familiar-unknown-present.jpg").exists()


def test_public_renderer_uses_product_friendly_overlay_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["familiar_unknown"] = {
        "status": "passed",
        "passed": True,
        "selected_window": {
            "scene": memory_teaching_evidence.FAMILIAR_UNKNOWN_SCENE,
            "frame": str(frame_paths["pic_teach_person"]),
        },
        "seen_count": 11,
        "observed_duration_ms": 10_000,
        "familiar_score": 1.0,
        "person_visual_evidence": {
            "source_bbox_xyxy": [40, 28, 150, 170],
            "source_bbox_coordinate_space": "source_frame",
            "face_detection": {
                "coordinate_space": "source_frame",
                "face_bbox_xyxy": [70, 42, 118, 96],
            },
        },
        "events": [{"event": "familiar_unknown_present"}],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out = tmp_path / "public"
    overlay_calls: list[dict[str, Any]] = []

    def fake_write_image_overlay(**kwargs: Any) -> bool:
        overlay_calls.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_valid_jpeg_bytes())
        return True

    monkeypatch.setattr(
        memory_teaching_evidence,
        "write_image_overlay",
        fake_write_image_overlay,
    )

    memory_teaching_evidence.render_memory_teaching_evidence(
        artifact=artifact,
        out=out,
        public_demo=True,
    )

    assert {call["output_path"].name for call in overlay_calls} == {
        "self-introduction-known-person.jpg",
        "third-person-pose-pointing.jpg",
        "teach-scene-scene-activated.jpg",
        "familiar-unknown-present.jpg",
    }
    overlay_text = json.dumps(
        [
            {
                "title": call["title"],
                "lines": call["lines"],
                "labels": [
                    box.get("label")
                    for box in call.get("boxes") or []
                    if isinstance(box, dict)
                ],
            }
            for call in overlay_calls
        ],
        ensure_ascii=False,
    )
    for expected_label in ("person", "face", "target", "introducer"):
        assert expected_label in overlay_text
    for forbidden_text in (
        "person_id",
        "event_id",
        "crop_hash",
        "resolver_target_ref",
        "front:track",
        "memory_match_id",
        "embedding_id",
    ):
        assert forbidden_text not in overlay_text
    pointing_call = next(
        call
        for call in overlay_calls
        if call["output_path"].name == "third-person-pose-pointing.jpg"
    )
    assert pointing_call["arrows"]
    assert pointing_call["points"]
    familiar_call = next(
        call
        for call in overlay_calls
        if call["output_path"].name == "familiar-unknown-present.jpg"
    )
    assert familiar_call["title"] == "熟悉的未命名人物"
    assert familiar_call["lines"] == [
        "见过 11 次",
        "累计 10s",
        "score 1.0",
        "person box available",
        "face: recorded",
    ]
    familiar_labels = [box["label"] for box in familiar_call["boxes"]]
    assert familiar_labels == ["familiar person", "face"]


def test_offline_renderer_separates_transcript_templates_from_actual_posts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    (artifact / "teach_payloads.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "payloads": [
                    {
                        "scene": "pic_teach_me",
                        "endpoint": "/v1/memory/teach/person",
                        "payload": {
                            "camera": "front",
                            "stream_ref": "ws_payload_fixture",
                            "target": {
                                "kind": "person",
                                "intent": "self_introduction",
                            },
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        artifact / "api_responses.jsonl",
        [
            {
                "scene": "pic_teach_me",
                "endpoint": "/v1/memory/teach/person",
                "operation": "local_teach_person_self",
                "status_code": 200,
                "payload": {
                    "camera": "front",
                    "stream_ref": "ws_runtime_self",
                },
                "response": {
                    "ok": True,
                    "status": "stored",
                    "outcome": "created_person",
                },
            },
            {
                "scene": "pic_teach_item_phone",
                "endpoint": "/v1/memory/resolve-target",
                "operation": "resolve_object_unsupported",
                "status_code": 200,
                "payload": {
                    "camera": "front",
                    "stream_ref": "ws_runtime_object",
                },
                "response": {
                    "ok": False,
                    "status": "not_found",
                    "error_code": "unsupported_target_kind",
                },
            },
        ],
    )
    out = tmp_path / "evidence"

    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code == 0
    visual_html = (out / "visual-evidence" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "Teach Payloads" not in visual_html
    assert "Transcript request templates" in visual_html
    assert "not posted as-is" in visual_html
    assert "ws_payload_fixture" in visual_html
    assert "placeholder/template stream ref" in visual_html
    assert "Actual posted payload summary" in visual_html
    for label in [
        "scene",
        "operation",
        "endpoint",
        "stream_ref",
        "status_code",
        "response status",
        "response outcome",
    ]:
        assert label in visual_html
    assert "pic_teach_me" in visual_html
    assert "local_teach_person_self" in visual_html
    assert "/v1/memory/teach/person" in visual_html
    assert "ws_runtime_self" in visual_html
    assert "200" in visual_html
    assert "stored" in visual_html
    assert "created_person" in visual_html
    assert "pic_teach_item_phone" in visual_html
    assert "resolve_object_unsupported" in visual_html
    assert "/v1/memory/resolve-target" in visual_html
    assert "ws_runtime_object" in visual_html
    assert "not_found" in visual_html


def test_offline_renderer_uses_selected_evidence_frame_for_overlay_and_keeps_reference_image(
    tmp_path: Path,
    monkeypatch,
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
    overlay_calls: list[dict[str, Any]] = []

    def fake_write_image_overlay(**kwargs: Any) -> bool:
        overlay_calls.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_valid_jpeg_bytes())
        return True

    monkeypatch.setattr(
        memory_teaching_evidence,
        "write_image_overlay",
        fake_write_image_overlay,
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
    assert scene_item["source_frame"] == str(frame_paths["pic_teach_scene_galbot"])
    assert scene_item["overlay_source_frame"] == str(
        frame_paths["pic_teach_scene_galbot"]
    )
    assert scene_item["selected_frame"] == str(
        frame_paths["pic_teach_scene_galbot"]
    )
    assert scene_item["selected_evidence_frame"] == str(
        frame_paths["pic_teach_scene_galbot"]
    )
    assert scene_item["source_image_path"] == str(source_image)
    assert scene_item["reference_source_image"] == str(source_image)
    assert scene_item["transcript_source_image"] == str(source_image)
    assert scene_item["source_text_path"] == str(source_text)
    assert scene_item["transcript_source"] == str(source_text)
    assert scene_item["frame_relation"] == "different"
    scene_call = next(
        call
        for call in overlay_calls
        if call["output_path"].name == "teach-scene-scene-activated.jpg"
    )
    assert scene_call["source_path"] == frame_paths["pic_teach_scene_galbot"]

    visual_html = (out / "visual-evidence" / "index.html").read_text(
        encoding="utf-8"
    )
    assert str(source_text) in visual_html
    assert "source_text_path" in visual_html


def test_offline_renderer_does_not_fallback_for_success_without_evidence_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    reference_image = frame_paths["pic_teach_me"].parent / "teach.jpeg"
    reference_image.write_bytes(_valid_jpeg_bytes((12, 34, 210)))

    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["self_smoke"]["selected_window"] = {"scene": "pic_teach_me"}
    report["self_smoke"]["source_image_path"] = str(reference_image)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    overlay_calls: list[dict[str, Any]] = []

    def fake_write_image_overlay(**kwargs: Any) -> bool:
        overlay_calls.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_valid_jpeg_bytes())
        return True

    monkeypatch.setattr(
        memory_teaching_evidence,
        "write_image_overlay",
        fake_write_image_overlay,
    )

    exit_code = module.main(
        ["--artifact", str(artifact), "--out", str(tmp_path / "out")]
    )

    assert exit_code == 0
    assert all(
        call["output_path"].name != "self-introduction-known-person.jpg"
        for call in overlay_calls
    )
    index = json.loads(
        (tmp_path / "out" / "visual_evidence_index.json").read_text(
            encoding="utf-8"
        )
    )
    item = {
        item["assertion_id"]: item for item in index
    }["self_introduction_known_person"]
    assert item["status"] == "missing_source_frame"
    assert item["selected_evidence_frame"] is None
    assert item["source_frame"] is None
    assert item["overlay_source_frame"] is None
    assert item["reference_source_image"] == str(reference_image)
    assert item["path"] is None


def test_offline_renderer_rejects_mismatched_explicit_overlay_source_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    scene_dir = frame_paths["pic_teach_person"].parent
    reference_image = scene_dir / "teach.jpeg"
    explicit_overlay = scene_dir / "overlay.jpeg"
    reference_image.write_bytes(_valid_jpeg_bytes((12, 34, 210)))
    explicit_overlay.write_bytes(_valid_jpeg_bytes((210, 180, 40)))

    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["third_person_probe"]["overlay_source_frame"] = str(explicit_overlay)
    report["third_person_probe"]["source_image_path"] = str(reference_image)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    overlay_calls: list[dict[str, Any]] = []

    def fake_write_image_overlay(**kwargs: Any) -> bool:
        overlay_calls.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_valid_jpeg_bytes())
        return True

    monkeypatch.setattr(
        memory_teaching_evidence,
        "write_image_overlay",
        fake_write_image_overlay,
    )

    exit_code = module.main(
        ["--artifact", str(artifact), "--out", str(tmp_path / "out")]
    )

    assert exit_code == 0
    assert all(
        call["output_path"].name != "third-person-pose-pointing.jpg"
        for call in overlay_calls
    )
    index = json.loads(
        (tmp_path / "out" / "visual_evidence_index.json").read_text(
            encoding="utf-8"
        )
    )
    item = {
        item["assertion_id"]: item for item in index
    }["third_person_pose_pointing"]
    assert item["status"] == "mismatched_overlay_source_frame"
    assert item["source_frame"] is None
    assert item["overlay_source_frame"] is None
    assert item["requested_overlay_source_frame"] == str(explicit_overlay)
    assert item["selected_evidence_frame"] == str(frame_paths["pic_teach_person"])
    assert item["reference_source_image"] == str(reference_image)
    assert item["path"] is None
    assert "does not match selected evidence frame" in item["reason"]


def test_offline_renderer_accepts_matching_explicit_overlay_source_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    frame_paths = _write_artifact(artifact)
    reference_image = frame_paths["pic_teach_person"].parent / "teach.jpeg"
    reference_image.write_bytes(_valid_jpeg_bytes((12, 34, 210)))

    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["third_person_probe"]["overlay_source_frame"] = str(
        frame_paths["pic_teach_person"]
    )
    report["third_person_probe"]["source_image_path"] = str(reference_image)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    overlay_calls: list[dict[str, Any]] = []

    def fake_write_image_overlay(**kwargs: Any) -> bool:
        overlay_calls.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_valid_jpeg_bytes())
        return True

    monkeypatch.setattr(
        memory_teaching_evidence,
        "write_image_overlay",
        fake_write_image_overlay,
    )

    exit_code = module.main(
        ["--artifact", str(artifact), "--out", str(tmp_path / "out")]
    )

    assert exit_code == 0
    third_call = next(
        call
        for call in overlay_calls
        if call["output_path"].name == "third-person-pose-pointing.jpg"
    )
    assert third_call["source_path"] == frame_paths["pic_teach_person"]
    index = json.loads(
        (tmp_path / "out" / "visual_evidence_index.json").read_text(
            encoding="utf-8"
        )
    )
    item = {
        item["assertion_id"]: item for item in index
    }["third_person_pose_pointing"]
    assert item["status"] == "present"
    assert item["source_frame"] == str(frame_paths["pic_teach_person"])
    assert item["overlay_source_frame"] == str(frame_paths["pic_teach_person"])
    assert item["selected_evidence_frame"] == str(frame_paths["pic_teach_person"])
    assert item["reference_source_image"] == str(reference_image)


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
        "teach_auto_merge_anonymous",
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
    auto_merge = by_assertion["teach_auto_merge_anonymous"]
    assert auto_merge["outcome"] == "merged_anonymous_person"
    assert auto_merge["person_id"] == "person_auto_merge"
    assert auto_merge["merged_anonymous_id"] == "anon_identity"
    assert auto_merge["copied_embedding_count"] == 2
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


def test_offline_renderer_flags_current_snapshot_center_uv_as_forbidden(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, include_identity_sidecars=True)
    snapshot_path = artifact / "current_visual_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["people"][0]["center_uv"] = [420.0, 360.0]
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out = tmp_path / "evidence"

    exit_code = module.main(["--artifact", str(artifact), "--out", str(out)])

    assert exit_code == 0
    index = json.loads(
        (out / "visual_evidence_index.json").read_text(encoding="utf-8")
    )
    by_assertion = {item["assertion_id"]: item for item in index}
    assert (
        by_assertion["cli_current_visual_snapshot"]["forbidden_fields_absent"]
        is False
    )


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
    assert by_assertion["teach_auto_merge_anonymous"]["status"] == "not_present"
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
