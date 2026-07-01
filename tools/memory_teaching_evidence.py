from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JPEG_SUFFIXES = {".jpeg", ".jpg"}
FAMILIAR_UNKNOWN_SCENE = "pic_familiar_face"
SNAPSHOT_FORBIDDEN_FIELDS = (
    "track_id",
    "bbox",
    "bbox_xyxy",
    "center_uv",
    "keypoints",
    "embedding",
    "crop",
    "crop_ref",
    "stream_ref",
    "raw_track_id",
    "source_frame",
    "request_snapshot_ref",
)
IDENTITY_STATUS_PRIORITY = {
    "known_person": 5,
    "familiar_unknown": 4,
    "unavailable": 3,
    "unknown": 2,
    "pending": 1,
}


class MemoryTeachingEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceScene:
    name: str
    path: Path
    jpeg_paths: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        return len(self.jpeg_paths)


def render_memory_teaching_evidence(
    *,
    artifact: Path,
    out: Path,
    public_demo: bool = False,
) -> dict[str, Any]:
    artifact = Path(artifact).resolve()
    out = Path(out)
    report = load_source_report(artifact)

    out.mkdir(parents=True, exist_ok=True)
    visual_evidence_dir = out / "visual-evidence"
    if visual_evidence_dir.exists():
        shutil.rmtree(visual_evidence_dir)
    visual_evidence_dir.mkdir(parents=True, exist_ok=True)
    if not public_demo:
        (visual_evidence_dir / "crops").mkdir(parents=True, exist_ok=True)

    scenes = scenes_from_report(report, artifact=artifact)
    payload_records = load_payload_records(artifact)
    actual_posted_payload_summary = load_actual_posted_payload_summary(artifact)
    manifest = report.get("manifest") if isinstance(report.get("manifest"), dict) else {}
    source_summary = source_report_summary(report, artifact=artifact)

    visual_evidence_index = build_artifact_visual_evidence_index(
        artifact=artifact,
        out=out,
        scenes=scenes,
        report=report,
        payload_records=payload_records,
        public_demo=public_demo,
    )
    render_failures = [
        item
        for item in visual_evidence_index
        if item.get("status") in {"missing_source_frame", "image_not_generated"}
    ]
    if render_failures:
        failures = ", ".join(
            f"{item.get('assertion_id')}:{item.get('status')}"
            for item in render_failures
        )
        raise MemoryTeachingEvidenceError(f"visual evidence render failed: {failures}")
    public_visual_evidence_index = (
        _public_demo_visual_evidence_index(visual_evidence_index)
        if public_demo
        else visual_evidence_index
    )
    public_source_summary = (
        public_demo_source_summary(source_summary) if public_demo else source_summary
    )
    write_visual_evidence_index(
        visual_evidence_dir / "index.html",
        scenes=scenes,
        payload_records=payload_records,
        manifest=manifest,
        mode=str(report.get("mode") or "unknown"),
        visual_evidence_index=public_visual_evidence_index,
        source_summary=public_source_summary,
        actual_posted_payload_summary=actual_posted_payload_summary,
        public_demo=public_demo,
    )
    if not public_demo:
        _write_json(out / "visual_evidence_index.json", public_visual_evidence_index)
    if not public_demo:
        _write_json(
            out / "source-artifact.json",
            {
                "schema_version": 1,
                "artifact_path": str(artifact),
                "report_path": str((artifact / "report.json").resolve()),
                "source_gate": source_summary,
                "source_artifacts": report.get("artifacts", {}),
                "source_report": report,
            },
        )
    (out / "index.html").write_text(
        (
            render_public_demo_root_index_html(
                source_summary=public_source_summary,
                visual_evidence_index=public_visual_evidence_index,
            )
            if public_demo
            else render_root_index_html(
                source_summary=source_summary,
                visual_evidence_index=public_visual_evidence_index,
            )
        ),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "out": str(out),
        "index_html": str(out / "index.html"),
        "source_report_ok": bool(report.get("ok")),
        "source_status": _source_gate_status(report),
        "source_report_path": str(artifact / "report.json"),
        "visual_evidence_index": public_visual_evidence_index,
        "report_path": str(out / "report.json"),
    }


def load_source_report(artifact: Path) -> dict[str, Any]:
    report_path = Path(artifact) / "report.json"
    if not report_path.is_file():
        raise MemoryTeachingEvidenceError(f"source artifact report.json not found: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MemoryTeachingEvidenceError(
            f"source artifact report.json is invalid JSON: {report_path}:{exc.lineno}:{exc.colno}"
        ) from exc
    if not isinstance(report, dict):
        raise MemoryTeachingEvidenceError(f"source artifact report.json must be an object: {report_path}")
    return report


def scenes_from_report(report: dict[str, Any], *, artifact: Path) -> list[EvidenceScene]:
    scenes = report.get("scenes")
    if not isinstance(scenes, list):
        return []

    result: list[EvidenceScene] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        name = scene.get("name")
        if not isinstance(name, str) or not name:
            continue
        scene_path = _resolve_optional_report_path(artifact, scene.get("path"))
        first_frame = _resolve_optional_report_path(artifact, scene.get("first_frame"))
        last_frame = _resolve_optional_report_path(artifact, scene.get("last_frame"))
        jpeg_paths: list[Path] = []
        for path in (first_frame, last_frame):
            if path is not None and path.suffix.lower() in JPEG_SUFFIXES and path not in jpeg_paths:
                jpeg_paths.append(path)
        if not jpeg_paths and scene_path is not None and scene_path.is_dir():
            jpeg_paths = sorted(
                path
                for path in scene_path.iterdir()
                if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES
            )
        result.append(
            EvidenceScene(
                name=name,
                path=scene_path or Path(name),
                jpeg_paths=tuple(jpeg_paths),
            )
        )
    return result


def load_payload_records(artifact: Path) -> list[dict[str, Any]]:
    payload_path = Path(artifact) / "teach_payloads.json"
    if not payload_path.is_file():
        return []
    try:
        payloads = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    records = payloads.get("payloads") if isinstance(payloads, dict) else None
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def load_actual_posted_payload_summary(artifact: Path) -> dict[str, Any]:
    path = Path(artifact) / "api_responses.jsonl"
    if not path.is_file():
        return {
            "status": "not_present",
            "source_path": "api_responses.jsonl",
            "records": [],
        }
    records = [
        _actual_posted_payload_summary_record(record)
        for record in _read_jsonl_objects(path)
        if _is_actual_post_record(record)
    ]
    return {
        "status": "present" if records else "empty",
        "source_path": "api_responses.jsonl",
        "records": records,
    }


def build_artifact_visual_evidence_index(
    *,
    artifact: Path,
    out: Path,
    scenes: list[EvidenceScene],
    report: dict[str, Any],
    payload_records: list[dict[str, Any]] | None = None,
    public_demo: bool = False,
) -> list[dict[str, Any]]:
    mode = str(report.get("mode") or "")
    scene_by_name = {scene.name: scene for scene in scenes}
    payload_by_scene = _payload_records_by_scene(payload_records or [])
    items: list[dict[str, Any]] = [
        {
            "assertion_id": "memory_teaching_evidence_index",
            "kind": "html_index",
            "path": "visual-evidence/index.html",
            "status": "present",
        }
    ]

    items.append(
        _build_self_visual_item(
            out=out,
            artifact=artifact,
            scene=scene_by_name.get("pic_teach_me"),
            result=_with_payload_source_fields(
                _self_result_from_report(report),
                payload_by_scene.get("pic_teach_me"),
            ),
            report_section="self_smoke" if mode == "local-smoke" else "checks.self_introduction_known_person_present.details",
            include_not_present=True,
            public_demo=public_demo,
        )
    )
    items.append(
        _build_third_person_visual_item(
            out=out,
            artifact=artifact,
            scene=scene_by_name.get("pic_teach_person"),
            result=_with_payload_source_fields(
                _third_person_result_from_report(report),
                payload_by_scene.get("pic_teach_person"),
            ),
            report_section="third_person_probe" if mode == "local-smoke" else "third_person_introduction",
            include_not_present=True,
            public_demo=public_demo,
        )
    )
    items.append(
        _build_scene_visual_item(
            out=out,
            artifact=artifact,
            scene=scene_by_name.get("pic_teach_scene_galbot"),
            result=_with_payload_source_fields(
                _scene_result_from_report(report),
                payload_by_scene.get("pic_teach_scene_galbot"),
            ),
            report_section="scene_smoke" if mode == "local-smoke" else "checks.teach_scene_scene_activated.details",
            include_not_present=True,
            public_demo=public_demo,
        )
    )
    items.append(_build_familiar_unknown_summary_item(report))
    if public_demo:
        return items
    items.append(
        _build_object_visual_item(
            out=out,
            artifact=artifact,
            scene=scene_by_name.get("pic_teach_item_phone"),
            result=_with_payload_source_fields(
                _object_result_from_report(report),
                payload_by_scene.get("pic_teach_item_phone"),
            ),
            include_not_present=True,
        )
    )
    items.append(_build_full_replay_summary_item(report))
    items.extend(_build_identity_summary_items(artifact))
    return items


def build_runner_visual_overlay_items(
    *,
    out: Path,
    scenes: list[Any],
    mode: str,
    self_result: dict[str, Any] | None,
    third_person_result: dict[str, Any] | None,
    scene_result: dict[str, Any] | None,
    object_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    scene_by_name = {scene.name: scene for scene in scenes}
    artifact = Path(out)
    visual_evidence_dir = Path(out) / "visual-evidence"
    visual_evidence_dir.mkdir(parents=True, exist_ok=True)

    for item in (
        _build_self_visual_item(
            out=out,
            artifact=artifact,
            scene=scene_by_name.get("pic_teach_me"),
            result=self_result or {},
            report_section=(
                "self_smoke"
                if mode == "local-smoke"
                else "checks.self_introduction_known_person_present.details"
            ),
        ),
        _build_third_person_visual_item(
            out=out,
            artifact=artifact,
            scene=scene_by_name.get("pic_teach_person"),
            result=third_person_result or {},
            report_section=(
                "third_person_probe"
                if mode == "local-smoke"
                else "third_person_introduction"
            ),
        ),
        _build_scene_visual_item(
            out=out,
            artifact=artifact,
            scene=scene_by_name.get("pic_teach_scene_galbot"),
            result=scene_result or {},
            report_section=(
                "scene_smoke"
                if mode == "local-smoke"
                else "checks.teach_scene_scene_activated.details"
            ),
        ),
        _build_object_visual_item(
            out=out,
            artifact=artifact,
            scene=scene_by_name.get("pic_teach_item_phone"),
            result=object_result or {},
        ),
    ):
        if item is not None:
            items.append(item)
    return items


def _build_self_visual_item(
    *,
    out: Path,
    artifact: Path,
    scene: Any | None,
    result: dict[str, Any] | None,
    report_section: str,
    include_not_present: bool = False,
    public_demo: bool = False,
) -> dict[str, Any] | None:
    result = result if isinstance(result, dict) else {}
    base = {
        "assertion_id": "self_introduction_known_person",
        "scene": "pic_teach_me",
        "report_section": report_section,
    }
    not_present = _not_present_item(base, result=result, scene=scene)
    if not_present is not None:
        return not_present if include_not_present else None

    source_path = _overlay_source_frame_path(scene, result, artifact=artifact)
    if source_path is None or not source_path.is_file():
        return _status_item(
            base,
            "missing_source_frame",
            "source frame not present",
            source=result,
        ) if include_not_present else None

    event = _first_compact_event(result, "known_person_present")
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    person_visual_evidence = _find_nested_dict(result, "person_visual_evidence")
    crop_source = _first_path_value(
        person_visual_evidence.get("embedding_crop_path") if person_visual_evidence else None,
        result.get("teach_crop_path_or_artifact_ref"),
    )
    crop_preview = (
        None
        if public_demo
        else _copy_crop_preview(
            artifact=artifact,
            out=out,
            source=crop_source,
            fallback_name="self-person-crop.jpg",
        )
    )
    source_bbox = _bbox_from_visual_evidence(
        person_visual_evidence,
        "source_bbox_xyxy",
        "crop_box_xyxy",
    )
    face_detection = (
        person_visual_evidence.get("face_detection")
        if isinstance(person_visual_evidence, dict)
        and isinstance(person_visual_evidence.get("face_detection"), dict)
        else None
    )
    face_detection_status = "recorded" if face_detection else "not_recorded"
    face_bbox_xyxy = _face_bbox_source_frame(person_visual_evidence)

    item = {
        **base,
        **_result_source_fields(result),
        "kind": "image_overlay",
        "status": "present",
        "path": "visual-evidence/self-introduction-known-person.jpg",
        "person_id": result.get("person_id"),
        "event_id": event.get("event_id"),
        "memory_match_id": evidence.get("memory_match_id"),
        "crop_hash": result.get("teach_crop_hash"),
        "crop_path_or_artifact_ref": result.get("teach_crop_path_or_artifact_ref"),
        "crop_preview_path": crop_preview,
        "face_detection": face_detection_status,
        "face_bbox_xyxy": face_bbox_xyxy,
        "person_visual_evidence": "present" if person_visual_evidence else "fallback",
        "target_bbox_xyxy": source_bbox,
        "selected_frame": _selected_frame_path(result),
        "source_frame": str(source_path),
    }
    boxes = (
        [
            {
                "label": "person" if public_demo else "person target",
                "bbox_xyxy": source_bbox,
                "color": (0, 190, 255),
            }
        ]
        if source_bbox is not None
        else []
    )
    if face_bbox_xyxy is not None:
        boxes.append(
            {
                "label": "face",
                "bbox_xyxy": face_bbox_xyxy,
                "color": (255, 120, 220),
            }
        )
    ok = write_image_overlay(
        source_path=source_path,
        output_path=Path(out) / item["path"],
        title=(
            "Self introduction remembered"
            if public_demo
            else "Self introduction / known person"
        ),
        lines=(
            [
                "person: remembered",
                f"face: {face_detection_status}",
            ]
            if public_demo
            else [
                f"person_id: {_short_text(item.get('person_id'))}",
                f"event_id: {_short_text(item.get('event_id'))}",
                f"crop_hash: {_short_text(item.get('crop_hash'))}",
                f"face_detection: {face_detection_status}",
            ]
        ),
        boxes=boxes,
    )
    return item if ok else (_status_item(
        base,
        "image_not_generated",
        "overlay image could not be written",
        source=result,
    ) if include_not_present else None)


def _build_third_person_visual_item(
    *,
    out: Path,
    artifact: Path,
    scene: Any | None,
    result: dict[str, Any] | None,
    report_section: str,
    include_not_present: bool = False,
    public_demo: bool = False,
) -> dict[str, Any] | None:
    result = result if isinstance(result, dict) else {}
    base = {
        "assertion_id": "third_person_pose_pointing",
        "scene": "pic_teach_person",
        "report_section": report_section,
    }
    not_present = _not_present_item(base, result=result, scene=scene)
    if not_present is not None:
        return not_present if include_not_present else None

    source_path = _overlay_source_frame_path(scene, result, artifact=artifact)
    if source_path is None or not source_path.is_file():
        return _status_item(
            base,
            "missing_source_frame",
            "source frame not present",
            source=result,
        ) if include_not_present else None

    resolve_target = result.get("resolve_target") if isinstance(result.get("resolve_target"), dict) else {}
    resolve_evidence = response_evidence(resolve_target)
    pose_visual_evidence = _find_nested_dict(result, "pose_visual_evidence")
    person_visual_evidence = _find_nested_dict(result, "person_visual_evidence")
    target_bbox_xyxy = _bbox_from_visual_evidence(
        pose_visual_evidence,
        "target_bbox_xyxy",
    ) or _first_resolve_candidate_bbox(resolve_target)
    introducer_bbox_xyxy = _bbox_from_visual_evidence(
        pose_visual_evidence,
        "introducer_bbox_xyxy",
    )
    pose_stability_window = (
        pose_visual_evidence.get("pose_stability_window")
        if isinstance(pose_visual_evidence, dict)
        and isinstance(pose_visual_evidence.get("pose_stability_window"), dict)
        else resolve_evidence.get("pose_stability_window")
    )
    if not isinstance(pose_stability_window, dict):
        pose_stability_window = {}
    scoring = result.get("pose_pointing_scoring") if isinstance(result.get("pose_pointing_scoring"), dict) else {}
    candidate_score = _candidate_score_for_target(
        pose_visual_evidence if isinstance(pose_visual_evidence, dict) else scoring,
        _track_id_from_ref(result.get("resolver_target_ref")),
    )
    candidate_metrics = _candidate_metrics_for_target(
        pose_visual_evidence if isinstance(pose_visual_evidence, dict) else scoring,
        _track_id_from_ref(result.get("resolver_target_ref")),
    )
    crop_source = _first_path_value(
        person_visual_evidence.get("embedding_crop_path")
        if person_visual_evidence
        else None,
        result.get("stored_crop_path_or_artifact_ref"),
    )
    crop_preview = (
        None
        if public_demo
        else _copy_crop_preview(
            artifact=artifact,
            out=out,
            source=crop_source,
            fallback_name="third-person-target-crop.jpg",
        )
    )
    face_detection = (
        person_visual_evidence.get("face_detection")
        if isinstance(person_visual_evidence, dict)
        and isinstance(person_visual_evidence.get("face_detection"), dict)
        else None
    )
    face_detection_status = "recorded" if face_detection else "not_recorded"
    face_bbox_xyxy = _face_bbox_source_frame(person_visual_evidence)
    item = {
        **base,
        **_result_source_fields(result),
        "kind": "image_overlay",
        "status": "present",
        "path": "visual-evidence/third-person-pose-pointing.jpg",
        "resolver_target_ref": result.get("resolver_target_ref")
        or (pose_visual_evidence or {}).get("target_ref"),
        "introducer_ref": result.get("introducer_ref")
        or (pose_visual_evidence or {}).get("introducer_ref"),
        "stored_embedding_source_track_ref": result.get("stored_embedding_source_track_ref"),
        "request_snapshot_ref": resolve_evidence.get("request_snapshot_ref")
        or (pose_visual_evidence or {}).get("request_snapshot_ref"),
        "source_frame_ref": resolve_evidence.get("source_frame_ref")
        or (pose_visual_evidence or {}).get("source_frame_ref"),
        "pose_visual_evidence": "present" if pose_visual_evidence else "fallback",
        "person_visual_evidence": (
            "present" if person_visual_evidence else "fallback"
        ),
        "pose_stability_window": _pose_stability_window_summary(pose_stability_window),
        "candidate_score": candidate_score,
        "ray_intersects_bbox": candidate_metrics.get("ray_intersects_bbox"),
        "perpendicular_distance": candidate_metrics.get("perpendicular_distance"),
        "target_bbox_xyxy": target_bbox_xyxy,
        "introducer_bbox_xyxy": introducer_bbox_xyxy,
        "arm_side": (pose_visual_evidence or {}).get("arm_side") if isinstance(pose_visual_evidence, dict) else None,
        "crop_hash": result.get("stored_crop_hash"),
        "crop_path_or_artifact_ref": result.get("stored_crop_path_or_artifact_ref"),
        "crop_preview_path": crop_preview,
        "face_detection": face_detection_status,
        "face_bbox_xyxy": face_bbox_xyxy,
        "selected_frame": _selected_frame_path(result),
        "source_frame": str(source_path),
    }

    boxes = []
    if target_bbox_xyxy is not None:
        boxes.append(
            {
                "label": (
                    "target"
                    if public_demo
                    else f"target {_short_text(item.get('resolver_target_ref'), max_len=24)}"
                ),
                "bbox_xyxy": target_bbox_xyxy,
                "color": (0, 190, 255),
            }
        )
    if introducer_bbox_xyxy is not None:
        boxes.append(
            {
                "label": (
                    "introducer"
                    if public_demo
                    else f"introducer {_short_text(item.get('introducer_ref'), max_len=20)}"
                ),
                "bbox_xyxy": introducer_bbox_xyxy,
                "color": (90, 220, 120),
            }
        )
    if face_bbox_xyxy is not None:
        boxes.append(
            {
                "label": "face",
                "bbox_xyxy": face_bbox_xyxy,
                "color": (255, 120, 220),
            }
        )
    arrows = _pose_arrows(pose_visual_evidence)
    points = _pose_points(pose_visual_evidence)
    ok = write_image_overlay(
        source_path=source_path,
        output_path=Path(out) / item["path"],
        title=(
            "Third-person pointing evidence"
            if public_demo
            else "Third-person pose pointing"
        ),
        lines=(
            [
                "pointing evidence: present",
                "target: selected",
                "introducer: present",
                f"score: {_short_text(candidate_score)}",
                f"arm: {_short_text(item.get('arm_side'))}",
                f"face: {face_detection_status}",
            ]
            if public_demo
            else [
                f"target: {_short_text(item.get('resolver_target_ref'))}",
                f"introducer: {_short_text(item.get('introducer_ref'))}",
                f"candidate_score: {_short_text(candidate_score)}",
                f"arm_side: {_short_text(item.get('arm_side'))}",
                f"crop_hash: {_short_text(item.get('crop_hash'))}",
                f"face_detection: {face_detection_status}",
            ]
        ),
        boxes=boxes,
        arrows=arrows,
        points=points,
    )
    return item if ok else (_status_item(
        base,
        "image_not_generated",
        "overlay image could not be written",
        source=result,
    ) if include_not_present else None)


def _build_scene_visual_item(
    *,
    out: Path,
    artifact: Path,
    scene: Any | None,
    result: dict[str, Any] | None,
    report_section: str,
    include_not_present: bool = False,
    public_demo: bool = False,
) -> dict[str, Any] | None:
    result = result if isinstance(result, dict) else {}
    base = {
        "assertion_id": "teach_scene_scene_activated",
        "scene": "pic_teach_scene_galbot",
        "report_section": report_section,
    }
    not_present = _not_present_item(base, result=result, scene=scene)
    if not_present is not None:
        return not_present if include_not_present else None

    source_path = _overlay_source_frame_path(scene, result, artifact=artifact)
    if source_path is None or not source_path.is_file():
        return _status_item(
            base,
            "missing_source_frame",
            "source frame not present",
            source=result,
        ) if include_not_present else None

    event = _first_compact_event(result, "scene_activated")
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    item = {
        **base,
        **_result_source_fields(result),
        "kind": "image_overlay",
        "status": "present",
        "path": "visual-evidence/teach-scene-scene-activated.jpg",
        "scene_id": result.get("scene_id"),
        "event_id": event.get("event_id"),
        "memory_match_id": evidence.get("memory_match_id"),
        "crop_hash": result.get("teach_crop_hash"),
        "crop_path_or_artifact_ref": result.get("teach_crop_path_or_artifact_ref"),
        "selected_frame": _selected_frame_path(result),
        "source_frame": str(source_path),
    }
    ok = write_image_overlay(
        source_path=source_path,
        output_path=Path(out) / item["path"],
        title=(
            "Scene teaching remembered"
            if public_demo
            else "Teach scene / scene activated"
        ),
        lines=(
            ["scene: remembered"]
            if public_demo
            else [
                f"scene_id: {_short_text(item.get('scene_id'))}",
                f"event_id: {_short_text(item.get('event_id'))}",
                f"crop_hash: {_short_text(item.get('crop_hash'))}",
            ]
        ),
    )
    return item if ok else (_status_item(
        base,
        "image_not_generated",
        "overlay image could not be written",
        source=result,
    ) if include_not_present else None)


def _build_object_visual_item(
    *,
    out: Path,
    artifact: Path,
    scene: Any | None,
    result: dict[str, Any] | None,
    include_not_present: bool = False,
) -> dict[str, Any] | None:
    result = result if isinstance(result, dict) else {}
    base = {
        "assertion_id": "object_unsupported_no_write",
        "scene": "pic_teach_item_phone",
        "report_section": "object_no_write",
    }
    not_present = _not_present_item(base, result=result, scene=scene)
    if not_present is not None:
        return not_present if include_not_present else None

    source_path = _overlay_source_frame_path(scene, result, artifact=artifact)
    if source_path is None or not source_path.is_file():
        return _status_item(
            base,
            "missing_source_frame",
            "source frame not present",
            source=result,
        ) if include_not_present else None

    resolve_target = result.get("resolve_target") if isinstance(result.get("resolve_target"), dict) else {}
    store_delta_summary = _store_delta_summary(result.get("store_delta"))
    item = {
        **base,
        **_result_source_fields(result),
        "kind": "image_overlay",
        "path": "visual-evidence/object-unsupported-no-write.jpg",
        "unsupported_target_kind": resolve_target.get("target_kind") or "object",
        "status": resolve_target.get("status"),
        "error_code": resolve_target.get("error_code"),
        "store_delta_summary": store_delta_summary,
        "source_frame": str(source_path),
    }
    ok = write_image_overlay(
        source_path=source_path,
        output_path=Path(out) / item["path"],
        title="Object unsupported / no write",
        lines=[
            f"status: {_short_text(resolve_target.get('status'))}",
            f"error_code: {_short_text(item.get('error_code'))}",
            f"no_write: {store_delta_summary.get('delta_all_zero')}",
        ],
    )
    return item if ok else (_status_item(
        base,
        "image_not_generated",
        "overlay image could not be written",
        source=result,
    ) if include_not_present else None)


def _build_full_replay_summary_item(report: dict[str, Any]) -> dict[str, Any]:
    replay = report.get("post_teach_scene_replay")
    base = {
        "assertion_id": "post_teach_full_scene_replay",
        "kind": "summary",
        "path": None,
        "scene": "all",
        "report_section": "post_teach_scene_replay",
    }
    if not isinstance(replay, dict) or replay.get("status") == "not_present":
        return {
            **base,
            "status": "not_present",
            "reason": "post_teach_scene_replay not present in source report",
        }
    return {
        **base,
        "status": "present",
        "passed": replay.get("passed"),
        "replayed_scene_count": replay.get("replayed_scene_count"),
        "scene_count": len(replay.get("scenes") or []) if isinstance(replay.get("scenes"), list) else None,
        "summary": _compact_replay_summary(replay),
    }


def _build_familiar_unknown_summary_item(report: dict[str, Any]) -> dict[str, Any]:
    result = report.get("familiar_unknown")
    base = {
        "assertion_id": "familiar_unknown_present",
        "kind": "memory_demo_summary",
        "path": None,
        "scene": FAMILIAR_UNKNOWN_SCENE,
        "report_section": "familiar_unknown",
    }
    if not isinstance(result, dict):
        return _status_item(base, "not_present", "familiar_unknown not present")
    if result.get("passed") is not True and result.get("status") != "passed":
        return _status_item(
            base,
            str(result.get("status") or "failed"),
            result.get("reason") or "familiar_unknown_present not confirmed",
            source=result,
        )
    event = _first_compact_event(result, "familiar_unknown_present")
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    anonymous = (
        event.get("memory_context", {}).get("anonymous_person")
        if isinstance(event.get("memory_context"), dict)
        else None
    )
    if not isinstance(anonymous, dict):
        anonymous = {}
    return {
        **base,
        "status": "present",
        "event_id": event.get("event_id"),
        "memory_match_id": evidence.get("memory_match_id"),
        "anonymous_id": result.get("anonymous_id") or anonymous.get("anonymous_id"),
        "seen_count": result.get("seen_count") or anonymous.get("seen_count"),
        "observed_duration_ms": result.get("observed_duration_ms")
        or anonymous.get("observed_duration_ms"),
        "familiar_score": result.get("familiar_score")
        or anonymous.get("familiar_score"),
        "selected_frame": _selected_frame_path(result),
    }


def _build_identity_summary_items(artifact: Path) -> list[dict[str, Any]]:
    visual_states = _visual_states_from_sidecar(Path(artifact) / "visual_states.jsonl")
    overlay_state = _first_visual_state_with_overlay(visual_states)
    event_state = _first_visual_state_with_event_identity(visual_states)
    botified_event_identity = _first_botified_event_identity(
        Path(artifact) / "botified_frames.jsonl"
    )
    api_response = _first_api_response_for_endpoint(
        Path(artifact) / "api_responses.jsonl",
        "/v1/memory/identify-current",
    )
    snapshot = _read_json_object(Path(artifact) / "current_visual_snapshot.json")

    return [
        _visual_state_identity_overlay_item(overlay_state),
        _event_identity_context_item(event_state, botified_event_identity),
        _identify_current_response_item(api_response),
        _teach_auto_merge_anonymous_item(Path(artifact) / "api_responses.jsonl"),
        _current_visual_snapshot_item(snapshot),
    ]


def _visual_state_identity_overlay_item(visual_state: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "assertion_id": "visual_state_identity_overlay",
        "kind": "identity_summary",
        "path": None,
        "scene": "sidecar",
        "report_section": "visual_states.jsonl",
    }
    if not isinstance(visual_state, dict):
        return _status_item(base, "not_present", "visual_states.jsonl not present")

    overlay = visual_state.get("identity_context")
    if not isinstance(overlay, dict):
        return _status_item(base, "not_present", "identity_context not present")

    identity = _first_overlay_identity(overlay)
    summary = _identity_summary(identity)
    return {
        **base,
        "status": "present",
        "overlay_status": overlay.get("overlay_status"),
        **summary,
    }


def _event_identity_context_item(
    visual_state: dict[str, Any] | None,
    botified_event_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "assertion_id": "event_identity_context",
        "kind": "identity_summary",
        "path": None,
        "scene": "sidecar",
        "report_section": None,
    }
    if isinstance(botified_event_identity, dict):
        identity = botified_event_identity.get("identity_context")
        return {
            **base,
            "status": "present",
            "source_label": "botified_frames.jsonl",
            "source_path": "botified_frames.jsonl",
            "report_section": "botified_frames.jsonl",
            "event": botified_event_identity.get("event"),
            "identity_status": identity.get("status") if isinstance(identity, dict) else None,
            "identity_source": identity.get("source") if isinstance(identity, dict) else None,
            **_identity_summary(identity),
        }

    if not isinstance(visual_state, dict):
        return _status_item(base, "not_present", "event identity_context not present")

    event = _first_event_with_identity(visual_state)
    if event is None:
        return _status_item(base, "not_present", "event identity_context not present")

    identity = event.get("identity_context")
    return {
        **base,
        "status": "present",
        "source_label": "visual_states.jsonl",
        "source_path": "visual_states.jsonl",
        "report_section": "visual_states.jsonl.semantic_events",
        "event": event.get("event"),
        "identity_status": identity.get("status") if isinstance(identity, dict) else None,
        "identity_source": identity.get("source") if isinstance(identity, dict) else None,
        **_identity_summary(identity),
    }


def _identify_current_response_item(record: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "assertion_id": "identify_current_response",
        "kind": "identity_summary",
        "path": None,
        "scene": "sidecar",
        "report_section": "api_responses.jsonl",
    }
    if not isinstance(record, dict):
        return _status_item(base, "not_present", "identify-current response not present")

    response = record.get("response")
    if not isinstance(response, dict):
        return _status_item(base, "not_present", "identify-current response body not present")

    person = _first_identity_person(response.get("people"))
    identity = person.get("identity_context") if isinstance(person, dict) else None
    return {
        **base,
        "status": "present",
        "endpoint": record.get("endpoint"),
        "operation": record.get("operation"),
        "status_code": record.get("status_code"),
        "identify_status": response.get("status"),
        "target_ref": person.get("target_ref") if isinstance(person, dict) else None,
        **_identity_summary(identity),
    }


def _teach_auto_merge_anonymous_item(path: Path) -> dict[str, Any]:
    base = {
        "assertion_id": "teach_auto_merge_anonymous",
        "kind": "identity_summary",
        "path": None,
        "scene": "sidecar",
        "report_section": "api_responses.jsonl",
    }
    record = _first_teach_auto_merge_response(path)
    if not isinstance(record, dict):
        return _status_item(base, "not_present", "teach auto merge response not present")

    response = record.get("response")
    if not isinstance(response, dict):
        return _status_item(base, "not_present", "teach auto merge response body not present")
    identity = response.get("identity_context")
    if not isinstance(identity, dict):
        person = response.get("person")
        if isinstance(person, dict):
            identity = {"status": "known_person", "person": person}
    return {
        **base,
        "status": "present",
        "endpoint": record.get("endpoint"),
        "operation": record.get("operation"),
        "status_code": record.get("status_code"),
        "outcome": response.get("outcome"),
        "person_id": response.get("person_id"),
        "merged_anonymous_id": response.get("merged_anonymous_id"),
        "copied_embedding_count": response.get("copied_embedding_count"),
        **_identity_summary(identity),
    }


def _current_visual_snapshot_item(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "assertion_id": "cli_current_visual_snapshot",
        "kind": "identity_summary",
        "path": None,
        "scene": "sidecar",
        "report_section": "current_visual_snapshot.json",
    }
    if not isinstance(snapshot, dict):
        return _status_item(base, "not_present", "current_visual_snapshot.json not present")

    person = _first_identity_person(snapshot.get("people"))
    identity = person.get("identity_context") if isinstance(person, dict) else None
    snapshot_text = json.dumps(snapshot, ensure_ascii=False)
    absent = all(field not in snapshot_text for field in SNAPSHOT_FORBIDDEN_FIELDS)
    return {
        **base,
        "status": "present",
        "snapshot_type": snapshot.get("type"),
        "overlay_status": snapshot.get("overlay_status"),
        "target_ref": person.get("target_ref") if isinstance(person, dict) else None,
        "active_target_ref": snapshot.get("active_target_ref"),
        "forbidden_fields_absent": absent,
        **_identity_summary(identity),
    }


def _visual_states_from_sidecar(path: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for record in _read_jsonl_objects(path):
        for key in ("visual_state", "response", "state"):
            value = record.get(key)
            if isinstance(value, dict) and value.get("type") == "visual_state":
                states.append(value)
                break
        else:
            if record.get("type") == "visual_state":
                states.append(record)
    return states


def _first_visual_state_with_overlay(
    visual_states: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best_state: dict[str, Any] | None = None
    best_priority = -1
    for state in visual_states:
        if isinstance(state.get("identity_context"), dict):
            priority = visual_state_identity_priority(state)
            if best_state is None or priority > best_priority:
                best_state = state
                best_priority = priority
    return best_state


def visual_state_identity_priority(visual_state: Any) -> int:
    if not isinstance(visual_state, dict):
        return 0
    overlay = visual_state.get("identity_context")
    if not isinstance(overlay, dict):
        return 0
    return _identity_overlay_priority(overlay)


def _identity_overlay_priority(overlay: dict[str, Any]) -> int:
    priorities = [_identity_status_priority(overlay.get("overlay_status"))]
    tracks = overlay.get("tracks")
    if isinstance(tracks, list):
        for item in tracks:
            if not isinstance(item, dict):
                continue
            identity = item.get("identity")
            if isinstance(identity, dict):
                priorities.append(_identity_status_priority(identity.get("status")))
    return max(priorities, default=0)


def _identity_status_priority(status: Any) -> int:
    if not isinstance(status, str):
        return 0
    return IDENTITY_STATUS_PRIORITY.get(status, 0)


def _first_visual_state_with_event_identity(
    visual_states: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for state in visual_states:
        if _first_event_with_identity(state) is not None:
            return state
    return None


def _first_botified_event_identity(path: Path) -> dict[str, Any] | None:
    for record in _read_jsonl_objects(path):
        summary = _botified_event_identity(record)
        if summary is not None:
            return summary
    return None


def _botified_event_identity(record: dict[str, Any]) -> dict[str, Any] | None:
    request = ""
    payload = record.get("payload")
    if isinstance(payload, dict):
        request_value = payload.get("request")
        if isinstance(request_value, str):
            request = request_value
    if not request:
        return None
    marker = " visual_context="
    marker_index = request.find(marker)
    if marker_index < 0:
        return None
    try:
        context = json.loads(request[marker_index + len(marker) :])
    except json.JSONDecodeError:
        return None
    visual_context = context.get("visual_context") if isinstance(context, dict) else None
    if not isinstance(visual_context, dict):
        return None
    identity = visual_context.get("identity_context")
    if not isinstance(identity, dict):
        return None
    event = record.get("event")
    if not isinstance(event, str):
        event = _request_event_name(request)
    return {"event": event, "identity_context": identity}


def _request_event_name(request: str) -> str | None:
    for part in request.split():
        if part.startswith("event="):
            value = part.split("=", 1)[1]
            return value or None
    return None


def _first_api_response_for_endpoint(path: Path, endpoint: str) -> dict[str, Any] | None:
    for record in _read_jsonl_objects(path):
        if record.get("endpoint") == endpoint:
            return record
    return None


def _first_teach_auto_merge_response(path: Path) -> dict[str, Any] | None:
    for record in _read_jsonl_objects(path):
        if record.get("endpoint") != "/v1/memory/teach/person":
            continue
        response = record.get("response")
        if isinstance(response, dict) and response.get("outcome") == "merged_anonymous_person":
            return record
    return None


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _is_actual_post_record(record: dict[str, Any]) -> bool:
    method = str(record.get("method") or "POST").upper()
    return (
        method == "POST"
        and record.get("dry_run") is not True
        and isinstance(record.get("payload"), dict)
    )


def _actual_posted_payload_summary_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    response = record.get("response")
    stream_ref = payload.get("stream_ref") if isinstance(payload, dict) else None
    return {
        "scene": record.get("scene"),
        "operation": record.get("operation"),
        "endpoint": record.get("endpoint"),
        "stream_ref": stream_ref,
        "status_code": record.get("status_code"),
        "response_status": response.get("status") if isinstance(response, dict) else None,
        "response_outcome": _response_outcome(response),
    }


def _response_outcome(response: Any) -> Any:
    if not isinstance(response, dict):
        return None
    return response.get("outcome") or response.get("teach_person_outcome")


def _first_overlay_identity(overlay: dict[str, Any]) -> dict[str, Any] | None:
    tracks = overlay.get("tracks")
    if not isinstance(tracks, list):
        return None
    for item in tracks:
        if isinstance(item, dict) and isinstance(item.get("identity"), dict):
            return item["identity"]
    return None


def _first_event_with_identity(visual_state: dict[str, Any]) -> dict[str, Any] | None:
    events = visual_state.get("semantic_events")
    if not isinstance(events, list):
        return None
    for event in events:
        if isinstance(event, dict) and isinstance(event.get("identity_context"), dict):
            return event
    return None


def _first_identity_person(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {}
    for item in value:
        if isinstance(item, dict):
            return item
    return {}


def _identity_summary(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        return {}
    summary: dict[str, Any] = {
        "identity_status": identity.get("status"),
        "identity_source": identity.get("source"),
    }
    person = identity.get("person")
    if isinstance(person, dict):
        summary["display_name"] = person.get("display_name")
        summary["person_id"] = person.get("person_id")
    anonymous = identity.get("anonymous_person")
    if isinstance(anonymous, dict):
        summary["anonymous_id"] = anonymous.get("anonymous_id")
    return {key: value for key, value in summary.items() if value is not None}


def write_visual_evidence_index(
    path: Path,
    *,
    scenes: list[Any],
    payload_records: list[dict[str, Any]],
    manifest: dict[str, Any],
    mode: str = "dry-run",
    visual_evidence_index: list[dict[str, Any]] | None = None,
    source_summary: dict[str, Any] | None = None,
    actual_posted_payload_summary: dict[str, Any] | None = None,
    public_demo: bool = False,
) -> None:
    scene_items = "\n".join(
        (
            f"<li><code>{html.escape(scene.name)}</code>: "
            f"{getattr(scene, 'frame_count', len(getattr(scene, 'jpeg_paths', [])))} JPEG frame(s)</li>"
        )
        for scene in scenes
    )
    payload_items = "\n".join(
        (
            f"<li><code>{html.escape(str(record.get('scene') or ''))}</code>: "
            f"{html.escape(str(record.get('endpoint') or ''))} "
            f"<pre>{html.escape(json.dumps(_result_source_fields(record), ensure_ascii=False, indent=2))}</pre>"
            f"<pre>{html.escape(json.dumps(record.get('payload', {}), ensure_ascii=False, indent=2))}</pre>"
            "</li>"
        )
        for record in payload_records
    )
    if not payload_items:
        payload_items = "<li><em>empty</em></li>"
    manifest_note = html.escape(
        "matches actual scene dirs"
        if manifest.get("matches_actual_scene_dirs")
        else "manifest mismatch recorded as non-blocking risk"
    )
    if public_demo:
        source_block = _demo_summary_html(source_summary) if source_summary else ""
        overlay_items = visual_evidence_items_html(
            visual_evidence_index or [],
            public_demo=True,
        )
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Memory Demo</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; line-height: 1.4; }}
    code, pre {{ background: #f5f5f5; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
    img.thumb {{ max-width: 220px; height: auto; }}
  </style>
</head>
<body>
  <h1>Memory Demo</h1>
  <h2>Demo Summary</h2>
  {source_block}
  {_models_html((source_summary or {}).get("models"))}
  <h2>Demo Items</h2>
  {overlay_items}
  <h2>Scenes</h2>
  <ul>
    {scene_items}
  </ul>
</body>
</html>
"""
        path.write_text(document, encoding="utf-8")
        return

    source_block = _source_summary_html(source_summary) if source_summary else ""
    overlay_items = visual_evidence_items_html(visual_evidence_index or [])
    if actual_posted_payload_summary is None:
        actual_posted_payload_summary = load_actual_posted_payload_summary(
            path.parent.parent
        )
    actual_posted_payload_block = actual_posted_payload_summary_html(
        actual_posted_payload_summary
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Memory Teaching {html.escape(mode)} Evidence</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; line-height: 1.4; }}
    code, pre {{ background: #f5f5f5; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
    img.thumb {{ max-width: 220px; height: auto; }}
  </style>
</head>
<body>
  <h1>Memory Teaching {html.escape(mode)} Evidence</h1>
  {source_block}
  <h2>Visual Evidence</h2>
  {overlay_items}
  <h2>Scenes</h2>
  <ul>
    {scene_items}
  </ul>
  <h2>Transcript request templates</h2>
  <p>These entries come from <code>teach_payloads.json</code>. They are transcript-derived request templates and are not posted as-is. <code>ws_payload_fixture</code> is a placeholder/template stream ref; use the actual posted payload summary below for runtime POST payloads.</p>
  <ul>
    {payload_items}
  </ul>
  {actual_posted_payload_block}
  <p>Manifest: {manifest_note}</p>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def actual_posted_payload_summary_html(summary: dict[str, Any]) -> str:
    status = str(summary.get("status") or "unknown")
    records = summary.get("records")
    if not isinstance(records, list):
        records = []
    if not records:
        return (
            "<h2>Actual posted payload summary</h2>"
            "<p><code>api_responses.jsonl</code>: "
            f"{html.escape(status)}</p>"
        )

    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(record.get('scene') or ''))}</code></td>"
            f"<td><code>{html.escape(str(record.get('operation') or ''))}</code></td>"
            f"<td><code>{html.escape(str(record.get('endpoint') or ''))}</code></td>"
            f"<td><code>{html.escape(str(record.get('stream_ref') or ''))}</code></td>"
            f"<td>{html.escape(str(record.get('status_code') or ''))}</td>"
            f"<td>{html.escape(str(record.get('response_status') or ''))}</td>"
            f"<td>{html.escape(str(record.get('response_outcome') or ''))}</td>"
            "</tr>"
        )
    return (
        "<h2>Actual posted payload summary</h2>"
        "<p>Read from <code>api_responses.jsonl</code>; this is the source of truth for POST payloads sent by the runner.</p>"
        "<table><thead><tr>"
        "<th>scene</th><th>operation</th><th>endpoint</th><th>stream_ref</th>"
        "<th>status_code</th><th>response status</th><th>response outcome</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def visual_evidence_items_html(
    visual_evidence_index: list[dict[str, Any]],
    *,
    public_demo: bool = False,
) -> str:
    if public_demo:
        return public_demo_items_html(visual_evidence_index)

    items = [
        item
        for item in visual_evidence_index
        if item.get("kind") != "html_index"
    ]
    if not items:
        return "<p>No overlay images generated for this mode.</p>"

    rows = []
    for item in items:
        href = _visual_evidence_href(item.get("path"))
        image_html = (
            f"<a href=\"{html.escape(href)}\"><img class=\"thumb\" src=\"{html.escape(href)}\" alt=\"\"></a>"
            if href != "#"
            else html.escape(str(item.get("status") or "not present"))
        )
        key_refs = {
            key: item.get(key)
            for key in (
                "request_snapshot_ref",
                "source_frame_ref",
                "introducer_ref",
                "resolver_target_ref",
                "event_id",
                "memory_match_id",
                "face_detection",
                "face_bbox_xyxy",
                "anonymous_id",
                "seen_count",
                "observed_duration_ms",
                "familiar_score",
                "source_text_path",
                "source_image_path",
                "transcript_source",
            )
            if item.get(key) is not None
        }
        for key in ("crop_hash", "crop_preview_path"):
            if item.get(key) is not None:
                key_refs[key] = item.get(key)
        details_html = (
            f"<td><details><summary>JSON</summary><pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre></details></td>"
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(item.get('assertion_id') or ''))}</code></td>"
            f"<td><code>{html.escape(str(item.get('scene') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('status') or ''))}</td>"
            f"<td>{image_html}</td>"
            f"<td><code>{html.escape(str(item.get('report_section') or ''))}</code></td>"
            + f"<td><pre>{html.escape(json.dumps(key_refs, ensure_ascii=False, indent=2))}</pre></td>"
            + details_html
            + "</tr>"
        )
    headers = (
        "<th>Assertion</th><th>Scene</th><th>Status</th><th>Image</th>"
        "<th>Report section</th><th>Key refs</th><th>Details</th>"
    )
    return (
        "<table><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def public_demo_items_html(visual_evidence_index: list[dict[str, Any]]) -> str:
    if not visual_evidence_index:
        return "<p>No completed demo highlights were generated.</p>"

    rows = []
    for item in visual_evidence_index:
        href = _visual_evidence_href(item.get("image"))
        image_html = (
            f"<a href=\"{html.escape(href)}\"><img class=\"thumb\" src=\"{html.escape(href)}\" alt=\"\"></a>"
            if href != "#"
            else ""
        )
        metrics = item.get("metrics")
        metrics_html = (
            f"<pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre>"
            if isinstance(metrics, dict) and metrics
            else ""
        )
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(str(item.get('title') or ''))}</strong></td>"
            f"<td><code>{html.escape(str(item.get('scene') or ''))}</code></td>"
            f"<td>{html.escape(str(item.get('status') or ''))}</td>"
            f"<td>{image_html}</td>"
            f"<td>{html.escape(str(item.get('summary') or ''))}{metrics_html}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Demo Item</th><th>Scene</th><th>Status</th><th>Image</th><th>What happened</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def render_root_index_html(
    *,
    source_summary: dict[str, Any],
    visual_evidence_index: list[dict[str, Any]],
) -> str:
    status = html.escape(str(source_summary.get("status") or "unknown"))
    renderer_status = "ok"
    rows = "\n".join(
        "<li>"
        f"<code>{html.escape(str(item.get('assertion_id') or ''))}</code>: "
        f"{html.escape(str(item.get('status') or ''))} "
        f"{html.escape(str(item.get('path') or ''))}"
        "</li>"
        for item in visual_evidence_index
        if item.get("kind") != "html_index"
    )
    summary_json = html.escape(json.dumps(source_summary, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Memory Teaching Evidence</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; line-height: 1.4; max-width: 1100px; }}
    code, pre {{ background: #f5f5f5; }}
    pre {{ padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Memory Teaching Evidence</h1>
  <p>This is the recommended entry point for human inspection.</p>
  <p>renderer status: {renderer_status}</p>
  <p>source gate: {status}</p>
  <ul>
    <li><a href="visual-evidence/index.html">Visual evidence page</a></li>
    <li><a href="visual_evidence_index.json">visual_evidence_index.json</a></li>
    <li><a href="source-artifact.json">source-artifact.json</a></li>
  </ul>
  <h2>Source Artifact</h2>
  <pre>{summary_json}</pre>
  <h2>Evidence Items</h2>
  <ul>{rows}</ul>
</body>
</html>
"""


DEMO_ITEM_LABELS = {
    "self_introduction_known_person": "记住自我介绍",
    "third_person_pose_pointing": "第三人称指向示教",
    "teach_scene_scene_activated": "记住场景",
    "familiar_unknown_present": "匿名熟客出现",
    "visual_state_identity_overlay": "Current identity overlay",
    "event_identity_context": "Event identity context",
    "identify_current_response": "Current visual snapshot",
    "teach_auto_merge_anonymous": "Anonymous profile merged after teaching",
    "cli_current_visual_snapshot": "Agent-facing snapshot",
}
DEMO_ITEM_PUBLIC_IDS = {
    "self_introduction_known_person": "self_introduction",
    "teach_scene_scene_activated": "scene_teaching",
    "third_person_pose_pointing": "pointing_teaching",
    "familiar_unknown_present": "familiar_unknown",
}
DEMO_ITEM_SUMMARIES = {
    "self_introduction_known_person": "自我介绍后再次看到本人时识别为已记住的人。",
    "third_person_pose_pointing": "根据第三人称指向选择被介绍的人并完成记忆。",
    "teach_scene_scene_activated": "示教过的场景再次出现时被识别出来。",
    "familiar_unknown_present": "未命名但重复出现的人被标记为熟悉的陌生人。",
}


def _public_demo_visual_evidence_index(
    visual_evidence_index: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in visual_evidence_index:
        assertion_id = str(item.get("assertion_id") or "")
        if assertion_id not in DEMO_ITEM_PUBLIC_IDS:
            continue
        if item.get("status") != "present":
            continue
        row: dict[str, Any] = {
            "id": DEMO_ITEM_PUBLIC_IDS[assertion_id],
            "title": DEMO_ITEM_LABELS[assertion_id],
            "status": "passed",
            "scene": item.get("scene"),
            "summary": DEMO_ITEM_SUMMARIES[assertion_id],
        }
        path = item.get("path")
        if isinstance(path, str) and path:
            row["image"] = path
        metrics = {
            key: item.get(key)
            for key in (
                "seen_count",
                "observed_duration_ms",
                "familiar_score",
                "face_detection",
            )
            if item.get(key) is not None
        }
        if metrics:
            row["metrics"] = metrics
        rows.append(row)
    order = {item_id: index for index, item_id in enumerate(DEMO_ITEM_PUBLIC_IDS.values())}
    return sorted(rows, key=lambda row: order.get(str(row.get("id") or ""), 999))


def render_public_demo_root_index_html(
    *,
    source_summary: dict[str, Any],
    visual_evidence_index: list[dict[str, Any]],
) -> str:
    status = html.escape(str(source_summary.get("status") or "unknown"))
    real_model = html.escape(str(source_summary.get("real_model_evidence")))
    rows = "\n".join(
        "<li>"
        f"<strong>{html.escape(str(item.get('title') or ''))}</strong>: "
        f"{html.escape(str(item.get('status') or ''))} "
        f"{html.escape(str(item.get('summary') or ''))}"
        "</li>"
        for item in visual_evidence_index
    )
    if not rows:
        rows = "<li>No completed demo highlights were generated.</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Memory Demo</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; line-height: 1.4; max-width: 1100px; }}
    code, pre {{ background: #f5f5f5; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Memory Demo</h1>
  <h2>Demo Summary</h2>
  <p>status: {status}; real_model_evidence: {real_model}</p>
  {_models_html(source_summary.get("models"))}
  <ul>
    <li><a href="visual-evidence/index.html">Visual evidence page</a></li>
    <li><a href="report.json">report.json</a></li>
  </ul>
  <h2>Demo Items</h2>
  <ul>{rows}</ul>
</body>
</html>
"""


def write_image_overlay(
    *,
    source_path: Path,
    output_path: Path,
    title: str,
    lines: list[str],
    boxes: list[dict[str, Any]] | None = None,
    arrows: list[dict[str, Any]] | None = None,
    points: list[dict[str, Any]] | None = None,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
    except Exception:
        return False

    try:
        with Image.open(source_path) as image:
            canvas = image.convert("RGB")
    except (OSError, UnidentifiedImageError):
        return False

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    width, _height = canvas.size
    text_lines = [title, *[line for line in lines if line]]
    panel_height = 18 + 16 * len(text_lines)
    draw.rectangle((0, 0, width, panel_height), fill=(0, 0, 0))
    y = 8
    for index, line in enumerate(text_lines):
        fill = (255, 255, 255) if index else (255, 230, 120)
        draw.text((10, y), line[:140], fill=fill, font=font)
        y += 16

    for box in boxes or []:
        bbox = _float_bbox(box.get("bbox_xyxy"))
        if bbox is None:
            continue
        color = box.get("color") or (255, 255, 0)
        draw.rectangle(tuple(bbox), outline=color, width=4)
        label = str(box.get("label") or "")
        if label:
            label_y = max(panel_height + 2, bbox[1] - 18)
            draw.rectangle(
                (bbox[0], label_y, bbox[0] + 180, label_y + 18),
                fill=(0, 0, 0),
            )
            draw.text(
                (bbox[0] + 4, label_y + 2),
                label[:42],
                fill=color,
                font=font,
            )

    for arrow in arrows or []:
        start = _xy_pair(arrow.get("start_xy"))
        end = _xy_pair(arrow.get("end_xy"))
        if start is None or end is None:
            continue
        color = arrow.get("color") or (255, 80, 80)
        draw.line((start, end), fill=color, width=5)
        radius = 7
        draw.ellipse(
            (
                end[0] - radius,
                end[1] - radius,
                end[0] + radius,
                end[1] + radius,
            ),
            fill=color,
        )

    for point in points or []:
        xy = _xy_pair(point.get("xy"))
        if xy is None:
            continue
        color = point.get("color") or (255, 230, 120)
        radius = int(point.get("radius") or 5)
        draw.ellipse(
            (
                xy[0] - radius,
                xy[1] - radius,
                xy[0] + radius,
                xy[1] + radius,
            ),
            fill=color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.save(output_path, format="JPEG", quality=90)
    except OSError:
        return False
    return True


def source_report_summary(report: dict[str, Any], *, artifact: Path) -> dict[str, Any]:
    return {
        "artifact_path": str(Path(artifact).resolve()),
        "report_path": str((Path(artifact) / "report.json").resolve()),
        "path_resolution": (
            "relative paths resolve artifact-relative first, then current "
            "working directory"
        ),
        "ok": bool(report.get("ok")),
        "status": _source_gate_status(report),
        "gate": report.get("gate"),
        "mode": report.get("mode"),
        "backend": report.get("backend"),
        "embedding_backend": report.get("embedding_backend"),
        "inference_backend": report.get("inference_backend"),
        "real_model_evidence": report.get("real_model_evidence"),
        "models": report.get("models"),
        "scene_count": report.get("scene_count"),
    }


def public_demo_source_summary(source_summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "real_model_evidence",
        "models",
        "scene_count",
    )
    return {key: source_summary.get(key) for key in keys if key in source_summary}


def _demo_summary_html(source_summary: dict[str, Any] | None) -> str:
    if not source_summary:
        return ""
    rows = "\n".join(
        f"<li><code>{html.escape(str(key))}</code>: {html.escape(str(value))}</li>"
        for key, value in source_summary.items()
        if key != "models"
    )
    return f"<ul>{rows}</ul>"


def response_evidence(body: dict[str, Any]) -> dict[str, Any]:
    evidence = body.get("evidence")
    return evidence if isinstance(evidence, dict) else {}


def _self_result_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(
        report.get("self_smoke"),
        report.get("self_introduction"),
        _check_details(report, "self_introduction_known_person_present"),
    )


def _third_person_result_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(
        report.get("third_person_probe"),
        report.get("third_person_introduction"),
        _check_details(report, "third_person_known_person_present"),
    )


def _scene_result_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(
        report.get("scene_smoke"),
        report.get("teach_scene"),
        _check_details(report, "teach_scene_scene_activated"),
    )


def _object_result_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(
        report.get("object_no_write"),
        _check_details(report, "object_resolve_unsupported_no_write"),
    )


def _check_details(report: dict[str, Any], name: str) -> dict[str, Any]:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return {}
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            details = check.get("details")
            return details if isinstance(details, dict) else {}
    return {}


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _not_present_item(
    base: dict[str, Any],
    *,
    result: dict[str, Any],
    scene: Any | None,
) -> dict[str, Any] | None:
    if scene is None:
        return _status_item(
            base,
            "not_present",
            "source scene not present",
            source=result,
        )
    if not result:
        return _status_item(
            base,
            "not_present",
            "source report section not present",
            source=result,
        )
    if result.get("passed") is True or result.get("status") == "passed":
        return None
    status = str(result.get("status") or "not_present")
    reason = result.get("reason") or result.get("error") or "source evidence not passed"
    return _status_item(base, status, reason, source=result)


def _status_item(
    base: dict[str, Any],
    status: str,
    reason: Any,
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **base,
        **_result_source_fields(source or {}),
        "kind": "not_present",
        "path": None,
        "status": status,
        "reason": reason,
    }


def _overlay_source_frame_path(
    scene: Any,
    result: dict[str, Any],
    *,
    artifact: Path,
) -> Path | None:
    source_image_path = result.get("source_image_path")
    if isinstance(source_image_path, str) and source_image_path:
        source_path = _resolve_report_path(artifact, source_image_path)
        if source_path.is_file():
            return source_path
    selected = _selected_frame_path(result)
    if selected:
        selected_path = _resolve_report_path(artifact, selected)
        if selected_path.is_file():
            return selected_path
    jpeg_paths = getattr(scene, "jpeg_paths", ())
    if jpeg_paths:
        return Path(jpeg_paths[0])
    return None


def _payload_records_by_scene(
    payload_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in payload_records:
        scene = record.get("scene")
        if isinstance(scene, str) and scene and scene not in result:
            result[scene] = record
    return result


def _with_payload_source_fields(
    result: dict[str, Any],
    payload_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload_record, dict):
        return result
    source_fields = _result_source_fields(payload_record)
    if not source_fields:
        return result
    return {**result, **source_fields}


def _result_source_fields(record: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("source_text_path", "source_image_path", "transcript_text"):
        value = record.get(key)
        if isinstance(value, str) and value:
            fields[key] = value
    if "source_text_path" in fields:
        fields["transcript_source"] = fields["source_text_path"]
    return fields


def _selected_frame_path(result: dict[str, Any]) -> str | None:
    selected_window = result.get("selected_window")
    if not isinstance(selected_window, dict):
        return None
    frame = selected_window.get("frame")
    return frame if isinstance(frame, str) and frame else None


def _first_compact_event(result: dict[str, Any], event_name: str) -> dict[str, Any]:
    events = result.get("events") if isinstance(result.get("events"), list) else []
    for event in events:
        if isinstance(event, dict) and event.get("event") == event_name:
            return event
    return {}


def _track_id_from_ref(ref: Any) -> int | None:
    if not isinstance(ref, str):
        return None
    marker = ":track:"
    if marker not in ref:
        return None
    try:
        return int(ref.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _candidate_score_for_target(
    scoring: dict[str, Any],
    target_track_id: int | None,
) -> float | None:
    metrics = _candidate_metrics_for_target(scoring, target_track_id)
    score = metrics.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _candidate_metrics_for_target(
    scoring: dict[str, Any],
    target_track_id: int | None,
) -> dict[str, Any]:
    candidates = scoring.get("candidate_scores")
    if not isinstance(candidates, list):
        return {}
    first_candidate: dict[str, Any] | None = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if first_candidate is None:
            first_candidate = candidate
        if target_track_id is not None and candidate.get("track_id") != target_track_id:
            continue
        return candidate
    return first_candidate or {}


def _first_resolve_candidate_bbox(resolve_target: dict[str, Any]) -> list[float] | None:
    candidates = resolve_target.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return None
    return _float_bbox(candidate.get("bbox_xyxy"))


def _pose_stability_window_summary(window: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "size",
        "fresh_snapshot_count",
        "required_pose_snapshot_count",
        "selected_target_track_id",
        "selected_arm_side",
        "selected_count",
        "failure_reason",
    )
    return {key: window.get(key) for key in keys if key in window}


def _store_delta_summary(store_delta: Any) -> dict[str, Any]:
    if not isinstance(store_delta, dict):
        return {
            "before_equals_after": False,
            "delta_all_zero": False,
            "delta": {},
        }
    before = store_delta.get("before")
    after = store_delta.get("after")
    delta = store_delta.get("delta")
    delta_dict = delta if isinstance(delta, dict) else {}
    return {
        "before_equals_after": before == after,
        "delta_all_zero": bool(delta_dict)
        and all(value == 0 for value in delta_dict.values()),
        "delta": delta_dict,
    }


def _compact_replay_summary(replay: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = replay.get("scenes")
    if not isinstance(scenes, list):
        return []
    result: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        result.append(
            {
                "scene": scene.get("scene"),
                "flags": scene.get("flags"),
                "events": scene.get("event_counts") or scene.get("events"),
            }
        )
    return result


def _find_nested_dict(value: Any, key: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        child = value.get(key)
        if isinstance(child, dict):
            return child
        for nested in value.values():
            found = _find_nested_dict(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_nested_dict(nested, key)
            if found is not None:
                return found
    return None


def _bbox_from_visual_evidence(
    evidence: dict[str, Any] | None,
    *keys: str,
) -> list[float] | None:
    if not isinstance(evidence, dict):
        return None
    for key in keys:
        bbox = _float_bbox(evidence.get(key))
        if bbox is not None:
            return bbox
    return None


def _face_bbox_source_frame(
    person_visual_evidence: dict[str, Any] | None,
) -> list[float] | None:
    if not isinstance(person_visual_evidence, dict):
        return None
    face_detection = person_visual_evidence.get("face_detection")
    if not isinstance(face_detection, dict):
        return None
    face_bbox = _float_bbox(face_detection.get("face_bbox_xyxy"))
    if face_bbox is None:
        return None

    coordinate_space = face_detection.get("coordinate_space")
    if coordinate_space == "source_frame":
        return face_bbox
    if coordinate_space != "crop":
        return None
    if person_visual_evidence.get("crop_box_coordinate_space") != "source_frame":
        return None
    crop_box = _float_bbox(person_visual_evidence.get("crop_box_xyxy"))
    if crop_box is None:
        return None
    left, top = crop_box[0], crop_box[1]
    return [
        face_bbox[0] + left,
        face_bbox[1] + top,
        face_bbox[2] + left,
        face_bbox[3] + top,
    ]


def _pose_arrows(evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(evidence, dict):
        return []
    arrows: list[dict[str, Any]] = []
    shoulder = _xy_pair(evidence.get("shoulder_xy"))
    elbow = _xy_pair(evidence.get("elbow_xy"))
    wrist = _xy_pair(evidence.get("wrist_xy"))
    ray_start = _xy_pair(evidence.get("ray_start_xy"))
    ray_end = _xy_pair(evidence.get("ray_end_xy"))
    if shoulder is not None and elbow is not None:
        arrows.append({"start_xy": shoulder, "end_xy": elbow, "color": (255, 230, 120)})
    if elbow is not None and wrist is not None:
        arrows.append({"start_xy": elbow, "end_xy": wrist, "color": (255, 230, 120)})
    if ray_start is not None and ray_end is not None:
        arrows.append({"start_xy": ray_start, "end_xy": ray_end, "color": (255, 80, 80)})
    return arrows


def _pose_points(evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(evidence, dict):
        return []
    points: list[dict[str, Any]] = []
    for key in ("shoulder_xy", "elbow_xy", "wrist_xy"):
        xy = _xy_pair(evidence.get(key))
        if xy is not None:
            points.append({"xy": xy, "color": (255, 230, 120), "radius": 5})
    return points


def _copy_crop_preview(
    *,
    artifact: Path,
    out: Path,
    source: str | None,
    fallback_name: str,
) -> str | None:
    if not source:
        return None
    source_path = _resolve_report_path(artifact, source)
    if not source_path.is_file():
        return None
    suffix = source_path.suffix.lower()
    name = source_path.name if suffix in JPEG_SUFFIXES else fallback_name
    dest = Path(out) / "visual-evidence" / "crops" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(source_path, dest)
    except OSError:
        return None
    return str(dest.relative_to(out))


def _first_path_value(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _resolve_optional_report_path(artifact: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return _resolve_report_path(artifact, value)


def _resolve_report_path(artifact: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (Path(artifact) / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _float_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _xy_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _short_text(value: Any, *, max_len: int = 32) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _source_gate_status(report: dict[str, Any]) -> str:
    status = report.get("status")
    if isinstance(status, str) and status:
        return status
    return "passed" if report.get("ok") is True else "failed"


def _source_summary_html(source_summary: dict[str, Any] | None) -> str:
    if not source_summary:
        return ""
    rows = "\n".join(
        f"<li><code>{html.escape(str(key))}</code>: {html.escape(str(value))}</li>"
        for key, value in source_summary.items()
    )
    return f"<h2>Source Report</h2><ul>{rows}</ul>"


def _models_html(models: Any) -> str:
    if not isinstance(models, dict) or not models:
        return "<p>models: <em>not recorded</em></p>"
    rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(str(key))}</th>"
        f"<td><code>{html.escape(str(value))}</code></td>"
        "</tr>"
        for key, value in models.items()
    )
    return (
        "<h2>Models</h2>"
        "<table><tbody>"
        + rows
        + "</tbody></table>"
    )


def _visual_evidence_href(path: Any) -> str:
    if not isinstance(path, str) or not path:
        return "#"
    prefix = "visual-evidence/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
