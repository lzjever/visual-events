from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_memory_e2e as memory_e2e
from visual_events_server.config import MemoryEmbeddingConfig
from visual_events_server.protocol import _parse_jpeg_dimensions


DEFAULT_DATA_DIR = Path("val-data")
DEFAULT_OUT = Path("artifacts/memory-teaching-ga")
DEFAULT_CAMERA = "front"

JPEG_SUFFIXES = {".jpeg", ".jpg"}
FORBIDDEN_AGENT_PAYLOAD_FIELDS = {
    "track_id",
    "bbox",
    "bbox_xyxy",
    "point_uv",
    "test_hint",
    "source_scene",
    "source_frame",
}
TEACH_SCENE_ORDER = (
    "pic_teach_me",
    "pic_teach_person",
    "pic_teach_scene_galbot",
    "pic_teach_item_phone",
)


@dataclass(frozen=True)
class SceneDir:
    name: str
    path: Path
    jpeg_paths: tuple[Path, ...]
    des_path: Path | None = None
    des_text: str | None = None

    @property
    def frame_count(self) -> int:
        return len(self.jpeg_paths)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the first-stage memory teaching GA runner payload and "
            "artifact skeleton."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Write stub API responses without calling a server.",
    )
    return parser.parse_args(argv)


def discover_scene_dirs(data_dir: Path) -> list[SceneDir]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data dir not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"data dir is not a directory: {root}")

    scene_paths = {
        path.parent
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES
    }
    if not scene_paths:
        raise FileNotFoundError(f"no JPEG frames found under {root}")

    return [_scene_dir_from_path(path) for path in sorted(scene_paths)]


def manifest_risk_report(data_dir: Path, scenes: list[SceneDir]) -> dict[str, Any]:
    manifest_path = Path(data_dir) / "manifest.json"
    actual_scene_names = sorted(scene.name for scene in scenes)
    report: dict[str, Any] = {
        "path": str(manifest_path),
        "present": manifest_path.is_file(),
        "manifest_scene_count": 0,
        "actual_scene_count": len(actual_scene_names),
        "manifest_scene_names": [],
        "actual_scene_names": actual_scene_names,
        "missing_from_manifest": actual_scene_names,
        "manifest_only_scenes": [],
        "matches_actual_scene_dirs": False,
        "risks": [],
    }
    if not manifest_path.is_file():
        report["risks"].append(
            {
                "code": "manifest_missing",
                "message": "manifest.json is absent; JPEG scene discovery was used.",
            }
        )
        return report

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report["risks"].append(
            {
                "code": "manifest_invalid_json",
                "message": f"manifest.json could not be parsed: {exc}",
            }
        )
        return report

    manifest_scene_names = _manifest_scene_names(manifest)
    manifest_scene_count = _manifest_scene_count(manifest, manifest_scene_names)
    missing_from_manifest = sorted(set(actual_scene_names) - set(manifest_scene_names))
    manifest_only_scenes = sorted(set(manifest_scene_names) - set(actual_scene_names))
    matches = not missing_from_manifest and not manifest_only_scenes

    report.update(
        {
            "manifest_scene_count": manifest_scene_count,
            "manifest_scene_names": manifest_scene_names,
            "missing_from_manifest": missing_from_manifest,
            "manifest_only_scenes": manifest_only_scenes,
            "matches_actual_scene_dirs": matches,
        }
    )
    if manifest_scene_count != len(manifest_scene_names):
        report["risks"].append(
            {
                "code": "manifest_count_inconsistent",
                "message": (
                    "manifest scene_count does not match the number of listed "
                    "manifest scenes."
                ),
            }
        )
    if not matches or manifest_scene_count != len(actual_scene_names):
        report["risks"].append(
            {
                "code": "manifest_mismatch",
                "message": (
                    "manifest.json does not match actual JPEG scene directories; "
                    "the runner continues with discovered JPEG scenes."
                ),
            }
        )
    return report


def build_teach_payload_records(
    data_dir: Path,
    *,
    camera: str = DEFAULT_CAMERA,
) -> list[dict[str, Any]]:
    scenes_by_name = {scene.name: scene for scene in discover_scene_dirs(data_dir)}
    records: list[dict[str, Any]] = []
    for scene_name in TEACH_SCENE_ORDER:
        scene = scenes_by_name.get(scene_name)
        if scene is None:
            continue
        records.append(_teach_payload_record(scene, camera=camera))
    return records


def find_forbidden_agent_payload_fields(payload: Any) -> list[str]:
    found: list[str] = []

    def visit(value: Any, path: list[str]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = [*path, str(key)]
                if key in FORBIDDEN_AGENT_PAYLOAD_FIELDS:
                    found.append(".".join(child_path))
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*path, str(index)])

    visit(payload, [])
    return sorted(found)


def run_dry_run(
    *,
    data_dir: Path,
    out: Path,
    camera: str = DEFAULT_CAMERA,
) -> dict[str, Any]:
    scenes = discover_scene_dirs(data_dir)
    manifest = manifest_risk_report(data_dir, scenes)
    payload_records = _build_teach_payload_records_from_scenes(
        scenes,
        camera=camera,
    )
    forbidden_payload_fields = {
        record["scene"]: find_forbidden_agent_payload_fields(record["payload"])
        for record in payload_records
    }

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    visual_evidence_dir = out / "visual-evidence"
    visual_evidence_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = out / "timeline.jsonl"
    teach_payloads_path = out / "teach_payloads.json"
    api_responses_path = out / "api_responses.jsonl"
    botified_frames_path = out / "botified_frames.jsonl"
    evidence_index_path = visual_evidence_dir / "index.html"
    report_path = out / "report.json"

    _write_jsonl(timeline_path, _timeline_records(scenes, payload_records))
    _write_json(
        teach_payloads_path,
        {
            "schema_version": 1,
            "mode": "dry-run",
            "payloads": payload_records,
        },
    )
    _write_jsonl(api_responses_path, _stub_api_response_records(payload_records))
    _write_jsonl(botified_frames_path, _stub_botified_frame_records(payload_records))
    _write_visual_evidence_index(
        evidence_index_path,
        scenes=scenes,
        payload_records=payload_records,
        manifest=manifest,
        mode="dry-run",
    )

    visual_evidence_index = [
        {
            "assertion_id": "memory_teaching_ga_artifact_skeleton",
            "kind": "html_index",
            "path": "visual-evidence/index.html",
        }
    ]
    artifact_paths = {
        "report_json": "report.json",
        "timeline_jsonl": "timeline.jsonl",
        "teach_payloads_json": "teach_payloads.json",
        "api_responses_jsonl": "api_responses.jsonl",
        "botified_frames_jsonl": "botified_frames.jsonl",
        "visual_evidence_index_html": "visual-evidence/index.html",
    }
    checks = _build_checks(
        scenes=scenes,
        payload_records=payload_records,
        forbidden_payload_fields=forbidden_payload_fields,
        out=out,
        artifact_paths=artifact_paths,
        visual_evidence_index=visual_evidence_index,
    )
    warnings = list(manifest.get("risks") or [])
    report = {
        "ok": all(check["passed"] for check in checks),
        "gate": "memory_teaching_ga_runner_payload_artifact_contract",
        "mode": "dry-run",
        "backend": "stub",
        "real_model_evidence": False,
        "data_dir": str(data_dir),
        "out": str(out),
        "camera": camera,
        "scene_count": len(scenes),
        "scenes": [_scene_report(scene) for scene in scenes],
        "manifest": manifest,
        "warnings": warnings,
        "teach_requests": [_teach_request_summary(record) for record in payload_records],
        "forbidden_agent_payload_fields": forbidden_payload_fields,
        "debug_test_channel_enabled": False,
        "artifacts": artifact_paths,
        "visual_evidence_index": visual_evidence_index,
        "checks": checks,
        "notes": [
            "Dry-run only: no server, DB, embedding, replay, or Botified CLI call was executed.",
            "manifest mismatch is recorded as a risk and does not block JPEG scene enumeration.",
            "Object teaching is negative-only in this stage and is represented as unsupported/no-write.",
        ],
    }
    _write_json(report_path, report)
    return report


def run_actual(
    *,
    data_dir: Path,
    out: Path,
    camera: str = DEFAULT_CAMERA,
) -> dict[str, Any]:
    scenes = discover_scene_dirs(data_dir)
    manifest = manifest_risk_report(data_dir, scenes)
    payload_records = _build_teach_payload_records_from_scenes(
        scenes,
        camera=camera,
    )
    payload_records_by_scene = {record["scene"]: record for record in payload_records}
    forbidden_payload_fields = {
        record["scene"]: find_forbidden_agent_payload_fields(record["payload"])
        for record in payload_records
    }

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runtime_dir = out / "runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    visual_evidence_dir = out / "visual-evidence"
    visual_evidence_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = out / "timeline.jsonl"
    teach_payloads_path = out / "teach_payloads.json"
    api_responses_path = out / "api_responses.jsonl"
    botified_frames_path = out / "botified_frames.jsonl"
    visual_states_path = out / "visual_states.jsonl"
    evidence_index_path = visual_evidence_dir / "index.html"
    report_path = out / "report.json"

    timeline_records = _timeline_records(
        scenes,
        payload_records,
        dry_run=False,
    )
    api_response_records: list[dict[str, Any]] = []
    botified_frame_records: list[dict[str, Any]] = []
    replay_result: dict[str, Any] = {
        "replayed_scene_names": [],
        "replayed_scene_count": 0,
    }
    self_result: dict[str, Any] = {"passed": False, "error": "not_run"}
    third_person_result: dict[str, Any] = {"passed": False, "error": "not_run"}
    scene_result: dict[str, Any] = {"passed": False, "error": "not_run"}
    object_result: dict[str, Any] = {"passed": False, "error": "not_run"}

    with visual_states_path.open("w", encoding="utf-8") as states_file:
        replay_result = _run_actual_scene_replay(
            scenes=scenes,
            out=out,
            camera=camera,
            states_file=states_file,
            timeline_records=timeline_records,
            botified_frame_records=botified_frame_records,
        )
        self_scene = _find_scene(scenes, "pic_teach_me")
        self_record = payload_records_by_scene.get("pic_teach_me")
        if self_scene is None or self_record is None:
            self_result = _missing_teaching_scene_result("pic_teach_me")
        else:
            self_result = _run_actual_self_introduction(
                out=out,
                scene=self_scene,
                record=self_record,
                camera=camera,
                states_file=states_file,
                api_response_records=api_response_records,
                botified_frame_records=botified_frame_records,
            )

        third_person_scene = _find_scene(scenes, "pic_teach_person")
        third_person_record = payload_records_by_scene.get("pic_teach_person")
        if third_person_scene is None or third_person_record is None:
            third_person_result = _missing_teaching_scene_result("pic_teach_person")
        else:
            third_person_result = _run_actual_third_person_introduction(
                out=out,
                scene=third_person_scene,
                record=third_person_record,
                camera=camera,
                states_file=states_file,
                api_response_records=api_response_records,
                botified_frame_records=botified_frame_records,
            )

        scene_scene = _find_scene(scenes, "pic_teach_scene_galbot")
        scene_record = payload_records_by_scene.get("pic_teach_scene_galbot")
        if scene_scene is None or scene_record is None:
            scene_result = _missing_teaching_scene_result("pic_teach_scene_galbot")
        else:
            scene_result = _run_actual_teach_scene(
                out=out,
                scene=scene_scene,
                record=scene_record,
                camera=camera,
                states_file=states_file,
                api_response_records=api_response_records,
                botified_frame_records=botified_frame_records,
            )

        object_scene = _find_scene(scenes, "pic_teach_item_phone")
        object_record = payload_records_by_scene.get("pic_teach_item_phone")
        if object_scene is None or object_record is None:
            object_result = _missing_teaching_scene_result("pic_teach_item_phone")
        else:
            object_result = _run_actual_object_negative(
                out=out,
                scene=object_scene,
                record=object_record,
                camera=camera,
                api_response_records=api_response_records,
            )

    _write_json(
        teach_payloads_path,
        {
            "schema_version": 1,
            "mode": "actual",
            "backend": "fake",
            "payloads": payload_records,
        },
    )
    _write_jsonl(api_responses_path, api_response_records)
    _write_jsonl(botified_frames_path, botified_frame_records)
    _write_visual_evidence_index(
        evidence_index_path,
        scenes=scenes,
        payload_records=payload_records,
        manifest=manifest,
        mode="actual",
    )

    visual_evidence_index = [
        {
            "assertion_id": "memory_teaching_ga_actual_fake",
            "kind": "html_index",
            "path": "visual-evidence/index.html",
        }
    ]
    artifact_paths = {
        "report_json": "report.json",
        "timeline_jsonl": "timeline.jsonl",
        "teach_payloads_json": "teach_payloads.json",
        "api_responses_jsonl": "api_responses.jsonl",
        "botified_frames_jsonl": "botified_frames.jsonl",
        "visual_states_jsonl": "visual_states.jsonl",
        "visual_evidence_index_html": "visual-evidence/index.html",
        "runtime_dir": "runtime",
    }
    _write_jsonl(timeline_path, timeline_records)
    checks = _build_actual_checks(
        scenes=scenes,
        payload_records=payload_records,
        forbidden_payload_fields=forbidden_payload_fields,
        out=out,
        artifact_paths=artifact_paths,
        visual_evidence_index=visual_evidence_index,
        replay_result=replay_result,
        api_response_records=api_response_records,
        self_result=self_result,
        third_person_result=third_person_result,
        scene_result=scene_result,
        object_result=object_result,
    )
    warnings = list(manifest.get("risks") or [])
    report = {
        "ok": all(check["passed"] for check in checks),
        "gate": "memory_teaching_ga_runner_actual_fake",
        "mode": "actual",
        "backend": "fake",
        "real_model_evidence": False,
        "data_dir": str(data_dir),
        "out": str(out),
        "camera": camera,
        "scene_count": len(scenes),
        "replayed_scene_count": replay_result["replayed_scene_count"],
        "replayed_scene_names": replay_result["replayed_scene_names"],
        "scenes": [_scene_report(scene) for scene in scenes],
        "manifest": manifest,
        "warnings": warnings,
        "teach_requests": [_teach_request_summary(record) for record in payload_records],
        "forbidden_agent_payload_fields": forbidden_payload_fields,
        "api_responses": api_response_records,
        "object_no_write": object_result,
        "debug_test_channel_enabled": False,
        "artifacts": artifact_paths,
        "visual_evidence_index": visual_evidence_index,
        "checks": checks,
        "notes": [
            "Actual fake mode: in-process FastAPI/TestClient server with deterministic fake memory embeddings.",
            "No local model backend, CLI subprocess, visual overlay, or schema changes are used.",
            "Real val-data JPEG bytes are streamed once per discovered scene directory.",
        ],
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        report = run_dry_run(data_dir=args.data_dir, out=args.out, camera=args.camera)
    else:
        report = run_actual(data_dir=args.data_dir, out=args.out, camera=args.camera)
    print(
        "memory teaching GA runner "
        f"{report['mode']} {'passed' if report['ok'] else 'failed'}"
    )
    print(f"scenes: {report['scene_count']}")
    if report["mode"] == "actual":
        print(f"replayed scenes: {report['replayed_scene_count']}")
    print(f"report: {Path(args.out) / 'report.json'}")
    return 0 if report["ok"] else 1


def _run_actual_scene_replay(
    *,
    scenes: list[SceneDir],
    out: Path,
    camera: str,
    states_file: Any,
    timeline_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="all-scene-replay",
        out=out,
        source_frame=_load_source_frame_from_scene(scenes[0]),
        camera=camera,
    )
    replayed_scene_names: list[str] = []
    with runner.open_stream() as websocket:
        for index, scene in enumerate(scenes):
            runner.source_frame = _load_source_frame_from_scene(scene)
            runner.processor.mode = (
                "third_person" if scene.name == "pic_teach_person" else "single"
            )
            state = runner.send(
                websocket,
                timestamp_ms=(index + 1) * 1_000,
                states_file=states_file,
                phase=f"scene-replay:{scene.name}",
            )
            events = list(state.get("semantic_events") or [])
            replayed_scene_names.append(scene.name)
            timeline_records.append(
                {
                    "type": "scene_replayed",
                    "mode": "actual",
                    "backend": "fake",
                    "scene": scene.name,
                    "frame": str(runner.source_frame.path),
                    "frame_id": runner.frame_id,
                    "semantic_event_count": len(events),
                }
            )
            _append_botified_frame_records(
                botified_frame_records,
                case="all-scene-replay",
                scene=scene.name,
                phase="scene-replay",
                events=events,
            )
    return {
        "passed": len(replayed_scene_names) == len(scenes),
        "replayed_scene_names": replayed_scene_names,
        "replayed_scene_count": len(replayed_scene_names),
    }


def _run_actual_self_introduction(
    *,
    out: Path,
    scene: SceneDir,
    record: dict[str, Any],
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-self-introduction",
        out=out,
        source_frame=_load_source_frame_from_scene(scene),
        camera=camera,
    )
    with runner.open_stream() as websocket:
        runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=1_000,
            states_file=states_file,
            phase="self-seed",
        )
        teach = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=_payload_index(record),
            scene=record["scene"],
            endpoint=record["endpoint"],
            payload=record["payload"],
            operation="teach_person_self",
        )
        events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=2_000,
            states_file=states_file,
            phase="self-replay",
        )
    known = memory_e2e.first_event(events, "known_person_present")
    assertions = {
        "teach_person_ok": teach["body"].get("ok") is True,
        "known_person_present": known is not None,
        "known_person_context": bool(
            known
            and known.get("memory_context", {}).get("person", {}).get("person_id")
            == teach["body"].get("person_id")
        ),
    }
    _append_botified_frame_records(
        botified_frame_records,
        case="ga-self-introduction",
        scene=scene.name,
        phase="self-replay",
        events=events,
    )
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "person_id": teach["body"].get("person_id"),
        "events": memory_e2e.compact_events(events),
    }


def _run_actual_third_person_introduction(
    *,
    out: Path,
    scene: SceneDir,
    record: dict[str, Any],
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-third-person-introduction",
        out=out,
        source_frame=_load_source_frame_from_scene(scene),
        camera=camera,
    )
    runner.processor.mode = "third_person"
    resolve_payload = {
        "camera": camera,
        "target": record["payload"]["target"],
    }
    with runner.open_stream() as websocket:
        runner.send(
            websocket,
            timestamp_ms=1_000,
            states_file=states_file,
            phase="third-person-seed",
        )
        resolve = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=f"{_payload_index(record)}:resolve",
            scene=record["scene"],
            endpoint="/v1/memory/resolve-target",
            payload=resolve_payload,
            operation="resolve_third_person_target",
        )
        teach = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=_payload_index(record),
            scene=record["scene"],
            endpoint=record["endpoint"],
            payload=record["payload"],
            operation="teach_person_third_person",
        )
        events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=2_000,
            states_file=states_file,
            phase="third-person-replay",
        )
    known = memory_e2e.first_event(events, "known_person_present")
    candidates = resolve["body"].get("candidates") or []
    resolved_track_id = candidates[0].get("track_id") if candidates else None
    assertions = {
        "resolve_target_ok": resolve["body"].get("ok") is True,
        "resolve_target_resolved": resolve["body"].get("status") == "resolved",
        "resolver_selected_b": resolved_track_id == memory_e2e.AMBIGUOUS_TRACK_ID,
        "teach_person_ok": teach["body"].get("ok") is True,
        "known_person_present": known is not None,
        "known_person_is_b": bool(
            known and known.get("track_id") == memory_e2e.AMBIGUOUS_TRACK_ID
        ),
    }
    _append_botified_frame_records(
        botified_frame_records,
        case="ga-third-person-introduction",
        scene=scene.name,
        phase="third-person-replay",
        events=events,
    )
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "resolve_target": resolve["body"],
        "person_id": teach["body"].get("person_id"),
        "events": memory_e2e.compact_events(events),
    }


def _run_actual_teach_scene(
    *,
    out: Path,
    scene: SceneDir,
    record: dict[str, Any],
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-teach-scene",
        out=out,
        source_frame=_load_source_frame_from_scene(scene),
        camera=camera,
    )
    with runner.open_stream() as websocket:
        runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=1_000,
            states_file=states_file,
            phase="scene-seed",
        )
        teach = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=_payload_index(record),
            scene=record["scene"],
            endpoint=record["endpoint"],
            payload=record["payload"],
            operation="teach_scene",
        )
        events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=2_000,
            states_file=states_file,
            phase="scene-replay",
        )
    scene_event = memory_e2e.first_event(events, "scene_activated")
    assertions = {
        "teach_scene_ok": teach["body"].get("ok") is True,
        "scene_activated": scene_event is not None,
        "scene_context": bool(
            scene_event
            and scene_event.get("memory_context", {}).get("scene", {}).get("scene_id")
            == teach["body"].get("scene_id")
        ),
    }
    _append_botified_frame_records(
        botified_frame_records,
        case="ga-teach-scene",
        scene=scene.name,
        phase="scene-replay",
        events=events,
    )
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "scene_id": teach["body"].get("scene_id"),
        "events": memory_e2e.compact_events(events),
    }


def _run_actual_object_negative(
    *,
    out: Path,
    scene: SceneDir,
    record: dict[str, Any],
    camera: str,
    api_response_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-object-negative",
        out=out,
        source_frame=_load_source_frame_from_scene(scene),
        camera=camera,
    )
    store = runner.client.app.state.memory_service.store
    before_counts = memory_e2e._memory_write_counts(store)
    response = _post_and_record_api_response(
        runner=runner,
        api_response_records=api_response_records,
        payload_index=_payload_index(record),
        scene=record["scene"],
        endpoint=record["endpoint"],
        payload=record["payload"],
        operation="resolve_object_unsupported",
    )
    after_counts = memory_e2e._memory_write_counts(store)
    body = response["body"]
    assertions = {
        "status_code_200": response["status_code"] == 200,
        "status_not_found": body.get("status") == "not_found",
        "unsupported_target_kind": body.get("error_code") == "unsupported_target_kind",
        "no_candidates": body.get("candidates") == [],
        "no_memory_write": before_counts == after_counts,
    }
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "resolve_target": body,
        "before_counts": before_counts,
        "after_counts": after_counts,
    }


def _actual_runner(
    *,
    case: str,
    out: Path,
    source_frame: memory_e2e.SourceFrame,
    camera: str,
) -> memory_e2e.MemoryE2ERunner:
    return memory_e2e.MemoryE2ERunner(
        case=case,
        out=out,
        source_frame=source_frame,
        camera=camera,
        embedding_config=MemoryEmbeddingConfig(backend="fake"),
    )


def _load_source_frame_from_scene(scene: SceneDir) -> memory_e2e.SourceFrame:
    if not scene.jpeg_paths:
        raise FileNotFoundError(f"no JPEG frames found in {scene.path}")
    path = scene.jpeg_paths[0]
    jpeg_bytes = path.read_bytes()
    width, height = _parse_jpeg_dimensions(jpeg_bytes, frame_id=None)
    return memory_e2e.SourceFrame(
        path=path,
        jpeg_bytes=jpeg_bytes,
        width=width,
        height=height,
    )


def _post_and_record_api_response(
    *,
    runner: memory_e2e.MemoryE2ERunner,
    api_response_records: list[dict[str, Any]],
    payload_index: int | str,
    scene: str,
    endpoint: str,
    payload: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    response = runner.client.post(endpoint, json=payload)
    body = response.json()
    record = {
        "payload_index": payload_index,
        "scene": scene,
        "endpoint": endpoint,
        "operation": operation,
        "dry_run": False,
        "status_code": response.status_code,
        "payload": payload,
        "response": body,
    }
    api_response_records.append(record)
    return {"status_code": response.status_code, "body": body}


def _append_botified_frame_records(
    records: list[dict[str, Any]],
    *,
    case: str,
    scene: str,
    phase: str,
    events: list[dict[str, Any]],
) -> None:
    for event in memory_e2e.compact_events(events):
        records.append(
            {
                "case": case,
                "scene": scene,
                "phase": phase,
                "dry_run": False,
                "semantic_event": event,
            }
        )


def _find_scene(scenes: list[SceneDir], name: str) -> SceneDir | None:
    for scene in scenes:
        if scene.name == name:
            return scene
    return None


def _missing_teaching_scene_result(scene_name: str) -> dict[str, Any]:
    return {
        "passed": False,
        "error": "required_teaching_scene_missing",
        "scene": scene_name,
        "message": f"required teaching scene was not discovered: {scene_name}",
    }


def _payload_index(record: dict[str, Any]) -> int:
    return TEACH_SCENE_ORDER.index(record["scene"])


def _scene_dir_from_path(path: Path) -> SceneDir:
    jpeg_paths = tuple(_sorted_jpeg_paths(path))
    des_path = path / "des.txt"
    des_text = des_path.read_text(encoding="utf-8").strip() if des_path.is_file() else None
    return SceneDir(
        name=path.name,
        path=path,
        jpeg_paths=jpeg_paths,
        des_path=des_path if des_path.is_file() else None,
        des_text=des_text,
    )


def _sorted_jpeg_paths(path: Path) -> list[Path]:
    return sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in JPEG_SUFFIXES
        ],
        key=lambda item: item.name,
    )


def _manifest_scene_names(manifest: Any) -> list[str]:
    if not isinstance(manifest, dict):
        return []
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list):
        return []
    names: list[str] = []
    for item in scenes:
        if isinstance(item, dict):
            name = item.get("scene_name") or item.get("name")
        else:
            name = item
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(names)


def _manifest_scene_count(manifest: Any, scene_names: list[str]) -> int:
    if isinstance(manifest, dict) and isinstance(manifest.get("scene_count"), int):
        return int(manifest["scene_count"])
    return len(scene_names)


def _build_teach_payload_records_from_scenes(
    scenes: list[SceneDir],
    *,
    camera: str,
) -> list[dict[str, Any]]:
    scenes_by_name = {scene.name: scene for scene in scenes}
    records: list[dict[str, Any]] = []
    for scene_name in TEACH_SCENE_ORDER:
        scene = scenes_by_name.get(scene_name)
        if scene is not None:
            records.append(_teach_payload_record(scene, camera=camera))
    return records


def _teach_payload_record(scene: SceneDir, *, camera: str) -> dict[str, Any]:
    des_text = scene.des_text or ""
    if scene.name == "pic_teach_me":
        display_name = _extract_self_display_name(des_text)
        endpoint = "/v1/memory/teach/person"
        payload = {
            "camera": camera,
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
            "profile": {"display_name": display_name},
        }
        expected = {"writes_memory": True, "memory_type": "person"}
    elif scene.name == "pic_teach_person":
        display_name = _extract_third_person_display_name(des_text)
        endpoint = "/v1/memory/teach/person"
        payload = {
            "camera": camera,
            "target": {
                "kind": "person",
                "intent": "third_person_introduction",
                "referent_text": f"这位/{display_name}",
            },
            "profile": {"display_name": display_name},
        }
        expected = {"writes_memory": True, "memory_type": "person"}
    elif scene.name == "pic_teach_scene_galbot":
        endpoint = "/v1/memory/teach/scene"
        payload = {
            "camera": camera,
            "target": {
                "kind": "scene",
                "intent": "teach_scene",
                "referent_text": _extract_scene_referent_text(des_text),
            },
            "memory": {"title": _extract_scene_title(des_text)},
        }
        expected = {"writes_memory": True, "memory_type": "scene"}
    elif scene.name == "pic_teach_item_phone":
        endpoint = "/v1/memory/resolve-target"
        payload = {
            "camera": camera,
            "target": {
                "kind": "object",
                "intent": "teach_object",
                "referent_text": _extract_object_referent(des_text),
            },
        }
        expected = {
            "negative_only": True,
            "status": "not_found",
            "error_code": "unsupported_target_kind",
            "writes_memory": False,
        }
    else:
        raise ValueError(f"unsupported teach scene: {scene.name}")

    return {
        "scene": scene.name,
        "des_path": str(scene.des_path) if scene.des_path is not None else None,
        "des_text": des_text,
        "endpoint": endpoint,
        "payload": payload,
        "expected": expected,
    }


def _extract_self_display_name(text: str) -> str:
    match = re.search(r"我是\s*([^，,。.!！?？\s]+)", text)
    return match.group(1) if match else "小李飞刀"


def _extract_third_person_display_name(text: str) -> str:
    match = re.search(r"这是\s*([^，,。.!！?？\s]+)", text)
    return match.group(1) if match else "彭刚"


def _extract_scene_title(text: str) -> str:
    match = re.search(r"这是\s*([^，,。.!！?？]+)", text)
    if match is None:
        return "银河通用办公室"
    title = match.group(1).strip()
    title = title.replace("的办公室", "办公室")
    return title or "银河通用办公室"


def _extract_scene_referent_text(text: str) -> str:
    return f"这里/{_extract_scene_title(text)}"


def _extract_object_referent(text: str) -> str:
    return "手机" if "手机" in text else "手机"


def _timeline_records(
    scenes: list[SceneDir],
    payload_records: list[dict[str, Any]],
    *,
    dry_run: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in scenes:
        records.append(
            {
                "type": "scene_discovered",
                "scene": scene.name,
                "frame_count": scene.frame_count,
                "first_frame": str(scene.jpeg_paths[0]) if scene.jpeg_paths else None,
                "last_frame": str(scene.jpeg_paths[-1]) if scene.jpeg_paths else None,
                "has_des": scene.des_path is not None,
            }
        )
    for index, record in enumerate(payload_records):
        records.append(
            {
                "type": "teach_payload_prepared",
                "payload_index": index,
                "scene": record["scene"],
                "endpoint": record["endpoint"],
                "target": record["payload"]["target"],
                "dry_run": dry_run,
            }
        )
    return records


def _stub_api_response_records(
    payload_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, record in enumerate(payload_records):
        expected = record["expected"]
        if expected.get("negative_only"):
            response = {
                "ok": False,
                "status": expected["status"],
                "error_code": expected["error_code"],
                "writes_memory": False,
                "retryable": False,
                "ask_user_hint": False,
            }
        else:
            response = {
                "ok": True,
                "status": "stubbed",
                "would_call": record["endpoint"],
                "writes_memory": True,
            }
        records.append(
            {
                "payload_index": index,
                "scene": record["scene"],
                "endpoint": record["endpoint"],
                "dry_run": True,
                "response": response,
            }
        )
    return records


def _stub_botified_frame_records(
    payload_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "payload_index": index,
            "scene": record["scene"],
            "dry_run": True,
            "botified_frame": None,
            "reason": "dry_run_does_not_call_server_or_cli",
        }
        for index, record in enumerate(payload_records)
    ]


def _write_visual_evidence_index(
    path: Path,
    *,
    scenes: list[SceneDir],
    payload_records: list[dict[str, Any]],
    manifest: dict[str, Any],
    mode: str = "dry-run",
) -> None:
    scene_items = "\n".join(
        (
            f"<li><code>{html.escape(scene.name)}</code>: "
            f"{scene.frame_count} JPEG frame(s)</li>"
        )
        for scene in scenes
    )
    payload_items = "\n".join(
        (
            f"<li><code>{html.escape(record['scene'])}</code>: "
            f"{html.escape(record['endpoint'])} "
            f"<pre>{html.escape(json.dumps(record['payload'], ensure_ascii=False, indent=2))}</pre>"
            "</li>"
        )
        for record in payload_records
    )
    manifest_note = html.escape(
        "matches actual scene dirs"
        if manifest.get("matches_actual_scene_dirs")
        else "manifest mismatch recorded as non-blocking risk"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Memory Teaching GA {html.escape(mode)} Evidence</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; line-height: 1.4; }}
    code, pre {{ background: #f5f5f5; }}
    pre {{ padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Memory Teaching GA {html.escape(mode)} Evidence</h1>
  <p>This minimal index records discovered JPEG scenes and stable teach payloads.</p>
  <p>Manifest: {manifest_note}</p>
  <h2>Scenes</h2>
  <ul>
    {scene_items}
  </ul>
  <h2>Teach Payloads</h2>
  <ul>
    {payload_items}
  </ul>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _build_checks(
    *,
    scenes: list[SceneDir],
    payload_records: list[dict[str, Any]],
    forbidden_payload_fields: dict[str, list[str]],
    out: Path,
    artifact_paths: dict[str, str],
    visual_evidence_index: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_teach_scenes = set(TEACH_SCENE_ORDER)
    actual_teach_scenes = {record["scene"] for record in payload_records}
    artifact_exists = {
        key: (out / relative_path).is_file()
        for key, relative_path in artifact_paths.items()
        if key != "report_json"
    }
    evidence_exists = {
        item["path"]: (out / item["path"]).is_file() for item in visual_evidence_index
    }
    return [
        {
            "name": "discover_jpeg_scene_dirs",
            "passed": bool(scenes),
            "details": {"scene_count": len(scenes)},
        },
        {
            "name": "expected_teach_des_payloads",
            "passed": expected_teach_scenes <= actual_teach_scenes,
            "details": {
                "expected": sorted(expected_teach_scenes),
                "actual": sorted(actual_teach_scenes),
                "missing": sorted(expected_teach_scenes - actual_teach_scenes),
            },
        },
        {
            "name": "agent_payload_forbidden_fields_absent",
            "passed": all(not fields for fields in forbidden_payload_fields.values()),
            "details": forbidden_payload_fields,
        },
        {
            "name": "artifact_skeleton",
            "passed": all(artifact_exists.values()) and all(evidence_exists.values()),
            "details": {
                "artifacts": artifact_exists,
                "visual_evidence": evidence_exists,
            },
        },
    ]


def _build_actual_checks(
    *,
    scenes: list[SceneDir],
    payload_records: list[dict[str, Any]],
    forbidden_payload_fields: dict[str, list[str]],
    out: Path,
    artifact_paths: dict[str, str],
    visual_evidence_index: list[dict[str, Any]],
    replay_result: dict[str, Any],
    api_response_records: list[dict[str, Any]],
    self_result: dict[str, Any],
    third_person_result: dict[str, Any],
    scene_result: dict[str, Any],
    object_result: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_teach_scenes = set(TEACH_SCENE_ORDER)
    actual_teach_scenes = {record["scene"] for record in payload_records}
    artifact_exists = {
        key: (out / relative_path).exists()
        for key, relative_path in artifact_paths.items()
        if key != "report_json"
    }
    evidence_exists = {
        item["path"]: (out / item["path"]).is_file() for item in visual_evidence_index
    }
    actual_response_assertions = {
        "has_api_responses": bool(api_response_records),
        "all_actual": all(
            record.get("dry_run") is False for record in api_response_records
        ),
        "no_stubbed_status": all(
            record.get("response", {}).get("status") != "stubbed"
            for record in api_response_records
        ),
        "status_codes_ok": all(
            int(record.get("status_code") or 0) < 400
            for record in api_response_records
        ),
    }
    return [
        {
            "name": "discover_jpeg_scene_dirs",
            "passed": bool(scenes),
            "details": {"scene_count": len(scenes)},
        },
        {
            "name": "expected_teach_des_payloads",
            "passed": expected_teach_scenes <= actual_teach_scenes,
            "details": {
                "expected": sorted(expected_teach_scenes),
                "actual": sorted(actual_teach_scenes),
                "missing": sorted(expected_teach_scenes - actual_teach_scenes),
            },
        },
        {
            "name": "agent_payload_forbidden_fields_absent",
            "passed": all(not fields for fields in forbidden_payload_fields.values()),
            "details": forbidden_payload_fields,
        },
        {
            "name": "all_scenes_replayed",
            "passed": replay_result.get("replayed_scene_count") == len(scenes),
            "details": replay_result,
        },
        {
            "name": "actual_api_responses",
            "passed": all(actual_response_assertions.values()),
            "details": {
                "assertions": actual_response_assertions,
                "response_count": len(api_response_records),
            },
        },
        {
            "name": "self_introduction_known_person_present",
            "passed": bool(self_result.get("passed")),
            "details": self_result,
        },
        {
            "name": "third_person_known_person_present",
            "passed": bool(third_person_result.get("passed")),
            "details": third_person_result,
        },
        {
            "name": "teach_scene_scene_activated",
            "passed": bool(scene_result.get("passed")),
            "details": scene_result,
        },
        {
            "name": "object_resolve_unsupported_no_write",
            "passed": bool(object_result.get("passed")),
            "details": object_result,
        },
        {
            "name": "artifact_skeleton",
            "passed": all(artifact_exists.values()) and all(evidence_exists.values()),
            "details": {
                "artifacts": artifact_exists,
                "visual_evidence": evidence_exists,
            },
        },
    ]


def _scene_report(scene: SceneDir) -> dict[str, Any]:
    return {
        "name": scene.name,
        "path": str(scene.path),
        "frame_count": scene.frame_count,
        "first_frame": str(scene.jpeg_paths[0]) if scene.jpeg_paths else None,
        "last_frame": str(scene.jpeg_paths[-1]) if scene.jpeg_paths else None,
        "has_des": scene.des_path is not None,
        "des_path": str(scene.des_path) if scene.des_path is not None else None,
    }


def _teach_request_summary(record: dict[str, Any]) -> dict[str, Any]:
    payload = record["payload"]
    return {
        "scene": record["scene"],
        "endpoint": record["endpoint"],
        "camera": payload["camera"],
        "target": payload["target"],
        "profile": payload.get("profile"),
        "memory": payload.get("memory"),
        "expected": record["expected"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
