from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from fastapi.testclient import TestClient

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_memory_e2e as memory_e2e
from visual_events_cli.botified_output import BotifiedStdoutWriter
from visual_events_cli.frame_pump import (
    FramePump,
    HeadMotion,
    InputFrame,
    LatestFrameSlot,
)
from visual_events_server.app import create_app, create_processor_from_config
from visual_events_server.config import (
    InferenceConfig,
    MemoryConfig,
    MemoryEmbeddingConfig,
    MemoryMatchingConfig,
    ServerConfig,
)
from visual_events_server.protocol import _parse_jpeg_dimensions
from visual_events_server.protocol import SCHEMA_VERSION, encode_frame_message


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
POST_TEACH_SCENE_REPLAY_CASE = "ga-post-teach-scene-replay"
CLI_BOTIFIED_FRAME_SOURCE = "cli_frame_pump_stdout"
BOTIFIED_OPEN = "<botified>"
BOTIFIED_CLOSE = "</botified>"


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
    parser.add_argument(
        "--local-smoke",
        action="store_true",
        default=False,
        help="Run a minimal real local model backend smoke on fixed val-data samples.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("fake", "local"),
        default="fake",
        help="Memory embedding backend. --local-smoke requires local.",
    )
    parser.add_argument(
        "--person-model-path",
        type=Path,
        help="Local person embedding model bundle path. Required for --local-smoke.",
    )
    parser.add_argument(
        "--scene-model-path",
        type=Path,
        help="Local scene embedding model bundle path. Required for --local-smoke.",
    )
    parser.add_argument(
        "--inference-backend",
        choices=("mock", "ultralytics"),
        default="mock",
        help="Inference backend. --local-smoke requires ultralytics.",
    )
    parser.add_argument(
        "--pose-model-path",
        type=Path,
        help="Ultralytics pose model path. Required for --local-smoke.",
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
    post_teach_scene_replay_result: dict[str, Any] = {
        "passed": False,
        "error": "not_run",
    }
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

        post_teach_scene_replay_result = _run_actual_post_teach_scene_replay(
            scenes=scenes,
            payload_records_by_scene=payload_records_by_scene,
            out=out,
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
        post_teach_scene_replay_result=post_teach_scene_replay_result,
        object_result=object_result,
        botified_frame_records=botified_frame_records,
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
        "third_person_introduction": third_person_result,
        "post_teach_scene_replay": post_teach_scene_replay_result,
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


def run_local_smoke(
    *,
    data_dir: Path,
    out: Path,
    camera: str = DEFAULT_CAMERA,
    embedding_backend: str,
    person_model_path: Path | None,
    scene_model_path: Path | None,
    inference_backend: str,
    pose_model_path: Path | None,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    preflight = _local_smoke_preflight(
        embedding_backend=embedding_backend,
        person_model_path=person_model_path,
        scene_model_path=scene_model_path,
        inference_backend=inference_backend,
        pose_model_path=pose_model_path,
    )
    if not preflight["passed"]:
        report = _local_smoke_preflight_failed_report(
            data_dir=data_dir,
            out=out,
            camera=camera,
            preflight=preflight,
            report_path=report_path,
        )
        _write_json(report_path, report)
        return report

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

    timeline_records = _timeline_records(
        scenes,
        payload_records,
        dry_run=False,
    )
    local_config = _local_smoke_config(
        out=out,
        embedding_backend=embedding_backend,
        person_model_path=person_model_path,
        scene_model_path=scene_model_path,
        inference_backend=inference_backend,
        pose_model_path=pose_model_path,
    )
    payloads_by_scene = {record["scene"]: record for record in payload_records}
    with visual_states_path.open("w", encoding="utf-8") as states_file:
        execution = _execute_local_smoke(
            scenes=scenes,
            payloads_by_scene=payloads_by_scene,
            out=out,
            camera=camera,
            config=local_config,
            states_file=states_file,
        )

    api_response_records = list(execution.get("api_response_records") or [])
    botified_frame_records = list(execution.get("botified_frame_records") or [])
    self_result = dict(execution.get("self_smoke") or _not_run_result())
    scene_result = dict(execution.get("scene_smoke") or _not_run_result())
    third_person_result = dict(
        execution.get("third_person_probe") or _not_run_result(status="insufficient_sample")
    )
    third_person_result = _with_local_third_person_debug_evidence(third_person_result)

    _write_json(
        teach_payloads_path,
        {
            "schema_version": 1,
            "mode": "local-smoke",
            "backend": "local",
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
        mode="local-smoke",
    )
    visual_evidence_index = [
        {
            "assertion_id": "memory_teaching_ga_local_smoke",
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
    checks = _build_local_smoke_checks(
        scenes=scenes,
        payload_records=payload_records,
        forbidden_payload_fields=forbidden_payload_fields,
        out=out,
        artifact_paths=artifact_paths,
        visual_evidence_index=visual_evidence_index,
        preflight=preflight,
        self_result=self_result,
        scene_result=scene_result,
        third_person_result=third_person_result,
    )
    ok = all(check["passed"] for check in checks)
    warnings = list(manifest.get("risks") or [])
    report = {
        "ok": ok,
        "status": "passed" if ok else "failed",
        "gate": "memory_teaching_ga_runner_local_smoke",
        "mode": "local-smoke",
        "backend": "local",
        "inference_backend": inference_backend,
        "real_model_evidence": True,
        "data_dir": str(data_dir),
        "out": str(out),
        "camera": camera,
        "scene_count": len(scenes),
        "scenes": [_scene_report(scene) for scene in scenes],
        "manifest": manifest,
        "warnings": warnings,
        "teach_requests": [_teach_request_summary(record) for record in payload_records],
        "forbidden_agent_payload_fields": forbidden_payload_fields,
        "self_smoke": self_result,
        "scene_smoke": scene_result,
        "third_person_probe": third_person_result,
        "api_responses": api_response_records,
        "debug_test_channel_enabled": False,
        "artifacts": artifact_paths,
        "visual_evidence_index": visual_evidence_index,
        "checks": checks,
        "notes": [
            "Local smoke mode uses real local embedding and pose inference backends with explicit model paths.",
            "Third-person local probe reports insufficient_sample when real YOLO/track/pose cannot resolve a target.",
            "No agent-facing teach payload adds track_id, bbox, point, source frame, or test hint fields.",
        ],
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.local_smoke:
        report = run_local_smoke(
            data_dir=args.data_dir,
            out=args.out,
            camera=args.camera,
            embedding_backend=args.embedding_backend,
            person_model_path=args.person_model_path,
            scene_model_path=args.scene_model_path,
            inference_backend=args.inference_backend,
            pose_model_path=args.pose_model_path,
        )
    elif args.dry_run:
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
    if report["mode"] == "local-smoke":
        print(f"self: {report.get('self_smoke', {}).get('status')}")
        print(f"scene: {report.get('scene_smoke', {}).get('status')}")
        print(f"third-person: {report.get('third_person_probe', {}).get('status')}")
    print(f"report: {Path(args.out) / 'report.json'}")
    return 0 if report["ok"] else 1


class _RecordingSession:
    def __init__(self, session: Any, owner: "_RecordingSessionFactory") -> None:
        self._session = session
        self._owner = owner

    async def process_frame(self, frame: Any) -> dict[str, Any]:
        return await self._session.process_frame(frame)

    def take_memory_frame_snapshot(self) -> Any | None:
        take_snapshot = getattr(self._session, "take_memory_frame_snapshot", None)
        snapshot = take_snapshot() if callable(take_snapshot) else None
        self._owner.last_snapshot = snapshot
        return snapshot


class _RecordingSessionFactory:
    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self.last_snapshot: Any | None = None

    def __call__(self) -> _RecordingSession:
        create_session = getattr(self._processor, "create_session", None)
        session = create_session() if callable(create_session) else self._processor
        wrapped = _RecordingSession(session, self)
        self.last_snapshot = None
        return wrapped


class LocalMemorySmokeRunner:
    def __init__(
        self,
        *,
        case: str,
        out: Path,
        camera: str,
        config: ServerConfig,
    ) -> None:
        self.case = case
        self.out = out
        self.camera = camera
        self.frame_id = 0
        case_runtime_dir = out / "runtime" / case
        case_config = _config_for_local_smoke_case(config, case_runtime_dir)
        self.processor = create_processor_from_config(case_config)
        self.session_factory = _RecordingSessionFactory(self.processor)
        self.client = TestClient(
            create_app(session_factory=self.session_factory, config=case_config)
        )

    def open_stream(self):
        return self.client.websocket_connect("/v1/stream")

    def send(
        self,
        websocket: Any,
        source_frame: memory_e2e.SourceFrame,
        *,
        timestamp_ms: int,
        states_file: Any,
        phase: str,
    ) -> dict[str, Any]:
        self.frame_id += 1
        header = {
            "type": "frame",
            "schema_version": SCHEMA_VERSION,
            "camera": self.camera,
            "frame_id": self.frame_id,
            "timestamp_ms": timestamp_ms,
            "encoding": "jpeg",
            "width": source_frame.width,
            "height": source_frame.height,
            "head_motion": {"state": "stationary"},
        }
        websocket.send_bytes(encode_frame_message(header, source_frame.jpeg_bytes))
        state = json.loads(websocket.receive_text())
        states_file.write(
            json.dumps(
                {
                    "case": self.case,
                    "phase": phase,
                    "source_frame": str(source_frame.path),
                    "visual_state": state,
                    "probe": _probe_from_state_and_snapshot(
                        state,
                        self.session_factory.last_snapshot,
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        states_file.flush()
        return state


def _execute_local_smoke(
    *,
    scenes: list[SceneDir],
    payloads_by_scene: dict[str, dict[str, Any]],
    out: Path,
    camera: str,
    config: ServerConfig,
    states_file: Any,
) -> dict[str, Any]:
    api_response_records: list[dict[str, Any]] = []
    botified_frame_records: list[dict[str, Any]] = []

    self_scene = _find_scene(scenes, "pic_teach_me")
    self_record = payloads_by_scene.get("pic_teach_me")
    if self_scene is None or self_record is None:
        self_result = _missing_teaching_scene_result("pic_teach_me")
        self_result["status"] = "failed"
    else:
        self_result = _run_local_self_smoke(
            out=out,
            scene=self_scene,
            record=self_record,
            camera=camera,
            config=config,
            states_file=states_file,
            api_response_records=api_response_records,
            botified_frame_records=botified_frame_records,
        )

    scene_scene = _find_scene(scenes, "pic_teach_scene_galbot")
    scene_record = payloads_by_scene.get("pic_teach_scene_galbot")
    if scene_scene is None or scene_record is None:
        scene_result = _missing_teaching_scene_result("pic_teach_scene_galbot")
        scene_result["status"] = "failed"
    else:
        scene_result = _run_local_scene_smoke(
            out=out,
            scene=scene_scene,
            record=scene_record,
            camera=camera,
            config=config,
            states_file=states_file,
            api_response_records=api_response_records,
            botified_frame_records=botified_frame_records,
        )

    third_scene = _find_scene(scenes, "pic_teach_person")
    third_record = payloads_by_scene.get("pic_teach_person")
    if third_scene is None or third_record is None:
        third_result = _with_local_third_person_debug_evidence(
            _insufficient_sample_result(
                reason="required_teaching_scene_missing",
                scene="pic_teach_person",
            )
        )
    else:
        third_result = _run_local_third_person_probe(
            out=out,
            scene=third_scene,
            record=third_record,
            camera=camera,
            config=config,
            states_file=states_file,
            api_response_records=api_response_records,
            botified_frame_records=botified_frame_records,
        )

    return {
        "self_smoke": self_result,
        "scene_smoke": scene_result,
        "third_person_probe": third_result,
        "api_response_records": api_response_records,
        "botified_frame_records": botified_frame_records,
    }


def _run_local_self_smoke(
    *,
    out: Path,
    scene: SceneDir,
    record: dict[str, Any],
    camera: str,
    config: ServerConfig,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = LocalMemorySmokeRunner(
        case="local-self-smoke",
        out=out,
        camera=camera,
        config=config,
    )
    last_reason = "no_active_interaction_target"
    last_probe: dict[str, Any] | None = None
    selected_frame: memory_e2e.SourceFrame | None = None
    selected_timestamp_ms: int | None = None
    teach: dict[str, Any] | None = None
    with runner.open_stream() as websocket:
        for index, source_frame in enumerate(_local_smoke_source_frames(scene)):
            timestamp_ms = 1_000 + (index * 400)
            state = runner.send(
                websocket,
                source_frame,
                timestamp_ms=timestamp_ms,
                states_file=states_file,
                phase="self-probe",
            )
            last_probe = _probe_from_state_and_snapshot(
                state,
                runner.session_factory.last_snapshot,
            )
            resolve = _post_and_record_api_response(
                runner=runner,
                api_response_records=api_response_records,
                payload_index=f"{_payload_index(record)}:local-resolve-self:{index}",
                scene=record["scene"],
                endpoint="/v1/memory/resolve-target",
                payload={"camera": camera, "target": record["payload"]["target"]},
                operation="local_resolve_self_target",
            )
            body = resolve["body"]
            if body.get("status") != "resolved":
                last_reason = _response_reason(body)
                continue
            teach = _post_and_record_api_response(
                runner=runner,
                api_response_records=api_response_records,
                payload_index=f"{_payload_index(record)}:local-teach-self",
                scene=record["scene"],
                endpoint=record["endpoint"],
                payload=record["payload"],
                operation="local_teach_person_self",
            )
            if teach["status_code"] >= 400 or teach["body"].get("ok") is not True:
                last_reason = _response_reason(teach["body"])
                teach = None
                continue
            selected_frame = source_frame
            selected_timestamp_ms = timestamp_ms
            break

        if selected_frame is None or selected_timestamp_ms is None or teach is None:
            return {
                "status": "failed",
                "passed": False,
                "reason": last_reason,
                "last_probe": last_probe,
            }

        events = _send_stable_query_and_drain_local(
            runner,
            websocket,
            selected_frame,
            base_timestamp_ms=selected_timestamp_ms,
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
        case="local-self-smoke",
        scene=scene.name,
        phase="self-replay",
        events=events,
    )
    passed = all(assertions.values())
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "reason": "" if passed else "known_person_present_not_replayed",
        "assertions": assertions,
        "person_id": teach["body"].get("person_id"),
        "teach_crop_hash": _teach_crop_hash(teach["body"]),
        "teach_crop_path_or_artifact_ref": _teach_crop_path_or_artifact_ref(
            teach["body"],
        ),
        "selected_window": _selected_window(scene, selected_frame),
        "last_probe": last_probe,
        "events": memory_e2e.compact_events(events),
    }


def _run_local_scene_smoke(
    *,
    out: Path,
    scene: SceneDir,
    record: dict[str, Any],
    camera: str,
    config: ServerConfig,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = LocalMemorySmokeRunner(
        case="local-scene-smoke",
        out=out,
        camera=camera,
        config=config,
    )
    source_frame = _load_source_frame_from_scene(scene)
    with runner.open_stream() as websocket:
        state = runner.send(
            websocket,
            source_frame,
            timestamp_ms=1_000,
            states_file=states_file,
            phase="scene-seed",
        )
        teach = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=f"{_payload_index(record)}:local-teach-scene",
            scene=record["scene"],
            endpoint=record["endpoint"],
            payload=record["payload"],
            operation="local_teach_scene",
        )
        if teach["status_code"] >= 400 or teach["body"].get("ok") is not True:
            return {
                "status": "failed",
                "passed": False,
                "reason": _response_reason(teach["body"]),
                "last_probe": _probe_from_state_and_snapshot(
                    state,
                    runner.session_factory.last_snapshot,
                ),
            }
        events = _send_stable_query_and_drain_local(
            runner,
            websocket,
            source_frame,
            base_timestamp_ms=1_000,
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
        case="local-scene-smoke",
        scene=scene.name,
        phase="scene-replay",
        events=events,
    )
    passed = all(assertions.values())
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "reason": "" if passed else "scene_activated_not_replayed",
        "assertions": assertions,
        "scene_id": teach["body"].get("scene_id"),
        "teach_crop_hash": _teach_crop_hash(teach["body"]),
        "teach_crop_path_or_artifact_ref": _teach_crop_path_or_artifact_ref(
            teach["body"],
        ),
        "selected_window": _selected_window(scene, source_frame),
        "events": memory_e2e.compact_events(events),
    }


def _run_local_third_person_probe(
    *,
    out: Path,
    scene: SceneDir,
    record: dict[str, Any],
    camera: str,
    config: ServerConfig,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    runner = LocalMemorySmokeRunner(
        case="local-third-person-probe",
        out=out,
        camera=camera,
        config=config,
    )
    observations: list[dict[str, Any]] = []
    last_reason = "no_active_interaction_target"
    invalid_resolve: dict[str, Any] | None = None
    with runner.open_stream() as websocket:
        for index, source_frame in enumerate(_local_third_person_source_frames(scene)):
            timestamp_ms = 1_000 + (index * 400)
            state = runner.send(
                websocket,
                source_frame,
                timestamp_ms=timestamp_ms,
                states_file=states_file,
                phase="third-person-probe",
            )
            probe = _probe_from_state_and_snapshot(
                state,
                runner.session_factory.last_snapshot,
            )
            observations.append(
                {
                    "frame": str(source_frame.path),
                    "frame_id": state.get("frame_id"),
                    **probe,
                }
            )
            resolve = _post_and_record_api_response(
                runner=runner,
                api_response_records=api_response_records,
                payload_index=f"{_payload_index(record)}:local-resolve-third:{index}",
                scene=record["scene"],
                endpoint="/v1/memory/resolve-target",
                payload={"camera": camera, "target": record["payload"]["target"]},
                operation="local_resolve_third_person_target",
            )
            body = resolve["body"]
            if body.get("status") != "resolved":
                last_reason = _response_reason(body)
                continue
            if not _third_person_resolve_has_pose_pointing_evidence(body):
                last_reason = "resolved_without_pose_pointing_evidence"
                invalid_resolve = body
                continue

            teach = _post_and_record_api_response(
                runner=runner,
                api_response_records=api_response_records,
                payload_index=f"{_payload_index(record)}:local-teach-third-person",
                scene=record["scene"],
                endpoint=record["endpoint"],
                payload=record["payload"],
                operation="local_teach_person_third_person",
            )
            if teach["status_code"] >= 400 or teach["body"].get("ok") is not True:
                pose_pointing_scoring = _third_person_pose_pointing_scoring(
                    teach["body"],
                    body,
                )
                return _with_local_third_person_debug_evidence(
                    {
                        "status": "failed",
                        "passed": False,
                        "reason": _response_reason(teach["body"]),
                        "resolve_target": body,
                        "person_id": teach["body"].get("person_id"),
                        "pose_pointing_scoring": pose_pointing_scoring,
                        "selected_window": _selected_window(scene, source_frame),
                        "events": [],
                        "observations": observations,
                    }
                )
            events = _send_stable_query_and_drain_local(
                runner,
                websocket,
                source_frame,
                base_timestamp_ms=timestamp_ms,
                states_file=states_file,
                phase="third-person-replay",
            )
            known = memory_e2e.first_event(events, "known_person_present")
            person_id = teach["body"].get("person_id")
            known_person_id = (
                known.get("memory_context", {}).get("person", {}).get("person_id")
                if known
                else None
            )
            resolve_evidence = _response_evidence(body)
            teach_evidence = _response_evidence(teach["body"])
            resolver_target_ref = (
                teach_evidence.get("resolver_target_ref")
                or resolve_evidence.get("resolver_target_ref")
            )
            introducer_ref = (
                teach_evidence.get("introducer_ref")
                or resolve_evidence.get("introducer_ref")
            )
            stored_embedding_source_track_ref = teach_evidence.get("source_track_ref")
            pose_pointing_scoring = _third_person_pose_pointing_scoring(
                teach["body"],
                body,
            )
            assertions = {
                "resolve_target_resolved": body.get("status") == "resolved",
                "resolve_target_pose_pointing": (
                    _third_person_resolve_has_pose_pointing_evidence(body)
                ),
                "teach_person_ok": teach["body"].get("ok") is True,
                "stored_embedding_source_is_target": bool(
                    stored_embedding_source_track_ref
                    and stored_embedding_source_track_ref == resolver_target_ref
                ),
                "stored_embedding_source_not_introducer": bool(
                    stored_embedding_source_track_ref
                    and introducer_ref
                    and stored_embedding_source_track_ref != introducer_ref
                ),
                "known_person_present": known is not None,
                "known_person_context": known_person_id == person_id,
                "target_not_introducer": _third_person_target_not_introducer(
                    body,
                    teach["body"],
                ),
                "pose_pointing_scoring_present": bool(pose_pointing_scoring),
                "pose_pointing_checks_passed": _pose_pointing_checks_passed(
                    pose_pointing_scoring,
                ),
            }
            if botified_frame_records is not None:
                _append_botified_frame_records(
                    botified_frame_records,
                    case="local-third-person-probe",
                    scene=scene.name,
                    phase="third-person-replay",
                    events=events,
                )
            passed = all(assertions.values())
            return _with_local_third_person_debug_evidence(
                {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "reason": "" if passed else _first_failed_assertion(assertions),
                    "assertions": assertions,
                    "resolve_target": body,
                    "person_id": person_id,
                    "resolver_target_ref": resolver_target_ref,
                    "introducer_ref": introducer_ref,
                    "pose_pointing_scoring": pose_pointing_scoring,
                    "stored_embedding_source_track_ref": (
                        stored_embedding_source_track_ref
                    ),
                    "stored_crop_hash": teach_evidence.get("crop_hash"),
                    "stored_crop_path_or_artifact_ref": teach_evidence.get(
                        "crop_path_or_artifact_ref"
                    ),
                    "selected_window": _selected_window(scene, source_frame),
                    "events": memory_e2e.compact_events(events),
                    "observations": observations,
                }
            )

    return _with_local_third_person_debug_evidence(
        {
            "status": "failed" if invalid_resolve is not None else "insufficient_sample",
            "passed": False,
            "reason": last_reason,
            "resolve_target": invalid_resolve,
            "person_id": None,
            "pose_pointing_scoring": (
                _third_person_pose_pointing_scoring(invalid_resolve)
                if invalid_resolve is not None
                else {}
            ),
            "selected_window": None,
            "events": [],
            "observations": observations,
        }
    )


def _third_person_resolve_has_pose_pointing_evidence(body: dict[str, Any]) -> bool:
    evidence = body.get("evidence") if isinstance(body.get("evidence"), dict) else {}
    candidates = (
        body.get("candidates") if isinstance(body.get("candidates"), list) else []
    )
    candidate_reason = ""
    if candidates and isinstance(candidates[0], dict):
        candidate_reason = str(candidates[0].get("reason") or "")
    has_pose_reason = (
        body.get("status") == "resolved"
        and (
            candidate_reason == "pose_pointing_to_person"
            or evidence.get("resolution_reason") == "pose_pointing_to_person"
        )
    )
    return has_pose_reason and bool(_third_person_pose_pointing_scoring(body))


def _third_person_target_not_introducer(
    resolve_body: dict[str, Any],
    teach_body: dict[str, Any],
) -> bool:
    for body in (resolve_body, teach_body):
        evidence = (
            body.get("evidence") if isinstance(body.get("evidence"), dict) else {}
        )
        target_ref = evidence.get("resolver_target_ref")
        introducer_ref = evidence.get("introducer_ref")
        if introducer_ref is None or target_ref is None:
            return False
        if target_ref == introducer_ref:
            return False
    return True


def _first_failed_assertion(assertions: dict[str, bool]) -> str:
    for name, passed in assertions.items():
        if not passed:
            return name
    return "assertion_failed"


def _send_stable_query_and_drain_local(
    runner: LocalMemorySmokeRunner,
    websocket: Any,
    source_frame: memory_e2e.SourceFrame,
    *,
    base_timestamp_ms: int,
    states_file: Any,
    phase: str,
) -> list[dict[str, Any]]:
    for offset_ms in (400, 800):
        runner.send(
            websocket,
            source_frame,
            timestamp_ms=base_timestamp_ms + offset_ms,
            states_file=states_file,
            phase=f"{phase}:warmup",
        )
    query_timestamp_ms = base_timestamp_ms + 1_200
    runner.send(
        websocket,
        source_frame,
        timestamp_ms=query_timestamp_ms,
        states_file=states_file,
        phase=f"{phase}:query",
    )
    for attempt in range(12):
        time.sleep(memory_e2e.QUERY_DRAIN_WAIT_SECONDS)
        drained = runner.send(
            websocket,
            source_frame,
            timestamp_ms=query_timestamp_ms + 1 + attempt,
            states_file=states_file,
            phase=f"{phase}:drain",
        )
        events = list(drained.get("semantic_events") or [])
        if events:
            return events
    return []


def _local_smoke_source_frames(
    scene: SceneDir,
    *,
    max_frames: int = 16,
) -> list[memory_e2e.SourceFrame]:
    paths = list(scene.jpeg_paths[:max_frames])
    if len(paths) == 1:
        paths = [paths[0], paths[0], paths[0]]
    return [_load_source_frame(path) for path in paths]


def _local_third_person_source_frames(
    scene: SceneDir,
) -> Iterator[memory_e2e.SourceFrame]:
    for path in scene.jpeg_paths:
        yield _load_source_frame(path)


def _load_source_frame(path: Path) -> memory_e2e.SourceFrame:
    jpeg_bytes = path.read_bytes()
    width, height = _parse_jpeg_dimensions(jpeg_bytes, frame_id=None)
    return memory_e2e.SourceFrame(
        path=path,
        jpeg_bytes=jpeg_bytes,
        width=width,
        height=height,
    )


def _selected_window(
    scene: SceneDir,
    source_frame: memory_e2e.SourceFrame,
) -> dict[str, Any]:
    try:
        frame_index = list(scene.jpeg_paths).index(source_frame.path)
    except ValueError:
        frame_index = None
    return {
        "scene": scene.name,
        "frame": str(source_frame.path),
        "frame_index": frame_index,
        "mode": "fixed_val_data_frame",
    }


def _probe_from_state_and_snapshot(
    state: dict[str, Any],
    snapshot: Any | None,
) -> dict[str, Any]:
    visual_tracks = list(state.get("tracks") or [])
    attention = state.get("attention") or {}
    snapshot_tracks = list(getattr(snapshot, "tracks", []) or [])
    keypoint_counts = []
    pointing_arm_counts = []
    for track in snapshot_tracks:
        keypoints = tuple(getattr(track, "keypoints", ()) or ())
        keypoint_counts.append(
            {
                "track_id": getattr(track, "track_id", None),
                "keypoint_count": len(keypoints),
            }
        )
        pointing_arm_counts.append(
            {
                "track_id": getattr(track, "track_id", None),
                "pointing_arm_count": _pointing_arm_count(keypoints),
            }
        )
    return {
        "track_count": len(visual_tracks),
        "visible_person_count": sum(
            1
            for track in visual_tracks
            if track.get("class") == "person" and int(track.get("lost_ms") or 0) == 0
        ),
        "attention_available": bool(attention),
        "attention_target_track_id": attention.get("target_track_id"),
        "scene_context": state.get("scene_context"),
        "keypoint_counts": keypoint_counts,
        "pointing_arm_counts": pointing_arm_counts,
    }


def _pointing_arm_count(keypoints: tuple[Any, ...]) -> int:
    by_name = {getattr(keypoint, "name", ""): keypoint for keypoint in keypoints}
    count = 0
    for side in ("left", "right"):
        if all(
            _valid_probe_keypoint(by_name.get(f"{side}_{joint}"))
            for joint in ("shoulder", "elbow", "wrist")
        ):
            count += 1
    return count


def _valid_probe_keypoint(keypoint: Any | None) -> bool:
    if keypoint is None:
        return False
    confidence = getattr(keypoint, "confidence", None)
    return confidence is None or float(confidence) >= 0.2


def _response_reason(body: dict[str, Any]) -> str:
    detail = body.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("code") or detail.get("message") or "request_failed")
    return str(
        body.get("error_code")
        or body.get("ambiguity_type")
        or body.get("status")
        or body.get("code")
        or "request_failed"
    )


def _response_evidence(body: dict[str, Any]) -> dict[str, Any]:
    evidence = body.get("evidence")
    return evidence if isinstance(evidence, dict) else {}


def _third_person_pose_pointing_scoring(
    *bodies: dict[str, Any],
) -> dict[str, Any]:
    for body in bodies:
        scoring = _response_evidence(body).get("pose_pointing_scoring")
        if isinstance(scoring, dict):
            return scoring
    return {}


def _pose_pointing_checks_passed(scoring: dict[str, Any]) -> bool:
    checks = scoring.get("checks") if isinstance(scoring, dict) else None
    if not isinstance(checks, dict):
        return False
    bool_values = [value for value in checks.values() if isinstance(value, bool)]
    return bool(bool_values) and all(bool_values)


def _local_third_person_debug_evidence() -> dict[str, Any]:
    return {
        "debug_test_channel_enabled": False,
        "fixture_inputs_consumed": [],
        "debug_fixture_used_for_target_resolution": False,
    }


def _with_local_third_person_debug_evidence(
    result: dict[str, Any],
) -> dict[str, Any]:
    return {**result, **_local_third_person_debug_evidence()}


def _teach_crop_hash(body: dict[str, Any]) -> Any:
    return _response_evidence(body).get("crop_hash")


def _teach_crop_path_or_artifact_ref(body: dict[str, Any]) -> Any:
    return _response_evidence(body).get("crop_path_or_artifact_ref")


def _local_smoke_preflight(
    *,
    embedding_backend: str,
    person_model_path: Path | None,
    scene_model_path: Path | None,
    inference_backend: str,
    pose_model_path: Path | None,
) -> dict[str, Any]:
    missing: list[str] = []
    invalid_paths: list[str] = []
    if embedding_backend != "local":
        missing.append("--embedding-backend local")
    if person_model_path is None:
        missing.append("--person-model-path")
    elif not person_model_path.exists():
        invalid_paths.append(f"--person-model-path={person_model_path}")
    if scene_model_path is None:
        missing.append("--scene-model-path")
    elif not scene_model_path.exists():
        invalid_paths.append(f"--scene-model-path={scene_model_path}")
    if inference_backend != "ultralytics":
        missing.append("--inference-backend ultralytics")
    if pose_model_path is None:
        missing.append("--pose-model-path")
    elif not pose_model_path.is_file():
        invalid_paths.append(f"--pose-model-path={pose_model_path}")
    return {
        "name": "local_smoke_explicit_real_backends",
        "passed": not missing and not invalid_paths,
        "details": {
            "missing": missing,
            "invalid_paths": invalid_paths,
            "embedding_backend": embedding_backend,
            "inference_backend": inference_backend,
            "person_model_path": str(person_model_path) if person_model_path else None,
            "scene_model_path": str(scene_model_path) if scene_model_path else None,
            "pose_model_path": str(pose_model_path) if pose_model_path else None,
        },
    }


def _local_smoke_preflight_failed_report(
    *,
    data_dir: Path,
    out: Path,
    camera: str,
    preflight: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "failed",
        "gate": "memory_teaching_ga_runner_local_smoke",
        "mode": "local-smoke",
        "backend": "local",
        "real_model_evidence": False,
        "data_dir": str(data_dir),
        "out": str(out),
        "camera": camera,
        "scene_count": 0,
        "forbidden_agent_payload_fields": {},
        "self_smoke": _not_run_result(),
        "scene_smoke": _not_run_result(),
        "third_person_probe": _with_local_third_person_debug_evidence(
            _not_run_result(status="insufficient_sample"),
        ),
        "artifacts": {"report_json": str(report_path.relative_to(out))},
        "checks": [preflight],
        "notes": [
            "Local smoke mode requires explicit local embedding and ultralytics pose model paths.",
        ],
    }


def _local_smoke_config(
    *,
    out: Path,
    embedding_backend: str,
    person_model_path: Path | None,
    scene_model_path: Path | None,
    inference_backend: str,
    pose_model_path: Path | None,
) -> ServerConfig:
    return ServerConfig(
        runtime_dir=out / "runtime",
        inference=InferenceConfig(
            backend=inference_backend,
            model_path=pose_model_path or Path("runtime/models/yolov8n-pose.pt"),
        ),
        memory=MemoryConfig(
            enabled=True,
            db_path=out / "runtime" / "memory.sqlite3",
            frame_cache_seconds=memory_e2e.FRAME_CACHE_SECONDS,
            query_interval_ms=memory_e2e.QUERY_INTERVAL_MS,
            queue_size=8,
            embedding=MemoryEmbeddingConfig(
                backend=embedding_backend,
                person_model_path=person_model_path,
                scene_model_path=scene_model_path,
            ),
            matching=MemoryMatchingConfig(event_cooldown_ms=memory_e2e.QUERY_INTERVAL_MS),
        ),
    )


def _config_for_local_smoke_case(config: ServerConfig, runtime_dir: Path) -> ServerConfig:
    return ServerConfig(
        runtime_dir=runtime_dir,
        inference=config.inference,
        tracking=config.tracking,
        attention=config.attention,
        events=config.events,
        metrics=config.metrics,
        memory=MemoryConfig(
            enabled=config.memory.enabled,
            db_path=runtime_dir / "memory.sqlite3",
            frame_cache_seconds=config.memory.frame_cache_seconds,
            query_interval_ms=config.memory.query_interval_ms,
            queue_size=config.memory.queue_size,
            embedding=config.memory.embedding,
            matching=config.memory.matching,
        ),
    )


def _not_run_result(*, status: str = "failed") -> dict[str, Any]:
    return {"status": status, "passed": False, "reason": "not_run"}


def _insufficient_sample_result(*, reason: str, scene: str) -> dict[str, Any]:
    return {
        "status": "insufficient_sample",
        "passed": False,
        "reason": reason,
        "scene": scene,
        "observations": [],
    }


class _StaticVisualStateServiceClient:
    def __init__(self) -> None:
        self.visual_state: dict[str, Any] = {}

    async def request_frame(self, _header: dict[str, Any], _jpeg: bytes) -> Any:
        return SimpleNamespace(visual_state=self.visual_state)


class _NoopGazePublisher:
    def publish(self, _payload: dict[str, Any]) -> None:
        return None


class _CliBotifiedProjectionRecorder:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self._stream = StringIO()
        self._writer = BotifiedStdoutWriter(stream=self._stream, max_queue_size=64)
        self._slot = LatestFrameSlot()
        self._service_client = _StaticVisualStateServiceClient()
        self._pump = FramePump(
            latest_frame_slot=self._slot,
            service_client=self._service_client,
            gaze_publisher=_NoopGazePublisher(),
            head_motion_provider=lambda: HeadMotion(state="stationary"),
            botified_writer=self._writer,
        )

    def record_visual_state(
        self,
        *,
        case: str,
        scene: str,
        phase: str,
        visual_state: dict[str, Any],
    ) -> None:
        semantic_events = visual_state.get("semantic_events")
        if not isinstance(semantic_events, list) or not semantic_events:
            return

        self._service_client.visual_state = visual_state
        self._slot.push(_input_frame_from_visual_state(visual_state))
        stream_position = self._stream.tell()
        asyncio.run(
            self._pump.process_one(
                now_ms=_int_value(visual_state.get("frame_timestamp_ms"), 0)
            )
        )
        self._writer.drain_available()

        self._stream.seek(stream_position)
        raw_frames = [
            line.strip() for line in self._stream.read().splitlines() if line.strip()
        ]
        self._stream.seek(0, 2)
        for raw_frame in raw_frames:
            payload = _payload_from_botified_frame(raw_frame)
            self._records.append(
                {
                    "case": case,
                    "scene": scene,
                    "phase": phase,
                    "dry_run": False,
                    "source": CLI_BOTIFIED_FRAME_SOURCE,
                    "botified_frame": raw_frame,
                    "payload": payload,
                    "event": _request_field(payload.get("request"), "event"),
                    "event_id": _event_id_from_payload(payload),
                }
            )


def _attach_cli_botified_projection_recorder(
    runner: Any,
    records: list[dict[str, Any]],
) -> None:
    recorder = _CliBotifiedProjectionRecorder(records)
    original_send = runner.send

    def send_with_cli_projection_recording(
        websocket: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        visual_state = original_send(websocket, *args, **kwargs)
        if isinstance(visual_state, dict):
            recorder.record_visual_state(
                case=str(getattr(runner, "case", "")),
                scene=_runner_source_scene(runner),
                phase=str(kwargs.get("phase") or ""),
                visual_state=visual_state,
            )
        return visual_state

    runner.send = send_with_cli_projection_recording


def _runner_source_scene(runner: Any) -> str:
    source_frame = getattr(runner, "source_frame", None)
    path = getattr(source_frame, "path", None)
    return path.parent.name if isinstance(path, Path) else ""


def _input_frame_from_visual_state(visual_state: dict[str, Any]) -> InputFrame:
    width, height = _image_size_from_visual_state(visual_state)
    return InputFrame(
        camera=str(visual_state.get("camera") or DEFAULT_CAMERA),
        timestamp_ms=_int_value(visual_state.get("frame_timestamp_ms"), 0),
        width=width,
        height=height,
        jpeg=b"",
    )


def _image_size_from_visual_state(visual_state: dict[str, Any]) -> tuple[int, int]:
    image_size = visual_state.get("image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return _int_value(image_size[0], 0), _int_value(image_size[1], 0)
    return 0, 0


def _payload_from_botified_frame(raw_frame: str) -> dict[str, Any]:
    if not raw_frame.startswith(BOTIFIED_OPEN) or not raw_frame.endswith(BOTIFIED_CLOSE):
        return {}
    inner = raw_frame[len(BOTIFIED_OPEN) : -len(BOTIFIED_CLOSE)]
    try:
        payload = json.loads(inner)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _event_id_from_payload(payload: dict[str, Any]) -> str | None:
    payload_id = payload.get("id")
    if isinstance(payload_id, str) and payload_id.startswith("visual:"):
        return payload_id.removeprefix("visual:")
    return None


def _request_field(request: Any, field: str) -> str | None:
    if not isinstance(request, str):
        return None
    marker = f"{field}="
    for part in request.split(" "):
        if part.startswith(marker):
            return part[len(marker) :]
    return None


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


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
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
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
    return {
        "passed": len(replayed_scene_names) == len(scenes),
        "replayed_scene_names": replayed_scene_names,
        "replayed_scene_count": len(replayed_scene_names),
    }


def _run_actual_post_teach_scene_replay(
    *,
    scenes: list[SceneDir],
    payload_records_by_scene: dict[str, dict[str, Any]],
    out: Path,
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    required_scene_names = (
        "pic_teach_me",
        "pic_teach_person",
        "pic_teach_scene_galbot",
    )
    scenes_by_name = {scene.name: scene for scene in scenes}
    missing = [
        name
        for name in required_scene_names
        if name not in scenes_by_name or name not in payload_records_by_scene
    ]
    if missing:
        return _post_teach_scene_replay_missing_result(missing)

    runner = _actual_runner(
        case=POST_TEACH_SCENE_REPLAY_CASE,
        out=out,
        source_frame=_load_source_frame_from_scene(scenes_by_name["pic_teach_me"]),
        camera=camera,
    )
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
    timestamp_ms = memory_e2e.QUERY_INTERVAL_MS
    timestamp_step_ms = memory_e2e.QUERY_INTERVAL_MS * 2

    with runner.open_stream() as websocket:
        self_record = payload_records_by_scene["pic_teach_me"]
        runner.processor.mode = "single"
        runner.source_frame = _load_source_frame_from_scene(
            scenes_by_name["pic_teach_me"]
        )
        runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=timestamp_ms,
            states_file=states_file,
            phase="post-teach-self-seed",
        )
        last_query_timestamp_ms = timestamp_ms
        self_teach = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=f"{_payload_index(self_record)}:post-teach",
            scene=self_record["scene"],
            endpoint=self_record["endpoint"],
            payload=self_record["payload"],
            operation="post_teach_person_self",
        )
        timestamp_ms += timestamp_step_ms

        third_person_record = payload_records_by_scene["pic_teach_person"]
        runner.processor.mode = "third_person"
        runner.source_frame = _load_source_frame_from_scene(
            scenes_by_name["pic_teach_person"]
        )
        _seed_stable_interaction_window(
            runner,
            websocket,
            start_timestamp_ms=(
                last_query_timestamp_ms + memory_e2e.QUERY_INTERVAL_MS - 2
            ),
            states_file=states_file,
            phase="post-teach-third-person-seed",
        )
        third_person_teach = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=f"{_payload_index(third_person_record)}:post-teach",
            scene=third_person_record["scene"],
            endpoint=third_person_record["endpoint"],
            payload=third_person_record["payload"],
            operation="post_teach_person_third_person",
        )
        timestamp_ms += timestamp_step_ms

        scene_record = payload_records_by_scene["pic_teach_scene_galbot"]
        runner.processor.mode = "single"
        runner.source_frame = _load_source_frame_from_scene(
            scenes_by_name["pic_teach_scene_galbot"]
        )
        runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=timestamp_ms,
            states_file=states_file,
            phase="post-teach-scene-seed",
        )
        scene_teach = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=f"{_payload_index(scene_record)}:post-teach",
            scene=scene_record["scene"],
            endpoint=scene_record["endpoint"],
            payload=scene_record["payload"],
            operation="post_teach_scene",
        )
        timestamp_ms += timestamp_step_ms

        self_person_id = self_teach["body"].get("person_id")
        third_person_id = third_person_teach["body"].get("person_id")
        scene_id = scene_teach["body"].get("scene_id")
        replayed_scene_names: list[str] = []
        replayed_scenes: list[dict[str, Any]] = []

        for scene in scenes:
            runner.source_frame = _load_source_frame_from_scene(scene)
            runner.processor.mode = (
                "third_person" if scene.name == "pic_teach_person" else "single"
            )
            events = runner.start_query_and_drain(
                websocket,
                query_timestamp_ms=timestamp_ms,
                states_file=states_file,
                phase=f"post-teach-scene-replay:{scene.name}",
            )
            replayed_scene_names.append(scene.name)
            flags = {
                "taught_self_known_person_present": (
                    _known_person_event_for_person(events, self_person_id) is not None
                ),
                "taught_third_person_known_person_present": (
                    _known_person_event_for_person(events, third_person_id) is not None
                ),
                "taught_scene_activated": (
                    _scene_event_for_scene(events, scene_id) is not None
                ),
            }
            replayed_scenes.append(
                {
                    "scene": scene.name,
                    "events": memory_e2e.compact_events(events),
                    "flags": flags,
                }
            )
            timestamp_ms += timestamp_step_ms

    scenes_by_result_name = {item["scene"]: item for item in replayed_scenes}
    assertions = {
        "all_required_teaching_scenes_present": True,
        "all_scenes_replayed": replayed_scene_names == [scene.name for scene in scenes],
        "self_positive_scene_confirmed": bool(
            scenes_by_result_name.get("pic_teach_me", {})
            .get("flags", {})
            .get("taught_self_known_person_present")
        ),
        "third_person_positive_scene_confirmed": bool(
            scenes_by_result_name.get("pic_teach_person", {})
            .get("flags", {})
            .get("taught_third_person_known_person_present")
        ),
        "scene_positive_scene_confirmed": bool(
            scenes_by_result_name.get("pic_teach_scene_galbot", {})
            .get("flags", {})
            .get("taught_scene_activated")
        ),
        "non_self_scenes_no_taught_self_confirmed": all(
            not item["flags"]["taught_self_known_person_present"]
            for item in replayed_scenes
            if item["scene"] != "pic_teach_me"
        ),
        "non_third_person_scenes_no_taught_third_person_confirmed": all(
            not item["flags"]["taught_third_person_known_person_present"]
            for item in replayed_scenes
            if item["scene"] != "pic_teach_person"
        ),
        "non_scene_scenes_no_taught_scene_confirmed": all(
            not item["flags"]["taught_scene_activated"]
            for item in replayed_scenes
            if item["scene"] != "pic_teach_scene_galbot"
        ),
    }
    passed = all(assertions.values())
    return {
        "runner_case": POST_TEACH_SCENE_REPLAY_CASE,
        "passed": passed,
        "reason": "" if passed else _first_failed_assertion(assertions),
        "replayed_scene_names": replayed_scene_names,
        "replayed_scene_count": len(replayed_scene_names),
        "self_person_id": self_person_id,
        "third_person_id": third_person_id,
        "scene_id": scene_id,
        "scenes": replayed_scenes,
        "assertions": assertions,
    }


def _post_teach_scene_replay_missing_result(missing: list[str]) -> dict[str, Any]:
    assertions = {
        "all_required_teaching_scenes_present": False,
        "all_scenes_replayed": False,
        "self_positive_scene_confirmed": False,
        "third_person_positive_scene_confirmed": False,
        "scene_positive_scene_confirmed": False,
        "non_self_scenes_no_taught_self_confirmed": False,
        "non_third_person_scenes_no_taught_third_person_confirmed": False,
        "non_scene_scenes_no_taught_scene_confirmed": False,
    }
    return {
        "runner_case": POST_TEACH_SCENE_REPLAY_CASE,
        "passed": False,
        "reason": "required_teaching_scene_missing",
        "missing": missing,
        "replayed_scene_names": [],
        "replayed_scene_count": 0,
        "self_person_id": None,
        "third_person_id": None,
        "scene_id": None,
        "scenes": [],
        "assertions": assertions,
    }


def _seed_stable_interaction_window(
    runner: memory_e2e.MemoryE2ERunner,
    websocket: Any,
    *,
    start_timestamp_ms: int,
    states_file: Any,
    phase: str,
) -> dict[str, Any]:
    client = getattr(runner, "client", None)
    app = getattr(client, "app", None)
    state_obj = getattr(app, "state", None)
    memory_service = getattr(state_obj, "memory_service", None)
    if memory_service is not None:
        # Seed frames establish the request interaction window only; avoid
        # kicking off recognition queries before the teach/resolve request.
        memory_service._last_query_frame_timestamp_ms[runner.camera] = (  # noqa: SLF001
            start_timestamp_ms
        )
    state: dict[str, Any] = {}
    for index in range(2):
        state = runner.send(
            websocket,
            timestamp_ms=start_timestamp_ms + index,
            states_file=states_file,
            phase=f"{phase}:stable-{index + 1}",
        )
    return state


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
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
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
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "person_id": teach["body"].get("person_id"),
        "teach_crop_hash": _teach_crop_hash(teach["body"]),
        "teach_crop_path_or_artifact_ref": _teach_crop_path_or_artifact_ref(
            teach["body"],
        ),
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
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
    runner.processor.mode = "third_person"
    resolve_payload = {
        "camera": camera,
        "target": record["payload"]["target"],
    }
    with runner.open_stream() as websocket:
        _seed_stable_interaction_window(
            runner,
            websocket,
            start_timestamp_ms=1_000,
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
        b_positive_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=2_000,
            states_file=states_file,
            phase="third-person-b-positive-replay",
        )
        runner.processor.mode = "single"
        a_only_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=3_000,
            states_file=states_file,
            phase="third-person-a-only-negative-replay",
        )
    stored_person_id = teach["body"].get("person_id")
    b_positive_known = _known_person_event_for_person(
        b_positive_events,
        stored_person_id,
    )
    a_only_known = _known_person_event_for_person(
        a_only_events,
        stored_person_id,
    )
    candidates = resolve["body"].get("candidates") or []
    resolved_track_id = candidates[0].get("track_id") if candidates else None
    resolve_evidence = (
        resolve["body"].get("evidence")
        if isinstance(resolve["body"].get("evidence"), dict)
        else {}
    )
    teach_evidence = (
        teach["body"].get("evidence")
        if isinstance(teach["body"].get("evidence"), dict)
        else {}
    )
    resolver_target_ref = (
        teach_evidence.get("resolver_target_ref")
        or resolve_evidence.get("resolver_target_ref")
    )
    introducer_ref = (
        teach_evidence.get("introducer_ref") or resolve_evidence.get("introducer_ref")
    )
    stored_embedding_source_track_ref = teach_evidence.get("source_track_ref")
    stored_crop_hash = teach_evidence.get("crop_hash")
    stored_crop_path_or_artifact_ref = teach_evidence.get(
        "crop_path_or_artifact_ref"
    )
    pose_pointing_scoring = _third_person_pose_pointing_scoring(
        teach["body"],
        resolve["body"],
    )
    b_positive_replay = _known_person_replay_summary(
        b_positive_events,
        stored_person_id,
    )
    a_only_negative_replay = _known_person_replay_summary(
        a_only_events,
        stored_person_id,
    )
    assertions = {
        "resolve_target_ok": resolve["body"].get("ok") is True,
        "resolve_target_resolved": resolve["body"].get("status") == "resolved",
        "resolver_selected_b": resolved_track_id == memory_e2e.AMBIGUOUS_TRACK_ID,
        "teach_person_ok": teach["body"].get("ok") is True,
        "stored_embedding_source_is_target": bool(
            stored_embedding_source_track_ref
            and stored_embedding_source_track_ref == resolver_target_ref
        ),
        "stored_embedding_source_not_introducer": bool(
            stored_embedding_source_track_ref
            and introducer_ref
            and stored_embedding_source_track_ref != introducer_ref
        ),
        "known_person_present": b_positive_known is not None,
        "known_person_is_b": bool(
            b_positive_known
            and b_positive_known.get("track_id") == memory_e2e.AMBIGUOUS_TRACK_ID
        ),
        "b_positive_known_person_present": b_positive_known is not None,
        "a_only_no_known_person_for_stored_person": a_only_known is None,
        "pose_pointing_scoring_present": bool(pose_pointing_scoring),
        "pose_pointing_checks_passed": _pose_pointing_checks_passed(
            pose_pointing_scoring,
        ),
    }
    return {
        **_local_third_person_debug_evidence(),
        "passed": all(assertions.values()),
        "assertions": assertions,
        "resolve_target": resolve["body"],
        "person_id": stored_person_id,
        "resolver_target_ref": resolver_target_ref,
        "introducer_ref": introducer_ref,
        "pose_pointing_scoring": pose_pointing_scoring,
        "stored_person_id": stored_person_id,
        "stored_embedding_source_track_ref": stored_embedding_source_track_ref,
        "stored_crop_hash": stored_crop_hash,
        "stored_crop_path_or_artifact_ref": stored_crop_path_or_artifact_ref,
        "b_positive_replay": b_positive_replay,
        "a_only_negative_replay": a_only_negative_replay,
        "events": b_positive_replay["events"],
    }


def _known_person_event_for_person(
    events: list[dict[str, Any]],
    person_id: Any,
) -> dict[str, Any] | None:
    if not person_id:
        return None
    for event in events:
        if event.get("event") != "known_person_present":
            continue
        context_person_id = (
            event.get("memory_context", {}).get("person", {}).get("person_id")
        )
        if context_person_id == person_id:
            return event
    return None


def _scene_event_for_scene(
    events: list[dict[str, Any]],
    scene_id: Any,
) -> dict[str, Any] | None:
    if not scene_id:
        return None
    for event in events:
        if event.get("event") != "scene_activated":
            continue
        context_scene_id = (
            event.get("memory_context", {}).get("scene", {}).get("scene_id")
        )
        if context_scene_id == scene_id:
            return event
    return None


def _known_person_replay_summary(
    events: list[dict[str, Any]],
    stored_person_id: Any,
) -> dict[str, Any]:
    stored_person_event = _known_person_event_for_person(events, stored_person_id)
    summary = {
        "known_person_present": memory_e2e.first_event(
            events,
            "known_person_present",
        )
        is not None,
        "stored_person_known_person_present": stored_person_event is not None,
        "events": memory_e2e.compact_events(events),
    }
    if stored_person_event is not None:
        summary["stored_person_track_id"] = stored_person_event.get("track_id")
    return summary


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
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
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
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "scene_id": teach["body"].get("scene_id"),
        "teach_crop_hash": _teach_crop_hash(teach["body"]),
        "teach_crop_path_or_artifact_ref": _teach_crop_path_or_artifact_ref(
            teach["body"],
        ),
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
    store_delta = memory_e2e._memory_store_delta(before_counts, after_counts)
    body = response["body"]
    assertions = {
        "status_code_200": response["status_code"] == 200,
        "status_not_found": body.get("status") == "not_found",
        "unsupported_target_kind": body.get("error_code") == "unsupported_target_kind",
        "no_candidates": body.get("candidates") == [],
        "no_memory_write": memory_e2e._no_memory_write(store_delta),
    }
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "resolve_target": body,
        "store_delta": store_delta,
        "store_delta_source": memory_e2e._memory_store_delta_source(),
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
    runner: Any,
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
    post_teach_scene_replay_result: dict[str, Any],
    object_result: dict[str, Any],
    botified_frame_records: list[dict[str, Any]],
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
    botified_event_counts: dict[str, int] = {}
    for record in botified_frame_records:
        event = record.get("event")
        if isinstance(event, str):
            botified_event_counts[event] = botified_event_counts.get(event, 0) + 1
    botified_projection_assertions = {
        "known_person_present": botified_event_counts.get("known_person_present", 0)
        >= 1,
        "scene_activated": botified_event_counts.get("scene_activated", 0) >= 1,
        "all_from_cli_frame_pump_stdout": bool(botified_frame_records)
        and all(
            record.get("source") == CLI_BOTIFIED_FRAME_SOURCE
            for record in botified_frame_records
        ),
        "all_raw_frames_wrapped": bool(botified_frame_records)
        and all(
            isinstance(record.get("botified_frame"), str)
            and record["botified_frame"].startswith(BOTIFIED_OPEN)
            and record["botified_frame"].endswith(BOTIFIED_CLOSE)
            for record in botified_frame_records
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
            "name": "cli_projection_botified_frames",
            "passed": all(botified_projection_assertions.values()),
            "details": {
                "assertions": botified_projection_assertions,
                "frame_count": len(botified_frame_records),
                "event_counts": dict(sorted(botified_event_counts.items())),
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
            "name": "post_teach_all_scenes_memory_behavior",
            "passed": bool(post_teach_scene_replay_result.get("passed")),
            "details": post_teach_scene_replay_result,
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


def _build_local_smoke_checks(
    *,
    scenes: list[SceneDir],
    payload_records: list[dict[str, Any]],
    forbidden_payload_fields: dict[str, list[str]],
    out: Path,
    artifact_paths: dict[str, str],
    visual_evidence_index: list[dict[str, Any]],
    preflight: dict[str, Any],
    self_result: dict[str, Any],
    scene_result: dict[str, Any],
    third_person_result: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_teach_scenes = {
        "pic_teach_me",
        "pic_teach_person",
        "pic_teach_scene_galbot",
    }
    actual_teach_scenes = {record["scene"] for record in payload_records}
    artifact_exists = {
        key: (out / relative_path).exists()
        for key, relative_path in artifact_paths.items()
        if key != "report_json"
    }
    evidence_exists = {
        item["path"]: (out / item["path"]).is_file() for item in visual_evidence_index
    }
    return [
        preflight,
        {
            "name": "discover_jpeg_scene_dirs",
            "passed": bool(scenes),
            "details": {"scene_count": len(scenes)},
        },
        {
            "name": "expected_local_smoke_scenes",
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
            "name": "self_local_smoke",
            "passed": self_result.get("status") == "passed",
            "details": self_result,
        },
        {
            "name": "scene_local_smoke",
            "passed": scene_result.get("status") == "passed",
            "details": scene_result,
        },
        {
            "name": "third_person_local_probe",
            "passed": (
                third_person_result.get("status") == "passed"
                and bool(third_person_result.get("passed"))
            ),
            "details": third_person_result,
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
