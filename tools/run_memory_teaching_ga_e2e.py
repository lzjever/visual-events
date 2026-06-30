from __future__ import annotations

import argparse
import asyncio
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

from tools import memory_teaching_evidence
from tools import run_memory_e2e as memory_e2e
from visual_events_cli.botified_output import (
    BotifiedStdoutWriter,
    build_current_visual_snapshot,
)
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
PAYLOAD_FIXTURE_STREAM_REF = memory_e2e.PAYLOAD_FIXTURE_STREAM_REF

JPEG_SUFFIXES = {".jpeg", ".jpg"}
JPEG_SUFFIX_ORDER = (".jpeg", ".jpg")
TRANSCRIPT_SUFFIX = ".transcript"
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
EVIDENCE_ONLY_ARTIFACT_KEYS = {"current_visual_snapshot_json"}
EVIDENCE_ONLY_API_OPERATIONS = {"supporting_identify_current"}
BOTIFIED_OPEN = "<botified>"
BOTIFIED_CLOSE = "</botified>"
BOUNDED_MULTI_PERSON_RECOGNITION_FIELDS = (
    "tracks_seen",
    "tracks_eligible",
    "tracks_queried",
    "tracks_skipped_reason",
    "queried_track_ids",
    "attention_target_track_id",
    "attention_target_only",
    "max_tracks_per_tick",
    "query_interval_ms",
    "event_cooldown_ms",
    "recognition_runs_in_executor",
)


@dataclass(frozen=True)
class InteractionCase:
    scene: str
    source_text_path: Path
    source_image_path: Path
    transcript_text: str


@dataclass(frozen=True)
class SceneDir:
    name: str
    path: Path
    jpeg_paths: tuple[Path, ...]
    interactions: tuple[InteractionCase, ...] = ()

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
    return _build_teach_payload_records_from_scenes(
        discover_scene_dirs(data_dir),
        camera=camera,
    )


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
    memory_teaching_evidence.write_visual_evidence_index(
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
    current_snapshot_path = out / "current_visual_snapshot.json"
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
    supporting_contracts_result: dict[str, Any] = {
        "passed": False,
        "error": "not_run",
    }

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

        supporting_contracts_result = _run_actual_supporting_contracts(
            scenes=scenes,
            out=out,
            camera=camera,
            states_file=states_file,
            api_response_records=api_response_records,
            botified_frame_records=botified_frame_records,
        )

    self_result = _with_record_source_fields(
        self_result,
        payload_records_by_scene.get("pic_teach_me"),
    )
    third_person_result = _with_record_source_fields(
        third_person_result,
        payload_records_by_scene.get("pic_teach_person"),
    )
    scene_result = _with_record_source_fields(
        scene_result,
        payload_records_by_scene.get("pic_teach_scene_galbot"),
    )
    object_result = _with_record_source_fields(
        object_result,
        payload_records_by_scene.get("pic_teach_item_phone"),
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
    _write_current_visual_snapshot_artifact(visual_states_path, current_snapshot_path)
    visual_evidence_index = [
        {
            "assertion_id": "memory_teaching_ga_actual_fake",
            "kind": "html_index",
            "path": "visual-evidence/index.html",
        }
    ]
    visual_evidence_index.extend(
        memory_teaching_evidence.build_runner_visual_overlay_items(
            out=out,
            scenes=scenes,
            mode="actual",
            self_result=self_result,
            third_person_result=third_person_result,
            scene_result=scene_result,
            object_result=object_result,
        )
    )
    memory_teaching_evidence.write_visual_evidence_index(
        evidence_index_path,
        scenes=scenes,
        payload_records=payload_records,
        manifest=manifest,
        mode="actual",
        visual_evidence_index=visual_evidence_index,
    )
    artifact_paths = {
        "report_json": "report.json",
        "timeline_jsonl": "timeline.jsonl",
        "teach_payloads_json": "teach_payloads.json",
        "api_responses_jsonl": "api_responses.jsonl",
        "botified_frames_jsonl": "botified_frames.jsonl",
        "visual_states_jsonl": "visual_states.jsonl",
        "current_visual_snapshot_json": "current_visual_snapshot.json",
        "visual_evidence_index_html": "visual-evidence/index.html",
        "runtime_dir": "runtime",
    }
    _write_jsonl(timeline_path, timeline_records)
    bounded_multi_person_recognition = _bounded_multi_person_recognition_from_results(
        third_person_result,
        post_teach_scene_replay_result,
    )
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
        supporting_contracts_result=supporting_contracts_result,
        botified_frame_records=botified_frame_records,
        bounded_multi_person_recognition=bounded_multi_person_recognition,
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
        "bounded_multi_person_recognition": bounded_multi_person_recognition,
        "object_no_write": object_result,
        "supporting_contracts": supporting_contracts_result,
        "debug_test_channel_enabled": False,
        "artifacts": artifact_paths,
        "visual_evidence_index": visual_evidence_index,
        "checks": checks,
        "notes": [
            "Actual fake mode: in-process FastAPI/TestClient server with deterministic fake memory embeddings.",
            "No local model backend, CLI subprocess, or schema changes are used.",
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
    current_snapshot_path = out / "current_visual_snapshot.json"
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
    self_result = _with_record_source_fields(
        self_result,
        payloads_by_scene.get("pic_teach_me"),
    )
    scene_result = _with_record_source_fields(
        scene_result,
        payloads_by_scene.get("pic_teach_scene_galbot"),
    )
    third_person_result = _with_record_source_fields(
        third_person_result,
        payloads_by_scene.get("pic_teach_person"),
    )

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
    _write_current_visual_snapshot_artifact(visual_states_path, current_snapshot_path)
    visual_evidence_index = [
        {
            "assertion_id": "memory_teaching_ga_local_smoke",
            "kind": "html_index",
            "path": "visual-evidence/index.html",
        }
    ]
    visual_evidence_index.extend(
        memory_teaching_evidence.build_runner_visual_overlay_items(
            out=out,
            scenes=scenes,
            mode="local-smoke",
            self_result=self_result,
            third_person_result=third_person_result,
            scene_result=scene_result,
            object_result=None,
        )
    )
    memory_teaching_evidence.write_visual_evidence_index(
        evidence_index_path,
        scenes=scenes,
        payload_records=payload_records,
        manifest=manifest,
        mode="local-smoke",
        visual_evidence_index=visual_evidence_index,
    )
    artifact_paths = {
        "report_json": "report.json",
        "timeline_jsonl": "timeline.jsonl",
        "teach_payloads_json": "teach_payloads.json",
        "api_responses_jsonl": "api_responses.jsonl",
        "botified_frames_jsonl": "botified_frames.jsonl",
        "visual_states_jsonl": "visual_states.jsonl",
        "current_visual_snapshot_json": "current_visual_snapshot.json",
        "visual_evidence_index_html": "visual-evidence/index.html",
        "runtime_dir": "runtime",
    }
    _write_jsonl(timeline_path, timeline_records)
    bounded_multi_person_recognition = _bounded_multi_person_recognition_from_results(
        third_person_result,
    )
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
        bounded_multi_person_recognition=bounded_multi_person_recognition,
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
        "bounded_multi_person_recognition": bounded_multi_person_recognition,
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
        self.latest_stream_ref: str | None = None
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
        stream_ref = state.get("stream_ref")
        if isinstance(stream_ref, str) and stream_ref:
            self.latest_stream_ref = stream_ref
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
            teach = _post_teach_person_recording_outcome(
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
        **_teach_person_report_fields(teach),
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

            teach = _post_teach_person_recording_outcome(
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
                        **_teach_person_report_fields(teach),
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
            bounded_multi_person_recognition = (
                _latest_bounded_recognition_report_from_runner(
                    runner,
                    camera=camera,
                )
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
                    **_teach_person_report_fields(teach),
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
                    "bounded_multi_person_recognition": (
                        bounded_multi_person_recognition
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


def _teach_person_report_fields(response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body")
    if not isinstance(body, dict):
        return {}
    fields: dict[str, Any] = {}
    person_visual_evidence = _person_visual_evidence_from_teach_body(body)
    if person_visual_evidence is not None:
        fields["person_visual_evidence"] = person_visual_evidence
    if "teach_person_outcome" in body:
        fields["teach_person_outcome"] = body["teach_person_outcome"]
    teach_person = body.get("teach_person")
    if teach_person is not None:
        fields["teach_person"] = teach_person
    merge_anonymous_person = body.get("merge_anonymous_person")
    if merge_anonymous_person is not None:
        fields["merge_anonymous_person"] = merge_anonymous_person
    return fields


def _person_visual_evidence_from_teach_body(
    body: dict[str, Any],
) -> dict[str, Any] | None:
    evidence = _response_evidence(body)
    person_visual_evidence = evidence.get("person_visual_evidence")
    if isinstance(person_visual_evidence, dict):
        return person_visual_evidence
    teach_person = body.get("teach_person")
    if isinstance(teach_person, dict):
        teach_evidence = _response_evidence(teach_person)
        person_visual_evidence = teach_evidence.get("person_visual_evidence")
        if isinstance(person_visual_evidence, dict):
            return person_visual_evidence
    return None


def _prefixed_teach_person_report_fields(
    prefix: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in _teach_person_report_fields(response).items()
    }


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
        "bounded_multi_person_recognition": {},
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


def _latest_bounded_recognition_report_from_runner(
    runner: Any,
    *,
    camera: str,
) -> dict[str, Any]:
    client = getattr(runner, "client", None)
    app = getattr(client, "app", None)
    state_obj = getattr(app, "state", None)
    memory_service = getattr(state_obj, "memory_service", None)
    latest_report = getattr(memory_service, "latest_recognition_report", None)
    if not callable(latest_report):
        return {}
    report = latest_report(camera)
    if not isinstance(report, dict):
        return {}
    return _bounded_multi_person_recognition_copy(report)


def _bounded_multi_person_recognition_from_results(
    *results: dict[str, Any],
) -> dict[str, Any]:
    for result in results:
        if not isinstance(result, dict):
            continue
        report = result.get("bounded_multi_person_recognition")
        if isinstance(report, dict) and report:
            return _bounded_multi_person_recognition_copy(report)
    return {}


def _bounded_multi_person_recognition_copy(report: dict[str, Any]) -> dict[str, Any]:
    copied = dict(report)
    skipped = copied.get("tracks_skipped_reason")
    copied["tracks_skipped_reason"] = dict(skipped) if isinstance(skipped, dict) else {}
    candidate_track_ids = copied.get("candidate_track_ids")
    if isinstance(candidate_track_ids, (list, tuple)):
        copied["candidate_track_ids"] = list(candidate_track_ids)
    else:
        copied["candidate_track_ids"] = []
    queried_track_ids = copied.get("queried_track_ids")
    if isinstance(queried_track_ids, (list, tuple)):
        copied["queried_track_ids"] = list(queried_track_ids)
    else:
        copied["queried_track_ids"] = []
    return copied


def _bounded_multi_person_recognition_check(
    report: dict[str, Any],
    *,
    require_non_attention_query: bool,
) -> dict[str, Any]:
    details = (
        _bounded_multi_person_recognition_copy(report)
        if isinstance(report, dict)
        else {}
    )
    missing_fields = [
        field
        for field in BOUNDED_MULTI_PERSON_RECOGNITION_FIELDS
        if field not in details
    ]
    tracks_queried = _int_value(details.get("tracks_queried"), -1)
    max_tracks_per_tick = _int_value(details.get("max_tracks_per_tick"), -1)
    tracks_eligible = _int_value(details.get("tracks_eligible"), 0)
    queried_track_ids = details.get("queried_track_ids")
    if not isinstance(queried_track_ids, list):
        queried_track_ids = []
    attention_target_track_id = details.get("attention_target_track_id")
    non_attention_queried = any(
        track_id != attention_target_track_id for track_id in queried_track_ids
    )
    passed = (
        not missing_fields
        and details.get("attention_target_only") is False
        and details.get("recognition_runs_in_executor") is True
        and tracks_queried >= 1
        and tracks_eligible >= 1
        and max_tracks_per_tick >= 1
        and tracks_queried <= max_tracks_per_tick
        and bool(queried_track_ids)
        and (non_attention_queried or not require_non_attention_query)
    )
    return {
        "name": "bounded_multi_person_recognition",
        "passed": passed,
        "details": details,
    }


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

    self_record = payload_records_by_scene["pic_teach_me"]
    third_person_record = payload_records_by_scene["pic_teach_person"]
    scene_record = payload_records_by_scene["pic_teach_scene_galbot"]
    runner = _actual_runner(
        case=POST_TEACH_SCENE_REPLAY_CASE,
        out=out,
        source_frame=_load_source_frame_from_record(
            self_record,
            scene=scenes_by_name["pic_teach_me"],
        ),
        camera=camera,
    )
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
    timestamp_ms = memory_e2e.QUERY_INTERVAL_MS
    timestamp_step_ms = memory_e2e.QUERY_INTERVAL_MS * 2

    with runner.open_stream() as websocket:
        runner.processor.mode = "single"
        runner.source_frame = _load_source_frame_from_record(
            self_record,
            scene=scenes_by_name["pic_teach_me"],
        )
        runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=timestamp_ms,
            states_file=states_file,
            phase="post-teach-self-seed",
        )
        last_query_timestamp_ms = timestamp_ms
        self_teach = _post_teach_person_recording_outcome(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=f"{_payload_index(self_record)}:post-teach",
            scene=self_record["scene"],
            endpoint=self_record["endpoint"],
            payload=self_record["payload"],
            operation="post_teach_person_self",
        )
        timestamp_ms += timestamp_step_ms

        runner.processor.mode = "third_person"
        runner.source_frame = _load_source_frame_from_record(
            third_person_record,
            scene=scenes_by_name["pic_teach_person"],
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
        third_person_teach = _post_teach_person_recording_outcome(
            runner=runner,
            api_response_records=api_response_records,
            payload_index=f"{_payload_index(third_person_record)}:post-teach",
            scene=third_person_record["scene"],
            endpoint=third_person_record["endpoint"],
            payload=third_person_record["payload"],
            operation="post_teach_person_third_person",
        )
        timestamp_ms += timestamp_step_ms

        runner.processor.mode = "single"
        runner.source_frame = _load_source_frame_from_record(
            scene_record,
            scene=scenes_by_name["pic_teach_scene_galbot"],
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
            replay_record = payload_records_by_scene.get(scene.name)
            runner.source_frame = (
                _load_source_frame_from_record(replay_record, scene=scene)
                if replay_record is not None
                else _load_source_frame_from_scene(scene)
            )
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
                    "source_image_path": str(runner.source_frame.path),
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
        **_prefixed_teach_person_report_fields("self", self_teach),
        **_prefixed_teach_person_report_fields("third_person", third_person_teach),
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
    processor = getattr(runner, "processor", None)
    if processor is not None:
        processor.drop_next_memory_snapshot = True
    state = runner.send(
        websocket,
        timestamp_ms=start_timestamp_ms,
        states_file=states_file,
        phase=f"{phase}:stream-ref",
    )
    stream_ref = state.get("stream_ref")
    if memory_service is not None and isinstance(stream_ref, str) and stream_ref:
        # Seed frames establish the request interaction window only; avoid
        # kicking off recognition queries before the teach/resolve request.
        memory_service._last_query_frame_timestamp_ms_by_stream[  # noqa: SLF001
            (stream_ref, runner.camera)
        ] = start_timestamp_ms
        _discard_seed_query_for_stream(
            memory_service,
            stream_ref=stream_ref,
            camera=runner.camera,
        )
    for index in range(2):
        state = runner.send(
            websocket,
            timestamp_ms=start_timestamp_ms + index + 1,
            states_file=states_file,
            phase=f"{phase}:stable-{index + 1}",
        )
    return state


def _discard_seed_query_for_stream(
    memory_service: Any,
    *,
    stream_ref: str,
    camera: str,
) -> None:
    stream_key = (stream_ref, camera)
    pending = memory_service._pending_queries_by_stream.get(stream_key)  # noqa: SLF001
    if pending is not None:
        for _ in range(50):
            if pending.done():
                break
            time.sleep(0.01)
        if pending.done():
            memory_service._collect_completed_query(stream_key, pending)  # noqa: SLF001
    queue = memory_service._completed_by_stream.get(stream_key)  # noqa: SLF001
    if queue is not None:
        queue.clear()
    _clear_seed_anonymous_memory(memory_service)


def _clear_seed_anonymous_memory(memory_service: Any) -> None:
    store = getattr(memory_service, "store", None)
    lock = getattr(store, "_lock", None)
    connection = getattr(store, "connection", None)
    if lock is None or connection is None:
        return
    with lock:
        with connection:
            connection.execute(
                "DELETE FROM embedding_provenance WHERE owner_type = 'anonymous'"
            )
            connection.execute("DELETE FROM anonymous_embedding_vectors")
            connection.execute("DELETE FROM anonymous_embeddings")
            connection.execute("DELETE FROM anonymous_profiles")


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
        source_frame=_load_source_frame_from_record(record, scene=scene),
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
        teach = _post_teach_person_recording_outcome(
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
        **_teach_person_report_fields(teach),
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
        source_frame=_load_source_frame_from_record(record, scene=scene),
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
        teach = _post_teach_person_recording_outcome(
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
        bounded_multi_person_recognition = (
            _latest_bounded_recognition_report_from_runner(
                runner,
                camera=camera,
            )
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
        **_teach_person_report_fields(teach),
        "resolver_target_ref": resolver_target_ref,
        "introducer_ref": introducer_ref,
        "pose_pointing_scoring": pose_pointing_scoring,
        "stored_person_id": stored_person_id,
        "stored_embedding_source_track_ref": stored_embedding_source_track_ref,
        "stored_crop_hash": stored_crop_hash,
        "stored_crop_path_or_artifact_ref": stored_crop_path_or_artifact_ref,
        "bounded_multi_person_recognition": bounded_multi_person_recognition,
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
        source_frame=_load_source_frame_from_record(record, scene=scene),
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
        source_frame=_load_source_frame_from_record(record, scene=scene),
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


def _run_actual_supporting_contracts(
    *,
    scenes: list[SceneDir],
    out: Path,
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    source_scene = _find_scene(scenes, "pic_teach_me") or (
        scenes[0] if scenes else None
    )
    if source_scene is None:
        return {
            "passed": False,
            "reason": "no_source_scene",
            "assertions": {"source_scene_present": False},
        }
    source_frame = _load_source_frame_from_scene(source_scene)
    summary_link = _run_actual_supporting_summary_link(
        out=out,
        source_frame=source_frame,
        camera=camera,
        states_file=states_file,
        api_response_records=api_response_records,
        botified_frame_records=botified_frame_records,
    )
    familiar_merge = _run_actual_supporting_familiar_merge(
        out=out,
        source_frame=source_frame,
        camera=camera,
        states_file=states_file,
        api_response_records=api_response_records,
        botified_frame_records=botified_frame_records,
    )
    correct_identity = _run_actual_supporting_correct_identity(
        out=out,
        source_frame=source_frame,
        camera=camera,
        states_file=states_file,
        api_response_records=api_response_records,
        botified_frame_records=botified_frame_records,
    )
    event_identity_context = _run_actual_supporting_event_identity_context(
        out=out,
        source_frame=source_frame,
        camera=camera,
        states_file=states_file,
        api_response_records=api_response_records,
        botified_frame_records=botified_frame_records,
    )
    resolve_states = _run_actual_supporting_resolve_target_states(
        out=out,
        source_frame=source_frame,
        camera=camera,
        states_file=states_file,
        api_response_records=api_response_records,
    )
    assertions = {
        "conversation_summary_context": bool(
            summary_link["assertions"].get("summary_context_present")
        ),
        "external_user_link_lookup": bool(
            summary_link["assertions"].get("external_link_lookup")
        ),
        "external_lookup_summary_present": bool(
            summary_link["assertions"].get("external_lookup_summary_present")
        ),
        "familiar_unknown_present": bool(
            familiar_merge["assertions"].get("familiar_unknown_present")
        ),
        "teach_auto_merge_suppressed_anonymous": bool(
            familiar_merge["assertions"].get("old_anonymous_suppressed")
        ),
        "teach_auto_merge_known_replay": bool(
            familiar_merge["assertions"].get("known_replay_present")
        ),
        "correct_identity_suppressed_wrong_person": bool(
            correct_identity["assertions"].get("wrong_person_not_returned")
        ),
        "event_identity_context_present": bool(
            event_identity_context["assertions"].get("event_identity_context_present")
        ),
        "event_identity_context_person_waving": bool(
            event_identity_context["assertions"].get("event_is_person_waving")
        ),
        "event_identity_context_known_person": bool(
            event_identity_context["assertions"].get("identity_status_known_person")
        ),
        "resolve_target_resolved": resolve_states["resolved"].get("status") == "resolved",
        "resolve_target_ambiguous": resolve_states["ambiguous"].get("status") == "ambiguous",
        "resolve_target_ambiguous_no_write": bool(
            resolve_states["ambiguous"].get("no_memory_write")
        ),
        "resolve_target_not_found": (
            resolve_states["not_found"].get("status") == "not_found"
        ),
        "resolve_target_not_found_no_write": bool(
            resolve_states["not_found"].get("no_memory_write")
        ),
    }
    passed = all(assertions.values())
    return {
        "passed": passed,
        "reason": "" if passed else _first_failed_assertion(assertions),
        "runner_cases": {
            "conversation_summary_context": summary_link["runner_case"],
            "familiar_unknown_auto_merge": familiar_merge["runner_case"],
            "correct_identity": correct_identity["runner_case"],
            "event_identity_context": event_identity_context["runner_case"],
            "resolve_target_states": resolve_states["runner_case"],
        },
        "assertions": assertions,
        "conversation_summary_context": summary_link["conversation_summary_context"],
        "external_user_link": summary_link["external_user_link"],
        "familiar_unknown": familiar_merge["familiar_unknown"],
        "teach_auto_merge_anonymous": familiar_merge["teach_auto_merge_anonymous"],
        "correct_identity": correct_identity["correct_identity"],
        "event_identity_context": event_identity_context["event_identity_context"],
        "resolve_target_states": {
            "resolved": resolve_states["resolved"],
            "ambiguous": resolve_states["ambiguous"],
            "not_found": resolve_states["not_found"],
        },
    }


def _run_actual_supporting_summary_link(
    *,
    out: Path,
    source_frame: memory_e2e.SourceFrame,
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-supporting-summary-link",
        out=out,
        source_frame=source_frame,
        camera=camera,
    )
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
    external_user_ref = "memory-ga-supporting:user"
    with runner.open_stream() as websocket:
        _seed_stable_interaction_window(
            runner,
            websocket,
            start_timestamp_ms=1_000,
            states_file=states_file,
            phase="supporting-summary-link-seed",
        )
        teach = _post_teach_person_recording_outcome(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:summary-link:teach-person",
            scene="supporting_contracts",
            endpoint="/v1/memory/teach/person",
            payload=memory_e2e.self_introduction_payload(
                camera=camera,
                stream_ref=runner.require_stream_ref(),
                display_name="Supporting Summary Person",
                description="supporting contracts summary/link fixture",
                tags=["memory-ga"],
            ),
            operation="supporting_teach_person_summary_link",
        )
        person_id = teach["body"].get("person_id")
        summary = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:summary-link:conversation-summary",
            scene="supporting_contracts",
            endpoint=f"/v1/memory/person/{person_id}/conversation-summary",
            payload={
                "summary": (
                    "Remembered as a supporting contracts GA person with a compact "
                    "background summary."
                ),
                "source": "agent",
                "source_conversation_id": "memory-ga-supporting",
            },
            operation="supporting_add_conversation_summary",
        )
        link = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:summary-link:external-link",
            scene="supporting_contracts",
            endpoint="/v1/memory/link-external-user",
            payload={
                "person_id": person_id,
                "external_user_ref": external_user_ref,
            },
            operation="supporting_link_external_user",
        )
        lookup = _get_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:summary-link:get-by-external",
            scene="supporting_contracts",
            endpoint=f"/v1/memory/person/by-external-user/{external_user_ref}",
            operation="supporting_get_person_by_external_user",
        )
        identify_current = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:summary-link:identify-current",
            scene="supporting_contracts",
            endpoint="/v1/memory/identify-current",
            payload={
                "camera": camera,
                "stream_ref": runner.require_stream_ref(),
                "target": {
                    "kind": "person",
                    "intent": "identify_current",
                    "referent_text": "当前这个人",
                },
                "scope": "active_target",
                "timeout_ms": 500,
            },
            operation="supporting_identify_current",
        )
        events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=3_000,
            states_file=states_file,
            phase="supporting-summary-link-replay",
        )
    known = _known_person_event_for_person(events, person_id)
    event_conversation_summaries = (
        known.get("memory_context", {}).get("conversation_summaries")
        if known is not None
        else None
    )
    if not isinstance(event_conversation_summaries, list):
        event_conversation_summaries = []
    lookup_conversation_summaries = lookup["body"].get("conversation_summaries")
    if not isinstance(lookup_conversation_summaries, list):
        lookup_conversation_summaries = []
    assertions = {
        "teach_person_ok": teach["body"].get("ok") is True,
        "summary_ok": summary["body"].get("ok") is True,
        "link_ok": link["body"].get("ok") is True,
        "external_link_lookup": (
            lookup["body"].get("person", {}).get("person_id") == person_id
        ),
        "external_lookup_summary_present": bool(lookup_conversation_summaries),
        "known_person_present": known is not None,
        "summary_context_present": bool(event_conversation_summaries),
    }
    return {
        "runner_case": runner.case,
        "assertions": assertions,
        "conversation_summary_context": {
            "person_id": person_id,
            "summary_id": summary["body"].get("summary_id"),
            "event_conversation_summaries": event_conversation_summaries,
            "lookup_conversation_summaries": lookup_conversation_summaries,
            "event": memory_e2e.compact_events([known])[0] if known is not None else None,
        },
        "external_user_link": {
            "external_user_ref": external_user_ref,
            "link": link["body"],
            "lookup": lookup["body"],
            "lookup_conversation_summaries": lookup_conversation_summaries,
        },
        "identify_current": identify_current["body"],
    }


def _run_actual_supporting_familiar_merge(
    *,
    out: Path,
    source_frame: memory_e2e.SourceFrame,
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-supporting-familiar-merge",
        out=out,
        source_frame=source_frame,
        camera=camera,
    )
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
    with runner.open_stream() as websocket:
        first_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=1_000,
            states_file=states_file,
            phase="supporting-familiar-first",
        )
        familiar_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=3_000,
            states_file=states_file,
            phase="supporting-familiar-repeat",
        )
        familiar = memory_e2e.first_event(
            familiar_events,
            "familiar_unknown_present",
        )
        anonymous_id = _anonymous_id_from_event(familiar)
        teach = _post_teach_person_recording_outcome(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:familiar-merge:teach-auto-merge",
            scene="supporting_contracts",
            endpoint="/v1/memory/teach/person",
            payload=memory_e2e.self_introduction_payload(
                camera=camera,
                stream_ref=runner.require_stream_ref(),
                display_name="Supporting Merged Person",
                description="created from familiar unknown supporting contract",
            ),
            operation="supporting_teach_auto_merge_anonymous",
        )
        merged_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=5_000,
            states_file=states_file,
            phase="supporting-familiar-merged-replay",
        )
    person_id = teach["body"].get("person_id")
    copied_embedding_count = teach["body"].get("copied_embedding_count")
    known = _known_person_event_for_person(merged_events, person_id)
    old_anonymous_events = [
        event
        for event in merged_events
        if event.get("event") == "familiar_unknown_present"
        and _anonymous_id_from_event(event) == anonymous_id
    ]
    assertions = {
        "first_unknown_has_no_event": first_events == [],
        "familiar_unknown_present": familiar is not None,
        "anonymous_id_present": bool(anonymous_id),
        "teach_ok": teach["body"].get("ok") is True,
        "teach_auto_merge_outcome": teach["body"].get("outcome") == "merged_anonymous_person",
        "merged_anonymous_id_matches": teach["body"].get("merged_anonymous_id") == anonymous_id,
        "person_id_present": bool(person_id),
        "copied_embedding_count": (
            isinstance(copied_embedding_count, int) and copied_embedding_count > 0
        ),
        "old_anonymous_suppressed": not old_anonymous_events,
        "known_replay_present": known is not None,
    }
    return {
        "runner_case": runner.case,
        "assertions": assertions,
        "familiar_unknown": {
            "present": familiar is not None,
            "anonymous_id": anonymous_id,
            "events": memory_e2e.compact_events(familiar_events),
        },
        "teach_auto_merge_anonymous": {
            "anonymous_id": anonymous_id,
            "person_id": person_id,
            "merged_anonymous_id": teach["body"].get("merged_anonymous_id"),
            "copied_embedding_count": copied_embedding_count,
            "teach": teach["body"],
            "old_anonymous_suppressed": not old_anonymous_events,
            "known_replay_present": known is not None,
            "events": memory_e2e.compact_events(merged_events),
        },
    }


def _run_actual_supporting_correct_identity(
    *,
    out: Path,
    source_frame: memory_e2e.SourceFrame,
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-supporting-correct-identity",
        out=out,
        source_frame=source_frame,
        camera=camera,
    )
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
    with runner.open_stream() as websocket:
        _seed_stable_interaction_window(
            runner,
            websocket,
            start_timestamp_ms=1_000,
            states_file=states_file,
            phase="supporting-correct-seed",
        )
        wrong = _post_teach_person_recording_outcome(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:correct-identity:teach-wrong-person",
            scene="supporting_contracts",
            endpoint="/v1/memory/teach/person",
            payload=memory_e2e.self_introduction_payload(
                camera=camera,
                stream_ref=runner.require_stream_ref(),
                display_name="Supporting Wrong Person",
            ),
            operation="supporting_teach_wrong_person",
        )
        wrong_person_id = wrong["body"].get("person_id")
        before_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=3_000,
            states_file=states_file,
            phase="supporting-correct-before",
        )
        before_known = _known_person_event_for_person(before_events, wrong_person_id)
        memory_match_id = (
            before_known.get("evidence", {}).get("memory_match_id")
            if before_known is not None
            else None
        )
        correction = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:correct-identity:correct",
            scene="supporting_contracts",
            endpoint="/v1/memory/correct-identity",
            payload={
                "memory_match_id": memory_match_id,
                "wrong_person_id": wrong_person_id,
            },
            operation="supporting_correct_identity",
        )
        after_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=5_000,
            states_file=states_file,
            phase="supporting-correct-after",
        )
    wrong_person_after = _known_person_event_for_person(after_events, wrong_person_id)
    assertions = {
        "wrong_person_known_before": before_known is not None,
        "memory_match_id_present": bool(memory_match_id),
        "correction_ok": correction["body"].get("ok") is True,
        "wrong_person_not_returned": wrong_person_after is None,
    }
    return {
        "runner_case": runner.case,
        "assertions": assertions,
        "correct_identity": {
            "wrong_person_id": wrong_person_id,
            "memory_match_id": memory_match_id,
            "correction": correction["body"],
            "wrong_person_not_returned": wrong_person_after is None,
            "before_events": memory_e2e.compact_events(before_events),
            "after_events": memory_e2e.compact_events(after_events),
        },
    }


def _run_actual_supporting_event_identity_context(
    *,
    out: Path,
    source_frame: memory_e2e.SourceFrame,
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
    botified_frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-supporting-event-identity-context",
        out=out,
        source_frame=source_frame,
        camera=camera,
    )
    _attach_cli_botified_projection_recorder(runner, botified_frame_records)
    with runner.open_stream() as websocket:
        _seed_stable_interaction_window(
            runner,
            websocket,
            start_timestamp_ms=1_000,
            states_file=states_file,
            phase="supporting-event-identity-seed",
        )
        teach = _post_teach_person_recording_outcome(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:event-identity:teach-person",
            scene="supporting_contracts",
            endpoint="/v1/memory/teach/person",
            payload=memory_e2e.self_introduction_payload(
                camera=camera,
                stream_ref=runner.require_stream_ref(),
                display_name="Supporting Event Person",
                description="supporting contracts event identity fixture",
                tags=["memory-ga"],
            ),
            operation="supporting_teach_person_event_identity_context",
        )
        runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=3_000,
            states_file=states_file,
            phase="supporting-event-identity-known-replay",
        )
        runner.processor.emit_next_ordinary_person_event = True
        state = runner.send(
            websocket,
            timestamp_ms=5_000,
            states_file=states_file,
            phase="supporting-event-identity-ordinary-event",
        )

    event = _first_event_with_identity_context(state)
    identity_context = (
        event.get("identity_context") if isinstance(event, dict) else None
    )
    assertions = {
        "teach_person_ok": teach["body"].get("ok") is True,
        "person_id_present": bool(teach["body"].get("person_id")),
        "event_identity_context_present": isinstance(identity_context, dict),
        "event_is_person_waving": (
            isinstance(event, dict) and event.get("event") == "person_waving"
        ),
        "identity_status_known_person": (
            isinstance(identity_context, dict)
            and identity_context.get("status") == "known_person"
        ),
    }
    return {
        "runner_case": runner.case,
        "assertions": assertions,
        "event_identity_context": {
            "person_id": teach["body"].get("person_id"),
            "event": memory_e2e.compact_events([event])[0]
            if event is not None
            else None,
            "identity_context": identity_context,
        },
    }


def _run_actual_supporting_resolve_target_states(
    *,
    out: Path,
    source_frame: memory_e2e.SourceFrame,
    camera: str,
    states_file: Any,
    api_response_records: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = _actual_runner(
        case="ga-supporting-resolve-target-states",
        out=out,
        source_frame=source_frame,
        camera=camera,
    )
    with runner.open_stream() as websocket:
        memory_service = runner.client.app.state.memory_service
        runner.processor.drop_next_memory_snapshot = True
        state = runner.send(
            websocket,
            timestamp_ms=1_000,
            states_file=states_file,
            phase="supporting-resolve-single-frame",
        )
        stream_ref = state.get("stream_ref")
        if isinstance(stream_ref, str) and stream_ref:
            memory_service._last_query_frame_timestamp_ms_by_stream[  # noqa: SLF001
                (stream_ref, camera)
            ] = 1_000
        resolved_response = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:resolve-target:resolved",
            scene="supporting_contracts",
            endpoint="/v1/memory/resolve-target",
            payload={
                "camera": camera,
                "target": {
                    "kind": "scene",
                    "intent": "teach_scene",
                    "referent_text": "这里",
                },
            },
            operation="supporting_resolve_target_resolved",
        )
        ambiguous_response = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:resolve-target:ambiguous",
            scene="supporting_contracts",
            endpoint="/v1/memory/resolve-target",
            payload={
                "camera": camera,
                "target": {
                    "kind": "person",
                    "intent": "self_introduction",
                    "referent_text": "我",
                },
            },
            operation="supporting_resolve_target_ambiguous",
        )
        not_found_response = _post_and_record_api_response(
            runner=runner,
            api_response_records=api_response_records,
            payload_index="supporting:resolve-target:not-found",
            scene="supporting_contracts",
            endpoint="/v1/memory/resolve-target",
            payload=memory_e2e.object_resolve_payload(
                camera=camera,
                stream_ref=runner.require_stream_ref(),
                referent_text="手机",
            ),
            operation="supporting_resolve_target_not_found",
        )
    resolved = resolved_response["body"]
    ambiguous = ambiguous_response["body"]
    not_found = not_found_response["body"]
    return {
        "runner_case": runner.case,
        "resolved": {
            "status_code": resolved_response["status_code"],
            "status": resolved.get("status"),
            "response": resolved,
        },
        "ambiguous": {
            "status_code": ambiguous_response["status_code"],
            "status": ambiguous.get("status"),
            "ambiguity_type": ambiguous.get("ambiguity_type"),
            "no_memory_write": _response_store_delta_no_write(ambiguous),
            "store_delta": ambiguous.get("store_delta"),
            "response": ambiguous,
        },
        "not_found": {
            "status_code": not_found_response["status_code"],
            "status": not_found.get("status"),
            "error_code": not_found.get("error_code"),
            "no_memory_write": _response_store_delta_no_write(not_found),
            "store_delta": not_found.get("store_delta"),
            "response": not_found,
        },
    }


def _anonymous_id_from_event(event: dict[str, Any] | None) -> Any:
    if not event:
        return None
    memory_context = event.get("memory_context", {})
    anonymous = memory_context.get("anonymous_person")
    if isinstance(anonymous, dict):
        return anonymous.get("anonymous_id")
    anonymous = memory_context.get("anonymous")
    if isinstance(anonymous, dict):
        return anonymous.get("anonymous_id")
    return None


def _first_event_with_identity_context(
    visual_state: dict[str, Any],
) -> dict[str, Any] | None:
    semantic_events = visual_state.get("semantic_events")
    if not isinstance(semantic_events, list):
        return None
    for event in semantic_events:
        if isinstance(event, dict) and isinstance(event.get("identity_context"), dict):
            return event
    return None


def _response_store_delta_no_write(body: dict[str, Any]) -> bool:
    store_delta = body.get("store_delta")
    if not isinstance(store_delta, dict):
        return False
    delta = store_delta.get("delta")
    if not isinstance(delta, dict):
        return False
    return all(value == 0 for value in delta.values())


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
    return _load_source_frame(scene.jpeg_paths[0])


def _load_source_frame_from_record(
    record: dict[str, Any],
    *,
    scene: SceneDir,
) -> memory_e2e.SourceFrame:
    source_image_path = record.get("source_image_path")
    if isinstance(source_image_path, str) and source_image_path:
        return _load_source_frame(Path(source_image_path))
    return _load_source_frame_from_scene(scene)


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
    payload = _payload_for_runner_stream(
        runner=runner,
        endpoint=endpoint,
        payload=payload,
    )
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


def _payload_for_runner_stream(
    *,
    runner: Any,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    stream_ref = getattr(runner, "latest_stream_ref", None)
    return memory_e2e._with_stream_ref_for_endpoint(  # noqa: SLF001
        endpoint=endpoint,
        payload=payload,
        stream_ref=stream_ref if isinstance(stream_ref, str) else None,
    )


def _get_and_record_api_response(
    *,
    runner: Any,
    api_response_records: list[dict[str, Any]],
    payload_index: int | str,
    scene: str,
    endpoint: str,
    operation: str,
) -> dict[str, Any]:
    response = runner.client.get(endpoint)
    body = response.json()
    record = {
        "payload_index": payload_index,
        "scene": scene,
        "endpoint": endpoint,
        "operation": operation,
        "method": "GET",
        "dry_run": False,
        "status_code": response.status_code,
        "payload": None,
        "response": body,
    }
    api_response_records.append(record)
    return {"status_code": response.status_code, "body": body}


def _post_teach_person_recording_outcome(
    *,
    runner: Any,
    api_response_records: list[dict[str, Any]],
    payload_index: int | str,
    scene: str,
    endpoint: str,
    payload: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    teach = _post_and_record_api_response(
        runner=runner,
        api_response_records=api_response_records,
        payload_index=payload_index,
        scene=scene,
        endpoint=endpoint,
        payload=payload,
        operation=operation,
    )
    body = dict(teach["body"]) if isinstance(teach["body"], dict) else {}
    if body.get("ok") is True and body.get("outcome"):
        body.setdefault(
            "teach_person_outcome",
            body["outcome"],
        )
    return {**teach, "body": body}


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
    return SceneDir(
        name=path.name,
        path=path,
        jpeg_paths=jpeg_paths,
        interactions=tuple(_interaction_cases_from_scene(path)),
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


def _interaction_cases_from_scene(path: Path) -> list[InteractionCase]:
    cases: list[InteractionCase] = []
    for source_text_path in _sorted_transcript_paths(path):
        source_image_path = _same_stem_image_path(source_text_path)
        if source_image_path is None:
            continue
        cases.append(
            InteractionCase(
                scene=path.name,
                source_text_path=source_text_path,
                source_image_path=source_image_path,
                transcript_text=source_text_path.read_text(encoding="utf-8").strip(),
            )
        )
    return cases


def _sorted_transcript_paths(path: Path) -> list[Path]:
    return sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() == TRANSCRIPT_SUFFIX
        ],
        key=lambda item: item.name,
    )


def _same_stem_image_path(transcript_path: Path) -> Path | None:
    for suffix in JPEG_SUFFIX_ORDER:
        candidate = transcript_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


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
        if scene is not None and scene.interactions:
            records.append(_teach_payload_record(scene.interactions[0], camera=camera))
    return records


def _teach_payload_record(
    interaction: InteractionCase,
    *,
    camera: str,
) -> dict[str, Any]:
    transcript_text = interaction.transcript_text
    if interaction.scene == "pic_teach_me":
        display_name = _extract_self_display_name(transcript_text)
        endpoint = "/v1/memory/teach/person"
        payload = {
            "camera": camera,
            "stream_ref": PAYLOAD_FIXTURE_STREAM_REF,
            "target": {
                "kind": "person",
                "intent": "self_introduction",
                "referent_text": "我",
            },
            "profile": {"display_name": display_name},
        }
        expected = {"writes_memory": True, "memory_type": "person"}
    elif interaction.scene == "pic_teach_person":
        display_name = _extract_third_person_display_name(transcript_text)
        endpoint = "/v1/memory/teach/person"
        payload = {
            "camera": camera,
            "stream_ref": PAYLOAD_FIXTURE_STREAM_REF,
            "target": {
                "kind": "person",
                "intent": "third_person_introduction",
                "referent_text": f"这位/{display_name}",
            },
            "profile": {"display_name": display_name},
        }
        expected = {"writes_memory": True, "memory_type": "person"}
    elif interaction.scene == "pic_teach_scene_galbot":
        endpoint = "/v1/memory/teach/scene"
        payload = {
            "camera": camera,
            "stream_ref": PAYLOAD_FIXTURE_STREAM_REF,
            "target": {
                "kind": "scene",
                "intent": "teach_scene",
                "referent_text": _extract_scene_referent_text(transcript_text),
            },
            "memory": {"title": _extract_scene_title(transcript_text)},
        }
        expected = {"writes_memory": True, "memory_type": "scene"}
    elif interaction.scene == "pic_teach_item_phone":
        endpoint = "/v1/memory/resolve-target"
        payload = {
            "camera": camera,
            "stream_ref": PAYLOAD_FIXTURE_STREAM_REF,
            "target": {
                "kind": "object",
                "intent": "teach_object",
                "referent_text": _extract_object_referent(transcript_text),
            },
        }
        expected = {
            "negative_only": True,
            "status": "not_found",
            "error_code": "unsupported_target_kind",
            "writes_memory": False,
        }
    else:
        raise ValueError(f"unsupported teach scene: {interaction.scene}")

    return {
        "scene": interaction.scene,
        "source_text_path": str(interaction.source_text_path),
        "source_image_path": str(interaction.source_image_path),
        "transcript_text": transcript_text,
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
                "transcript_count": len(scene.interactions),
                "transcript_paths": [
                    str(interaction.source_text_path)
                    for interaction in scene.interactions
                ],
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
            "name": "expected_teach_transcript_payloads",
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
    supporting_contracts_result: dict[str, Any],
    botified_frame_records: list[dict[str, Any]],
    bounded_multi_person_recognition: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_teach_scenes = set(TEACH_SCENE_ORDER)
    actual_teach_scenes = {record["scene"] for record in payload_records}
    artifact_exists = {
        key: (out / relative_path).exists()
        for key, relative_path in artifact_paths.items()
        if key != "report_json" and key not in EVIDENCE_ONLY_ARTIFACT_KEYS
    }
    evidence_exists = {
        item["path"]: (out / item["path"]).is_file() for item in visual_evidence_index
    }
    gate_api_response_records = _gate_api_response_records(api_response_records)
    actual_response_assertions = {
        "has_api_responses": bool(gate_api_response_records),
        "all_actual": all(
            record.get("dry_run") is False for record in gate_api_response_records
        ),
        "no_stubbed_status": all(
            record.get("response", {}).get("status") != "stubbed"
            for record in gate_api_response_records
        ),
        "status_codes_ok": all(
            _api_response_status_ok(record)
            for record in gate_api_response_records
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
            "name": "expected_teach_transcript_payloads",
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
                "gate_response_count": len(gate_api_response_records),
                "evidence_only_response_count": (
                    len(api_response_records) - len(gate_api_response_records)
                ),
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
        _bounded_multi_person_recognition_check(
            bounded_multi_person_recognition,
            require_non_attention_query=True,
        ),
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
            "name": "supporting_contracts",
            "passed": bool(supporting_contracts_result.get("passed")),
            "details": supporting_contracts_result,
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


def _api_response_status_ok(record: dict[str, Any]) -> bool:
    status_code = int(record.get("status_code") or 0)
    return status_code < 400


def _gate_api_response_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("operation") not in EVIDENCE_ONLY_API_OPERATIONS
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
    bounded_multi_person_recognition: dict[str, Any],
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
        if key != "report_json" and key not in EVIDENCE_ONLY_ARTIFACT_KEYS
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
        _bounded_multi_person_recognition_check(
            bounded_multi_person_recognition,
            require_non_attention_query=False,
        ),
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
        "transcript_count": len(scene.interactions),
        "transcript_paths": [
            str(interaction.source_text_path) for interaction in scene.interactions
        ],
    }


def _teach_request_summary(record: dict[str, Any]) -> dict[str, Any]:
    payload = record["payload"]
    return {
        "scene": record["scene"],
        **_record_source_fields(record),
        "endpoint": record["endpoint"],
        "camera": payload["camera"],
        "target": payload["target"],
        "profile": payload.get("profile"),
        "memory": payload.get("memory"),
        "expected": record["expected"],
    }


def _with_record_source_fields(
    result: dict[str, Any],
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        return result
    fields = _record_source_fields(record)
    if not fields:
        return result
    return {**fields, **result}


def _record_source_fields(record: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("source_text_path", "source_image_path", "transcript_text"):
        value = record.get(key)
        if isinstance(value, str) and value:
            fields[key] = value
    return fields


def _write_current_visual_snapshot_artifact(
    visual_states_path: Path,
    current_snapshot_path: Path,
) -> None:
    visual_state = _best_identity_visual_state_from_sidecar(visual_states_path)
    now_ms = None
    if isinstance(visual_state, dict):
        frame_timestamp_ms = visual_state.get("frame_timestamp_ms")
        if isinstance(frame_timestamp_ms, (int, float)):
            now_ms = int(frame_timestamp_ms)
    _write_json(
        current_snapshot_path,
        build_current_visual_snapshot(visual_state, now_ms=now_ms),
    )


def _best_identity_visual_state_from_sidecar(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest: dict[str, Any] | None = None
    best_identity_state: dict[str, Any] | None = None
    best_identity_priority = -1
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = _visual_state_from_sidecar_record(record)
            if state is not None:
                latest = state
                if isinstance(state.get("identity_context"), dict):
                    priority = memory_teaching_evidence.visual_state_identity_priority(
                        state
                    )
                    if (
                        best_identity_state is None
                        or priority > best_identity_priority
                    ):
                        best_identity_state = state
                        best_identity_priority = priority
    return best_identity_state if best_identity_state is not None else latest


def _visual_state_from_sidecar_record(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    for key in ("visual_state", "state", "response"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    if any(key in record for key in ("semantic_events", "tracks", "identity_context")):
        return record
    return None


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
