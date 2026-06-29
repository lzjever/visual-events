from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from visual_events_server.attention import AttentionResult
from visual_events_server.app import create_app
from visual_events_server.config import (
    MemoryConfig,
    MemoryEmbeddingConfig,
    MemoryMatchingConfig,
    ServerConfig,
)
from visual_events_server.inference.base import PoseKeypoint
from visual_events_server.memory.frame_cache import MemoryFrameSnapshot
from visual_events_server.processor import VisualFrameProcessor
from visual_events_server.protocol import (
    SCHEMA_VERSION,
    FrameMessage,
    encode_frame_message,
)
from visual_events_server.protocol import _parse_jpeg_dimensions
from visual_events_server.tracking import TrackSnapshot


DEFAULT_DATA_DIR = Path("val-data")
DEFAULT_OUT = Path("artifacts/memory-e2e")
DEFAULT_SCENE = "pic_hello"
DEFAULT_CAMERA = "front"
QUERY_INTERVAL_MS = 1000
FRAME_CACHE_SECONDS = 10
QUERY_DRAIN_WAIT_SECONDS = 0.25
PRIMARY_TRACK_ID = 7
AMBIGUOUS_TRACK_ID = 8
PRIMARY_PERSON_BBOX_XYXY = (580.0, 60.0, 805.0, 650.0)
AMBIGUOUS_PERSON_BBOX_XYXY = (650.0, 70.0, 875.0, 650.0)
THIRD_PERSON_BBOX_XYXY = (880.0, 170.0, 1080.0, 420.0)
PRIMARY_ATTENTION_UV = (692.5, 160.0)
AMBIGUOUS_POINT_UV = PRIMARY_ATTENTION_UV


@dataclass(frozen=True)
class SourceFrame:
    path: Path
    jpeg_bytes: bytes
    width: int
    height: int


class MemoryScenarioProcessor(VisualFrameProcessor):
    def __init__(self) -> None:
        self.mode = "single"
        self._last_snapshot: MemoryFrameSnapshot | None = None

    async def process_frame(self, frame: FrameMessage) -> dict[str, Any]:
        tracks = self._tracks(frame)
        self._last_snapshot = self._memory_snapshot(frame)
        return {
            "type": "visual_state",
            "schema_version": SCHEMA_VERSION,
            "camera": frame.camera,
            "frame_id": frame.frame_id,
            "frame_timestamp_ms": frame.timestamp_ms,
            "server_timestamp_ms": int(time.time() * 1000),
            "image_size": [frame.width, frame.height],
            "tracks": tracks,
            "attention": {
                "target_track_id": PRIMARY_TRACK_ID,
                "target_uv": list(PRIMARY_ATTENTION_UV),
                "reason": "memory_e2e_stable_target",
                "confidence": 0.96,
            },
            "scene_context": {
                "engagement_state": "available",
                "attention_available": True,
                "target_track_id": PRIMARY_TRACK_ID,
                "no_engage_reasons": [],
                "target_reacquired": False,
            },
            "scene_flags": {
                "has_person": True,
                "person_count": len(tracks),
                "largest_person_stable": True,
                "someone_near_center": True,
            },
            "semantic_events": [],
        }

    def take_memory_frame_snapshot(self) -> MemoryFrameSnapshot | None:
        snapshot = self._last_snapshot
        self._last_snapshot = None
        return snapshot

    def _tracks(self, frame: FrameMessage) -> list[dict[str, Any]]:
        primary = _track(
            track_id=PRIMARY_TRACK_ID,
            bbox_xyxy=list(PRIMARY_PERSON_BBOX_XYXY),
            timestamp_ms=frame.timestamp_ms,
        )
        if self.mode == "third_person":
            return [
                primary,
                _track(
                    track_id=AMBIGUOUS_TRACK_ID,
                    bbox_xyxy=list(THIRD_PERSON_BBOX_XYXY),
                    timestamp_ms=frame.timestamp_ms,
                    confidence=0.91,
                ),
            ]
        if self.mode != "ambiguous":
            return [primary]
        return [
            primary,
            _track(
                track_id=AMBIGUOUS_TRACK_ID,
                bbox_xyxy=list(AMBIGUOUS_PERSON_BBOX_XYXY),
                timestamp_ms=frame.timestamp_ms,
                confidence=0.9,
            ),
        ]

    def _memory_snapshot(self, frame: FrameMessage) -> MemoryFrameSnapshot:
        return MemoryFrameSnapshot(
            connection_id="",
            frame=frame,
            source_frame_ref=f"{frame.camera}:{frame.frame_id}:{frame.timestamp_ms}",
            snapshot_ref=f"memory-e2e:{frame.camera}:{frame.frame_id}",
            observed_at_ms=int(time.time() * 1000),
            image_size=(frame.width, frame.height),
            tracks=self._snapshot_tracks(frame),
            attention=AttentionResult(
                target_track_id=PRIMARY_TRACK_ID,
                target_uv=PRIMARY_ATTENTION_UV,
                reason="memory_e2e_stable_target",
                confidence=0.96,
                largest_person_stable=True,
            ),
            scene_context={
                "engagement_state": "available",
                "attention_available": True,
                "target_track_id": PRIMARY_TRACK_ID,
            },
            semantic_events=[],
        )

    def _snapshot_tracks(self, frame: FrameMessage) -> list[TrackSnapshot]:
        primary_keypoints: tuple[PoseKeypoint, ...] = ()
        if self.mode == "third_person":
            primary_keypoints = (
                PoseKeypoint("left_shoulder", 670.0, 250.0, 0.9),
                PoseKeypoint("left_elbow", 780.0, 270.0, 0.9),
                PoseKeypoint("left_wrist", 860.0, 285.0, 0.9),
            )
        tracks = [
            _snapshot_track(
                track_id=PRIMARY_TRACK_ID,
                bbox_xyxy=PRIMARY_PERSON_BBOX_XYXY,
                timestamp_ms=frame.timestamp_ms,
                keypoints=primary_keypoints,
            )
        ]
        if self.mode == "third_person":
            tracks.append(
                _snapshot_track(
                    track_id=AMBIGUOUS_TRACK_ID,
                    bbox_xyxy=THIRD_PERSON_BBOX_XYXY,
                    timestamp_ms=frame.timestamp_ms,
                    confidence=0.91,
                )
            )
        elif self.mode == "ambiguous":
            tracks.append(
                _snapshot_track(
                    track_id=AMBIGUOUS_TRACK_ID,
                    bbox_xyxy=AMBIGUOUS_PERSON_BBOX_XYXY,
                    timestamp_ms=frame.timestamp_ms,
                    confidence=0.9,
                )
            )
        return tracks


class MemoryE2ERunner:
    def __init__(
        self,
        *,
        case: str,
        out: Path,
        source_frame: SourceFrame,
        camera: str,
        embedding_config: MemoryEmbeddingConfig,
    ) -> None:
        self.case = case
        self.out = out
        self.source_frame = source_frame
        self.camera = camera
        self.embedding_config = embedding_config
        self.processor = MemoryScenarioProcessor()
        self.client = TestClient(
            create_app(processor=self.processor, config=self._config())
        )
        self.frame_id = 0

    def _config(self) -> ServerConfig:
        db_path = self.out / "runtime" / f"{self.case}.sqlite3"
        return ServerConfig(
            memory=MemoryConfig(
                enabled=True,
                db_path=db_path,
                frame_cache_seconds=FRAME_CACHE_SECONDS,
                query_interval_ms=QUERY_INTERVAL_MS,
                queue_size=8,
                embedding=self.embedding_config,
                matching=MemoryMatchingConfig(
                    known_person_threshold=0.99,
                    known_person_margin=0.0,
                    anonymous_threshold=0.99,
                    anonymous_margin=0.0,
                    familiar_seen_count=2,
                    familiar_threshold=0.99,
                    scene_threshold=0.99,
                    event_cooldown_ms=QUERY_INTERVAL_MS,
                ),
            )
        )

    def open_stream(self):
        return self.client.websocket_connect("/v1/stream")

    def send(
        self,
        websocket: Any,
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
            "width": self.source_frame.width,
            "height": self.source_frame.height,
            "head_motion": {"state": "stationary"},
        }
        websocket.send_bytes(
            encode_frame_message(header, self.source_frame.jpeg_bytes)
        )
        state = json.loads(websocket.receive_text())
        states_file.write(
            json.dumps(
                {
                    "case": self.case,
                    "phase": phase,
                    "source_frame": str(self.source_frame.path),
                    "visual_state": state,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        states_file.flush()
        return state

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
        # Give the in-process memory worker a short window to finish, then send a
        # drain-only frame whose timestamp does not satisfy query_interval_ms.
        time.sleep(QUERY_DRAIN_WAIT_SECONDS)
        drained = self.send(
            websocket,
            timestamp_ms=query_timestamp_ms + 1,
            states_file=states_file,
            phase=f"{phase}:drain",
        )
        return list(drained.get("semantic_events") or [])

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(path, json=payload)
        body = response.json()
        if response.status_code >= 400:
            raise RuntimeError(f"POST {path} failed: {response.status_code} {body}")
        return body


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the manual memory E2E gate with val-data JPEG bytes."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument(
        "--embedding-backend",
        choices=("fake", "local"),
        default="fake",
        help="Embedding backend to use. Defaults to deterministic fake embeddings.",
    )
    parser.add_argument(
        "--person-model-path",
        type=Path,
        help="Local person embedding model bundle path. Required for --embedding-backend local.",
    )
    parser.add_argument(
        "--scene-model-path",
        type=Path,
        help="Local scene embedding model bundle path. Required for --embedding-backend local.",
    )
    return parser.parse_args(argv)


def build_notes(args: argparse.Namespace) -> list[str]:
    notes = ["Manual gate only; not wired into the default publish gate."]
    if args.embedding_backend == "fake":
        notes.append(
            "Uses deterministic fake memory embeddings and stable synthetic "
            "visual_state; this is not evidence of real local model behavior."
        )
    else:
        notes.append(
            "Uses the real local memory embedding backend with explicit model "
            "paths; model files are local-only artifacts and are not written to Git."
        )
    notes.append("Real val-data JPEG bytes are used as frame payloads.")
    return notes


def build_embedding_config(args: argparse.Namespace) -> MemoryEmbeddingConfig:
    if args.embedding_backend == "fake":
        return MemoryEmbeddingConfig(backend="fake")
    if args.person_model_path is None or args.scene_model_path is None:
        missing = []
        if args.person_model_path is None:
            missing.append("--person-model-path")
        if args.scene_model_path is None:
            missing.append("--scene-model-path")
        raise ValueError(
            "--embedding-backend local requires " + " and ".join(missing)
        )
    missing_paths = []
    if not args.person_model_path.exists():
        missing_paths.append(f"person_model_path={args.person_model_path}")
    if not args.scene_model_path.exists():
        missing_paths.append(f"scene_model_path={args.scene_model_path}")
    if missing_paths:
        raise FileNotFoundError(
            "local embedding model path missing: " + ", ".join(missing_paths)
        )
    return MemoryEmbeddingConfig(
        backend="local",
        person_model_path=args.person_model_path,
        scene_model_path=args.scene_model_path,
    )


def self_introduction_payload(
    *,
    camera: str,
    display_name: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {"display_name": display_name}
    if description:
        profile["description"] = description
    if tags:
        profile["tags"] = tags
    return {
        "camera": camera,
        "target": {
            "kind": "person",
            "intent": "self_introduction",
            "referent_text": "我",
        },
        "profile": profile,
    }


def third_person_introduction_payload(
    *,
    camera: str,
    display_name: str,
    referent_text: str = "这位",
) -> dict[str, Any]:
    return {
        "camera": camera,
        "target": {
            "kind": "person",
            "intent": "third_person_introduction",
            "referent_text": referent_text,
        },
        "profile": {"display_name": display_name},
    }


def teach_scene_payload(
    *,
    camera: str,
    title: str,
    description: str | None = None,
    activation_hint: str | None = None,
) -> dict[str, Any]:
    memory: dict[str, Any] = {"title": title}
    if description:
        memory["description"] = description
    if activation_hint:
        memory["activation_hint"] = activation_hint
    return {
        "camera": camera,
        "target": {
            "kind": "scene",
            "intent": "teach_scene",
            "referent_text": "这里",
        },
        "memory": memory,
    }


def object_resolve_payload(*, camera: str, referent_text: str) -> dict[str, Any]:
    return {
        "camera": camera,
        "target": {
            "kind": "object",
            "intent": "teach_object",
            "referent_text": referent_text,
        },
    }


def embedding_report(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "embedding_backend": args.embedding_backend,
    }
    if args.embedding_backend == "fake":
        report["real_model_evidence"] = False
        report["evidence_note"] = (
            "fake mode uses deterministic test embeddings and is not proof that "
            "real local models load or infer correctly"
        )
    else:
        report["uses_real_model_backend"] = True
        report["person_model_path"] = (
            str(args.person_model_path) if args.person_model_path is not None else None
        )
        report["scene_model_path"] = (
            str(args.scene_model_path) if args.scene_model_path is not None else None
        )
        report["evidence_note"] = (
            "local mode runs the configured LocalEmbeddingBackend; require "
            "ok=true before treating this report as real model evidence"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    runtime_dir = out / "runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    states_path = out / "states.jsonl"
    report_path = out / "report.json"
    checks: list[dict[str, Any]] = []
    notes = build_notes(args)

    try:
        embedding_config = build_embedding_config(args)
        checks.append(
            {
                "name": "preflight_embedding_backend",
                "passed": True,
                "details": embedding_report(args),
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "preflight_embedding_backend",
                "passed": False,
                "details": {"error": str(exc), "error_type": type(exc).__name__},
            }
        )
        report = build_report(
            ok=False,
            args=args,
            checks=checks,
            notes=notes,
            states_path=states_path,
            report_path=report_path,
            source_frame=None,
        )
        write_report(report_path, report)
        print(f"memory E2E failed embedding preflight: {exc}", file=sys.stderr)
        print(f"report: {report_path}")
        return 1

    try:
        source_frame = load_source_frame(args.data_dir, args.scene)
        shutil.copyfile(source_frame.path, out / "source_frame.jpeg")
    except Exception as exc:
        checks.append(
            {
                "name": "preflight_val_data_source_frame",
                "passed": False,
                "details": {"error": str(exc), "error_type": type(exc).__name__},
            }
        )
        report = build_report(
            ok=False,
            args=args,
            checks=checks,
            notes=notes,
            states_path=states_path,
            report_path=report_path,
            source_frame=None,
        )
        write_report(report_path, report)
        print(f"memory E2E failed preflight: {exc}", file=sys.stderr)
        print(f"report: {report_path}")
        return 1

    with states_path.open("w", encoding="utf-8") as states_file:
        run_check(
            checks,
            "public self_introduction replay known_person_present and scene_activated",
            lambda: check_teach_person_scene_summary_link(
                out=out,
                source_frame=source_frame,
                camera=args.camera,
                embedding_config=embedding_config,
                states_file=states_file,
            ),
        )
        run_check(
            checks,
            "public third_person_introduction resolves B writes B and replays known_person_present",
            lambda: check_third_person_introduction(
                out=out,
                source_frame=source_frame,
                camera=args.camera,
                embedding_config=embedding_config,
                states_file=states_file,
            ),
        )
        run_check(
            checks,
            "v0.4 unknown repeat then merge anonymous to person",
            lambda: check_unknown_repeat_merge(
                out=out,
                source_frame=source_frame,
                camera=args.camera,
                embedding_config=embedding_config,
                states_file=states_file,
            ),
        )
        run_check(
            checks,
            "v0.4 correct identity suppresses same wrong person",
            lambda: check_correct_identity(
                out=out,
                source_frame=source_frame,
                camera=args.camera,
                embedding_config=embedding_config,
                states_file=states_file,
            ),
        )
        run_check(
            checks,
            "public object resolve-target returns unsupported no-write",
            lambda: check_object_resolve_unsupported_no_write(
                out=out,
                source_frame=source_frame,
                camera=args.camera,
                embedding_config=embedding_config,
                states_file=states_file,
            ),
        )

    ok = all(check["passed"] for check in checks)
    report = build_report(
        ok=ok,
        args=args,
        checks=checks,
        notes=notes,
        states_path=states_path,
        report_path=report_path,
        source_frame=source_frame,
    )
    write_report(report_path, report)
    print(f"memory E2E {'passed' if ok else 'failed'}")
    print(f"checks: {sum(1 for check in checks if check['passed'])}/{len(checks)} passed")
    print(f"report: {report_path}")
    print(f"states: {states_path}")
    return 0 if ok else 1


def check_teach_person_scene_summary_link(
    *,
    out: Path,
    source_frame: SourceFrame,
    camera: str,
    embedding_config: MemoryEmbeddingConfig,
    states_file: Any,
) -> dict[str, Any]:
    runner = MemoryE2ERunner(
        case="teach-person-scene",
        out=out,
        source_frame=source_frame,
        camera=camera,
        embedding_config=embedding_config,
    )
    with runner.open_stream() as websocket:
        runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=1_000,
            states_file=states_file,
            phase="seed",
        )
        person = runner.post(
            "/v1/memory/teach/person",
            self_introduction_payload(
                camera=camera,
                display_name="Memory E2E Person",
                description="stable synthetic target over val-data JPEG",
                tags=["memory-e2e"],
            ),
        )
        scene = runner.post(
            "/v1/memory/teach/scene",
            teach_scene_payload(
                camera=camera,
                title="Memory E2E Scene",
                description="scene taught from the selected val-data JPEG",
                activation_hint="use remembered scene context",
            ),
        )
        summary = runner.post(
            f"/v1/memory/person/{person['person_id']}/conversation-summary",
            {
                "summary": "Asked about the memory E2E fixture and prefers concise summaries.",
                "source": "agent",
                "source_conversation_id": "memory-e2e-conv",
            },
        )
        link = runner.post(
            "/v1/memory/link-external-user",
            {
                "person_id": person["person_id"],
                "external_user_ref": "memory-e2e:user",
            },
        )
        external_response = runner.client.get(
            "/v1/memory/person/by-external-user/memory-e2e:user"
        )
        external = external_response.json()
        if external_response.status_code >= 400:
            raise RuntimeError(f"external link lookup failed: {external}")
        events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=2_000,
            states_file=states_file,
            phase="replay",
        )

    known = first_event(events, "known_person_present")
    scene_event = first_event(events, "scene_activated")
    assertions = {
        "person_teach_ok": person.get("ok") is True,
        "known_person_present": known is not None,
        "known_person_context": bool(
            known
            and known.get("memory_context", {}).get("person", {}).get("person_id")
            == person["person_id"]
        ),
        "known_person_match_evidence": bool(
            known and known.get("evidence", {}).get("memory_match_id")
        ),
        "conversation_summary_in_event_context": bool(
            known
            and known.get("memory_context", {}).get("conversation_summaries")
        ),
        "scene_teach_ok": scene.get("ok") is True,
        "scene_activated": scene_event is not None,
        "scene_context": bool(
            scene_event
            and scene_event.get("memory_context", {})
            .get("scene", {})
            .get("scene_id")
            == scene.get("scene_id")
        ),
        "external_link_lookup": bool(
            external.get("person", {}).get("person_id") == person["person_id"]
            and external.get("conversation_summaries")
        ),
    }
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "person_id": person.get("person_id"),
        "scene_id": scene.get("scene_id"),
        "summary_id": summary.get("summary_id"),
        "link": link,
        "events": compact_events(events),
        "external_lookup": external,
    }


def check_third_person_introduction(
    *,
    out: Path,
    source_frame: SourceFrame,
    camera: str,
    embedding_config: MemoryEmbeddingConfig,
    states_file: Any,
) -> dict[str, Any]:
    runner = MemoryE2ERunner(
        case="third-person-introduction",
        out=out,
        source_frame=source_frame,
        camera=camera,
        embedding_config=embedding_config,
    )
    runner.processor.mode = "third_person"
    with runner.open_stream() as websocket:
        runner.send(
            websocket,
            timestamp_ms=1_000,
            states_file=states_file,
            phase="third-person-seed",
        )
        resolve = runner.post(
            "/v1/memory/resolve-target",
            {
                "camera": camera,
                "target": {
                    "kind": "person",
                    "intent": "third_person_introduction",
                    "referent_text": "这位",
                },
            },
        )
        person = runner.post(
            "/v1/memory/teach/person",
            third_person_introduction_payload(
                camera=camera,
                display_name="Introduced Memory E2E Person",
                referent_text="这位",
            ),
        )
        events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=2_000,
            states_file=states_file,
            phase="third-person-replay",
        )

    known = first_event(events, "known_person_present")
    candidates = resolve.get("candidates") or []
    resolved_track_id = candidates[0].get("track_id") if candidates else None
    known_track_id = known.get("track_id") if known else None
    assertions = {
        "resolve_target_ok": resolve.get("ok") is True,
        "resolve_target_resolved": resolve.get("status") == "resolved",
        "resolver_selected_b": resolved_track_id == AMBIGUOUS_TRACK_ID,
        "teach_person_ok": person.get("ok") is True,
        "known_person_present": known is not None,
        "known_person_is_b": known_track_id == AMBIGUOUS_TRACK_ID,
        "known_person_context": bool(
            known
            and known.get("memory_context", {}).get("person", {}).get("person_id")
            == person.get("person_id")
        ),
    }
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "resolve_target": resolve,
        "person_id": person.get("person_id"),
        "events": compact_events(events),
    }


def check_unknown_repeat_merge(
    *,
    out: Path,
    source_frame: SourceFrame,
    camera: str,
    embedding_config: MemoryEmbeddingConfig,
    states_file: Any,
) -> dict[str, Any]:
    runner = MemoryE2ERunner(
        case="unknown-repeat-merge",
        out=out,
        source_frame=source_frame,
        camera=camera,
        embedding_config=embedding_config,
    )
    with runner.open_stream() as websocket:
        first_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=1_000,
            states_file=states_file,
            phase="unknown-first",
        )
        familiar_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=2_000,
            states_file=states_file,
            phase="unknown-repeat",
        )
        familiar = first_event(familiar_events, "familiar_unknown_present")
        if familiar is None:
            raise AssertionError(f"missing familiar_unknown_present: {familiar_events}")
        anonymous_id = familiar["memory_context"]["anonymous_person"]["anonymous_id"]
        merge = runner.post(
            "/v1/memory/merge-anonymous-person",
            {
                "anonymous_id": anonymous_id,
                "profile": {
                    "display_name": "Merged Memory E2E Person",
                    "description": "created from familiar anonymous profile",
                },
                "merge_reason": "memory_e2e_manual_merge",
            },
        )
        merged_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=3_000,
            states_file=states_file,
            phase="merged-replay",
        )

    known = first_event(merged_events, "known_person_present")
    old_anonymous = [
        event
        for event in merged_events
        if event.get("event") == "familiar_unknown_present"
        and event.get("memory_context", {})
        .get("anonymous_person", {})
        .get("anonymous_id")
        == anonymous_id
    ]
    assertions = {
        "first_unknown_has_no_event": first_events == [],
        "familiar_unknown_present": familiar is not None,
        "merge_ok": merge.get("ok") is True,
        "known_after_merge": bool(
            known
            and known.get("memory_context", {}).get("person", {}).get("person_id")
            == merge.get("person_id")
        ),
        "old_anonymous_suppressed": not old_anonymous,
    }
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "anonymous_id": anonymous_id,
        "merge": merge,
        "familiar_events": compact_events(familiar_events),
        "merged_events": compact_events(merged_events),
    }


def check_correct_identity(
    *,
    out: Path,
    source_frame: SourceFrame,
    camera: str,
    embedding_config: MemoryEmbeddingConfig,
    states_file: Any,
) -> dict[str, Any]:
    runner = MemoryE2ERunner(
        case="correct-identity",
        out=out,
        source_frame=source_frame,
        camera=camera,
        embedding_config=embedding_config,
    )
    with runner.open_stream() as websocket:
        runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=1_000,
            states_file=states_file,
            phase="seed",
        )
        wrong = runner.post(
            "/v1/memory/teach/person",
            self_introduction_payload(
                camera=camera,
                display_name="Wrong Memory E2E Person",
            ),
        )
        before_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=2_000,
            states_file=states_file,
            phase="before-correction",
        )
        before_known = first_event(before_events, "known_person_present")
        if before_known is None:
            raise AssertionError(f"missing pre-correction known event: {before_events}")
        memory_match_id = before_known["evidence"]["memory_match_id"]
        correction = runner.post(
            "/v1/memory/correct-identity",
            {
                "memory_match_id": memory_match_id,
                "wrong_person_id": wrong["person_id"],
            },
        )
        after_events = runner.start_query_and_drain(
            websocket,
            query_timestamp_ms=4_000,
            states_file=states_file,
            phase="after-correction",
        )

    wrong_person_events = [
        event
        for event in after_events
        if event.get("event") == "known_person_present"
        and event.get("memory_context", {}).get("person", {}).get("person_id")
        == wrong["person_id"]
    ]
    if not after_events:
        post_correction_outcome = "no_event"
    elif first_event(after_events, "familiar_unknown_present") is not None:
        post_correction_outcome = "anonymous_or_familiar_unknown"
    else:
        post_correction_outcome = "other_events"
    assertions = {
        "wrong_person_known_before_correction": before_known is not None,
        "correction_ok": correction.get("ok") is True,
        "wrong_person_not_returned_after_correction": not wrong_person_events,
    }
    return {
        "passed": all(assertions.values()),
        "assertions": assertions,
        "wrong_person_id": wrong["person_id"],
        "corrected_memory_match_id": memory_match_id,
        "post_correction_outcome": post_correction_outcome,
        "before_events": compact_events(before_events),
        "after_events": compact_events(after_events),
    }


def check_object_resolve_unsupported_no_write(
    *,
    out: Path,
    source_frame: SourceFrame,
    camera: str,
    embedding_config: MemoryEmbeddingConfig,
    states_file: Any,
) -> dict[str, Any]:
    runner = MemoryE2ERunner(
        case="object-negative",
        out=out,
        source_frame=source_frame,
        camera=camera,
        embedding_config=embedding_config,
    )
    store = runner.client.app.state.memory_service.store
    before_counts = _memory_write_counts(store)
    response = runner.client.post(
        "/v1/memory/resolve-target",
        json=object_resolve_payload(camera=camera, referent_text="手机"),
    )
    body = response.json()
    after_counts = _memory_write_counts(store)
    assertions = {
        "status_code_200": response.status_code == 200,
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


def _track(
    *,
    track_id: int,
    bbox_xyxy: list[float],
    timestamp_ms: int,
    confidence: float = 0.93,
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox_xyxy
    center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    return {
        "track_id": track_id,
        "class": "person",
        "bbox_xyxy": bbox_xyxy,
        "bbox_area_ratio": ((x2 - x1) * (y2 - y1)) / (1280.0 * 720.0),
        "center_uv": center,
        "head_uv": [center[0], y1 + ((y2 - y1) * 0.12)],
        "velocity_uv_s": [0.0, 0.0],
        "age_ms": 800,
        "lost_ms": 0,
        "confidence": confidence,
        "pose_confidence": 0.86,
    }


def _snapshot_track(
    *,
    track_id: int,
    bbox_xyxy: tuple[float, float, float, float],
    timestamp_ms: int,
    confidence: float = 0.93,
    keypoints: tuple[PoseKeypoint, ...] = (),
) -> TrackSnapshot:
    x1, y1, x2, y2 = bbox_xyxy
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return TrackSnapshot(
        track_id=track_id,
        first_seen_ms=timestamp_ms - 800,
        last_seen_ms=timestamp_ms,
        frame_timestamp_ms=timestamp_ms,
        bbox_xyxy=bbox_xyxy,
        confidence=confidence,
        pose_confidence=0.86,
        head_uv=(center[0], y1 + ((y2 - y1) * 0.12)),
        velocity_uv_s=(0.0, 0.0),
        lost_ms=0,
        hits=2,
        misses=0,
        keypoints=keypoints,
    )


def _memory_write_counts(store: Any) -> dict[str, int]:
    tables = (
        "person_profiles",
        "person_embeddings",
        "scene_memories",
        "scene_embeddings",
        "anonymous_profiles",
        "anonymous_embeddings",
    )
    return {
        table: int(
            store.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()[
                "count"
            ]
        )
        for table in tables
    }


def load_source_frame(data_dir: Path, scene: str) -> SourceFrame:
    scene_dir = Path(data_dir) / scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory not found: {scene_dir}")
    paths = sorted(
        [
            path
            for pattern in ("*.jpeg", "*.jpg")
            for path in scene_dir.glob(pattern)
            if path.is_file()
        ]
    )
    if not paths:
        raise FileNotFoundError(f"no JPEG frames found in {scene_dir}")
    path = paths[0]
    jpeg_bytes = path.read_bytes()
    width, height = _parse_jpeg_dimensions(jpeg_bytes, frame_id=None)
    return SourceFrame(path=path, jpeg_bytes=jpeg_bytes, width=width, height=height)


def run_check(
    checks: list[dict[str, Any]],
    name: str,
    callback: Any,
) -> None:
    try:
        details = callback()
        passed = bool(details.pop("passed"))
        checks.append({"name": name, "passed": passed, "details": details})
    except Exception as exc:
        checks.append(
            {
                "name": name,
                "passed": False,
                "details": {"error": str(exc), "error_type": type(exc).__name__},
            }
        )


def first_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") == event_name:
            return event
    return None


def compact_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = []
    for event in events:
        compacted.append(
            {
                "event": event.get("event"),
                "event_id": event.get("event_id"),
                "track_id": event.get("track_id"),
                "evidence": event.get("evidence"),
                "memory_context": event.get("memory_context"),
            }
        )
    return compacted


def build_report(
    *,
    ok: bool,
    args: argparse.Namespace,
    checks: list[dict[str, Any]],
    notes: list[str],
    states_path: Path,
    report_path: Path,
    source_frame: SourceFrame | None,
) -> dict[str, Any]:
    source: dict[str, Any] | None = None
    if source_frame is not None:
        source = {
            "path": str(source_frame.path),
            "width": source_frame.width,
            "height": source_frame.height,
            "bytes": len(source_frame.jpeg_bytes),
            "copied_to": str(Path(args.out) / "source_frame.jpeg"),
        }
    return {
        "ok": ok,
        "gate": "manual_memory_e2e",
        "embedding_backend": args.embedding_backend,
        "embedding": embedding_report(args),
        "data_dir": str(args.data_dir),
        "scene": args.scene,
        "camera": args.camera,
        "source_frame": source,
        "artifacts": {
            "report_json": str(report_path),
            "states_jsonl": str(states_path),
            "runtime_dir": str(Path(args.out) / "runtime"),
        },
        "notes": notes,
        "checks": checks,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
