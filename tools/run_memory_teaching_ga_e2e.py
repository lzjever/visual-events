from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        default=True,
        help="Write stub API responses without calling a server. Currently required.",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_dry_run(data_dir=args.data_dir, out=args.out, camera=args.camera)
    print(f"memory teaching GA runner dry-run {'passed' if report['ok'] else 'failed'}")
    print(f"scenes: {report['scene_count']}")
    print(f"report: {Path(args.out) / 'report.json'}")
    return 0 if report["ok"] else 1


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
                "dry_run": True,
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
  <title>Memory Teaching GA Dry Run Evidence</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; line-height: 1.4; }}
    code, pre {{ background: #f5f5f5; }}
    pre {{ padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Memory Teaching GA Dry Run Evidence</h1>
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
