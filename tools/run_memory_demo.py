from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.public_demo_output import PublicDemoOutputError, resolve_public_demo_out

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("val-data")
DEFAULT_OUT = Path("artifacts/demo/memory")
DEFAULT_CAMERA = "front"
DEFAULT_FACE_MODEL_PATH = Path("runtime/models/face-buffalo-s")
DEFAULT_SCENE_MODEL_PATH = Path("runtime/models/scene-mobileclip2-s0")
DEFAULT_POSE_MODEL_PATH = Path("runtime/models/yolov8n-pose.pt")
PUBLIC_INFERENCE_CACHE_ENV = "VISUAL_EVENTS_INFERENCE_CACHE_DIR"
PUBLIC_INFERENCE_CACHE_RELATIVE_PATH = Path(".cache/inference")
ULTRALYTICS_SETTINGS_VERSION = "0.0.6"

INFERENCE_CACHE_ENV_KEYS = (
    PUBLIC_INFERENCE_CACHE_ENV,
    "YOLO_CONFIG_DIR",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
)
PUBLIC_ARTIFACTS = {
    "report_json": "report.json",
    "index_html": "index.html",
    "visual_evidence_html": "visual-evidence/index.html",
}
PUBLIC_CASES = (
    {
        "case_id": "self_introduction",
        "case_type": "self_intro",
        "title": "记住自我介绍",
        "section_keys": ("self_introduction", "self_smoke"),
        "visual_ids": ("self_introduction", "self_introduction_known_person"),
        "default_expected_target": "当前说话的人",
        "default_expected_outcome": "teach -> recall known person",
        "success_event": "known_person_present",
    },
    {
        "case_id": "third_person_pointing_teach",
        "case_type": "third_person_pointing_teach",
        "title": "第三人称指向示教",
        "section_keys": ("third_person_introduction", "third_person_probe"),
        "visual_ids": ("pointing_teaching", "third_person_pose_pointing"),
        "default_expected_target": "被指向的人",
        "default_expected_outcome": "teach -> recall known person",
        "success_event": "known_person_present",
    },
    {
        "case_id": "scene_teach",
        "case_type": "scene_teach",
        "title": "记住场景",
        "section_keys": ("teach_scene", "scene_smoke"),
        "visual_ids": ("teach_scene", "scene_teaching", "teach_scene_scene_activated"),
        "default_expected_target": "当前场景",
        "default_expected_outcome": "write -> activation scene_activated",
        "success_event": "scene_activated",
    },
    {
        "case_id": "familiar_unknown",
        "case_type": "familiar_unknown",
        "title": "见过但还不知道名字的人",
        "section_keys": ("familiar_unknown",),
        "visual_ids": ("familiar_unknown", "familiar_unknown_present"),
        "default_expected_target": "见过但还不知道名字的人",
        "default_expected_outcome": "no_user_input -> familiar result",
        "success_event": "familiar_unknown_present",
        "passive_observation": True,
        "fallback_scene": "pic_familiar_face",
    },
)
PUBLIC_VISUAL_ID_TO_CASE_ID = {
    visual_id: str(case["case_id"])
    for case in PUBLIC_CASES
    for visual_id in case["visual_ids"]
}
PUBLIC_CHECK_NAMES = {
    "real_model_paths": "real_models_loaded",
    "local_smoke_explicit_real_backends": "real_models_loaded",
    "self_introduction_known_person_present": "self_introduction_remembered",
    "self_local_smoke": "self_introduction_remembered",
    "teach_scene_scene_activated": "scene_teaching_remembered",
    "scene_local_smoke": "scene_teaching_remembered",
    "third_person_pose_pointing_known_person": "pointing_teaching_resolved",
    "third_person_known_person_present": "pointing_teaching_resolved",
    "third_person_local_probe": "pointing_teaching_resolved",
    "familiar_unknown_present": "familiar_unknown_detected",
    "familiar_unknown_local_demo": "familiar_unknown_detected",
    "demo_outputs": "outputs_written",
    "artifact_skeleton": "outputs_written",
}


class MemoryDemoError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public real-model memory demo."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_memory_demo_from_args(args)
    except MemoryDemoError as exc:
        print(f"memory demo failed: {exc}", file=sys.stderr)
        return 2

    print(f"memory demo report: {summary['report_path']}")
    print(f"memory demo index: {summary['index_html']}")
    return 0 if summary.get("ok") is True else 1


def run_memory_demo_from_args(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out = _resolve_public_demo_out(Path(args.out))
    _reject_out_inside_data_dir(out=out, data_dir=data_dir)
    _validate_data_dir(data_dir)
    _validate_default_real_model_paths()
    _prepare_public_output_dir(out)

    with tempfile.TemporaryDirectory(prefix="memory-demo-internal-") as internal_root:
        internal_artifact = Path(internal_root) / "artifact"
        with _public_inference_cache_env(
            Path(internal_root)
        ), _suppress_known_starlette_testclient_warning():
            memory_teaching_evidence, runner = _load_runtime_modules()
            runner.run_real_model_memory_demo(
                data_dir=data_dir,
                out=internal_artifact,
                camera=args.camera,
                person_model_path=DEFAULT_FACE_MODEL_PATH,
                scene_model_path=DEFAULT_SCENE_MODEL_PATH,
                pose_model_path=DEFAULT_POSE_MODEL_PATH,
            )
            internal_report = _read_report(internal_artifact / "report.json")
            render_error: str | None = None
            try:
                summary = memory_teaching_evidence.render_memory_teaching_evidence(
                    artifact=internal_artifact,
                    out=out,
                    public_demo=True,
                )
            except memory_teaching_evidence.MemoryTeachingEvidenceError as exc:
                render_error = str(exc)
                summary = _write_visual_evidence_render_failure(out, render_error)
            visual_evidence_index = summary.get("visual_evidence_index")
            public_report = _build_public_demo_report(
                internal_report,
                data_dir=data_dir,
                out=out,
                camera=args.camera,
                visual_evidence_index=(
                    visual_evidence_index
                    if isinstance(visual_evidence_index, list)
                    else []
                ),
                render_error=render_error,
            )
            _write_json(out / "report.json", public_report)
            _write_public_demo_index_html(out / "index.html", public_report)
            ok = public_report.get("status") == "passed"
            return {
                **summary,
                "ok": ok,
                "report_path": str(out / "report.json"),
                "index_html": str(out / "index.html"),
            }

    raise MemoryDemoError("memory demo did not produce a report")


def _resolve_public_demo_out(out: Path) -> Path:
    try:
        return resolve_public_demo_out(out, repo_root=REPO_ROOT)
    except PublicDemoOutputError as exc:
        raise MemoryDemoError(str(exc)) from exc


def _load_runtime_modules() -> tuple[Any, Any]:
    from tools import memory_teaching_evidence
    from tools import run_memory_teaching_ga_e2e as runner

    return memory_teaching_evidence, runner


@contextmanager
def _public_inference_cache_env(out: Path) -> Any:
    cache_dir = _ensure_public_inference_cache(out)
    previous = {key: os.environ.get(key) for key in INFERENCE_CACHE_ENV_KEYS}
    os.environ[PUBLIC_INFERENCE_CACHE_ENV] = str(cache_dir)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _ensure_public_inference_cache(out: Path) -> Path:
    cache_dir = Path(out) / PUBLIC_INFERENCE_CACHE_RELATIVE_PATH
    for child in ("yolo", "torch", "xdg", "matplotlib"):
        (cache_dir / child).mkdir(parents=True, exist_ok=True)
    _ensure_ultralytics_settings(cache_dir / "yolo" / "Ultralytics")
    return cache_dir


def _ensure_ultralytics_settings(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    settings_path = config_dir / "settings.json"
    if settings_path.is_file() and _valid_ultralytics_settings(settings_path):
        return
    settings_path.write_text(
        json.dumps(
            _ultralytics_settings_defaults(Path.cwd()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _valid_ultralytics_settings(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == set(_ultralytics_settings_defaults(Path.cwd()))
        and value.get("settings_version") == ULTRALYTICS_SETTINGS_VERSION
    )


def _ultralytics_settings_defaults(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    return {
        "api_key": "",
        "clearml": True,
        "comet": True,
        "datasets_dir": str(root.parent / "datasets"),
        "dvc": True,
        "hub": True,
        "mlflow": True,
        "neptune": True,
        "openai_api_key": "",
        "openvino_msg": True,
        "raytune": True,
        "runs_dir": str(root / "runs"),
        "settings_version": ULTRALYTICS_SETTINGS_VERSION,
        "sync": True,
        "tensorboard": False,
        "uuid": "visual-events-memory-demo",
        "vscode_msg": True,
        "wandb": False,
        "weights_dir": str(root / "weights"),
    }


@contextmanager
def _suppress_known_starlette_testclient_warning() -> Any:
    from starlette.exceptions import StarletteDeprecationWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"^Using `httpx` with `starlette\.testclient` is deprecated; "
                r"install `httpx2` instead\.$"
            ),
            category=StarletteDeprecationWarning,
        )
        yield


def _validate_default_real_model_paths() -> None:
    missing: list[str] = []
    for label, path, expected_kind in (
        ("face", DEFAULT_FACE_MODEL_PATH, "directory"),
        ("scene", DEFAULT_SCENE_MODEL_PATH, "directory"),
        ("pose", DEFAULT_POSE_MODEL_PATH, "file"),
    ):
        exists = path.is_dir() if expected_kind == "directory" else path.is_file()
        if not exists:
            missing.append(f"{label} {expected_kind}: {path}")
    if missing:
        raise MemoryDemoError(
            "missing required real model path(s): " + "; ".join(missing)
        )


def _validate_data_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        raise MemoryDemoError(f"data-dir does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise MemoryDemoError(f"data-dir is not a directory: {data_dir}")


def _prepare_public_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        _remove_path(child)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _reject_out_inside_data_dir(*, out: Path, data_dir: Path) -> None:
    out_resolved = _resolve_for_compare(out)
    data_dir_resolved = _resolve_for_compare(data_dir)
    if out_resolved == data_dir_resolved or data_dir_resolved in out_resolved.parents:
        raise MemoryDemoError(
            f"--out must not be inside data-dir: out={out} data-dir={data_dir}"
        )


def _resolve_for_compare(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _read_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MemoryDemoError(f"report.json was not generated: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MemoryDemoError(
            f"report.json is invalid JSON: {path}:{exc.lineno}:{exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise MemoryDemoError(f"report.json must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_visual_evidence_render_failure(out: Path, reason: str) -> dict[str, Any]:
    visual_evidence_dir = Path(out) / "visual-evidence"
    visual_evidence_dir.mkdir(parents=True, exist_ok=True)
    (visual_evidence_dir / "index.html").write_text(
        "<!doctype html><html lang=\"en\"><body>"
        "<h1>Visual evidence render failed</h1>"
        f"<p>{html.escape(reason)}</p>"
        "</body></html>\n",
        encoding="utf-8",
    )
    return {
        "ok": False,
        "out": str(out),
        "index_html": str(Path(out) / "index.html"),
        "visual_evidence_index": [],
        "render_error": reason,
    }


def _build_public_demo_report(
    report: dict[str, Any],
    *,
    data_dir: Path,
    out: Path,
    camera: str,
    visual_evidence_index: list[Any],
    render_error: str | None = None,
) -> dict[str, Any]:
    scene_count = _public_scene_count(report, data_dir=data_dir)
    cases = _public_cases(
        report,
        data_dir=data_dir,
        out=out,
        visual_evidence_index=visual_evidence_index,
    )
    error_count = sum(1 for case in cases if case.get("result_status") != "passed")
    if render_error:
        error_count += 1
    return {
        "demo": "memory",
        "status": _public_report_status(
            report_status=str(report.get("status") or "unknown"),
            cases=cases,
            render_error=render_error,
        ),
        "camera": str(camera),
        "data_dir": str(data_dir),
        "real_model_evidence": report.get("real_model_evidence") is True,
        "models": _public_models(report.get("models")),
        "scene_count": scene_count,
        "case_count": len(cases),
        "error_count": error_count,
        "cases": cases,
        "checks": _public_checks(report.get("checks"), scene_count=scene_count),
        "artifacts": dict(PUBLIC_ARTIFACTS),
    }


def _public_report_status(
    *,
    report_status: str,
    cases: list[dict[str, Any]],
    render_error: str | None,
) -> str:
    if render_error:
        return "failed"
    if report_status != "passed":
        return report_status
    if any(case.get("result_status") != "passed" for case in cases):
        return "failed"
    return report_status


def _public_cases(
    report: dict[str, Any],
    *,
    data_dir: Path,
    out: Path,
    visual_evidence_index: list[Any],
) -> list[dict[str, Any]]:
    visual_by_case_id = _visual_evidence_by_case_id(visual_evidence_index)
    cases: list[dict[str, Any]] = []
    for config in PUBLIC_CASES:
        section = _first_report_section(report, config["section_keys"])
        visual_item = visual_by_case_id.get(str(config["case_id"]))
        if section is None and visual_item is None:
            continue
        case = _public_case(
            config,
            section=section or {},
            visual_item=visual_item,
            data_dir=data_dir,
            out=out,
        )
        cases.append(case)
    return cases


def _visual_evidence_by_case_id(
    visual_evidence_index: list[Any],
) -> dict[str, dict[str, Any]]:
    by_case_id: dict[str, dict[str, Any]] = {}
    for item in visual_evidence_index:
        if not isinstance(item, dict):
            continue
        item_id = _first_text(item.get("id"), item.get("assertion_id"))
        case_id = PUBLIC_VISUAL_ID_TO_CASE_ID.get(item_id)
        if case_id is not None and case_id not in by_case_id:
            by_case_id[case_id] = item
    return by_case_id


def _first_report_section(
    report: dict[str, Any],
    keys: Any,
) -> dict[str, Any] | None:
    for key in keys:
        value = report.get(key)
        if isinstance(value, dict):
            return value
    return None


def _public_case(
    config: dict[str, Any],
    *,
    section: dict[str, Any],
    visual_item: dict[str, Any] | None,
    data_dir: Path,
    out: Path,
) -> dict[str, Any]:
    passive_observation = config.get("passive_observation") is True
    reference_frame, reference_status = _public_frame_ref(
        _first_text(
            section.get("reference_frame"),
            section.get("transcript_source_image"),
            section.get("source_image_path"),
        ),
        data_dir=data_dir,
        out=out,
    )
    selected_frame = _selected_evidence_frame(section)
    if selected_frame is None and passive_observation:
        selected_frame = _first_scene_image(data_dir, str(config.get("fallback_scene") or ""))
    evidence_frame, evidence_status = _public_frame_ref(
        selected_frame,
        data_dir=data_dir,
        out=out,
    )
    if evidence_frame is None and passive_observation:
        selected_frame = _first_scene_image(data_dir, str(config.get("fallback_scene") or ""))
        evidence_frame, evidence_status = _public_frame_ref(
            selected_frame,
            data_dir=data_dir,
            out=out,
        )
    overlay_source_frame, overlay_source_status = _public_frame_ref(
        _first_text(section.get("overlay_source_frame"), selected_frame),
        data_dir=data_dir,
        out=out,
    )
    overlay_image, overlay_status = _public_overlay_image_ref(
        visual_item,
        fallback_image=evidence_frame,
        out=out,
    )

    explicit_failure_reason = _first_text(
        section.get("failure_reason"),
        section.get("reason"),
        _nested_text(section, ("resolve_target", "error_code")),
        _nested_text(section, ("resolve_target", "status")),
    )
    image_failure = _first_image_failure_reason(
        reference_status=reference_status,
        evidence_status=evidence_status,
        overlay_source_status=overlay_source_status,
        overlay_status=overlay_status,
        require_reference=not passive_observation,
    )
    failure_reason = explicit_failure_reason or image_failure
    passed = _section_passed(section) and image_failure is None
    result_status = "passed" if passed else str(section.get("status") or "failed")
    if result_status == "present":
        result_status = "passed"
    if image_failure is not None and result_status == "passed":
        result_status = "failed"
    if passive_observation:
        user_message = "无用户输入，来自重复观察"
    else:
        user_message = _first_text(
            section.get("user_message"),
            section.get("transcript_text"),
            "not present",
        )
    event_result = _event_result(section, str(config.get("success_event") or ""))
    actual_outcome = _first_text(
        section.get("actual_outcome"),
        event_result,
        _status_sentence(section),
    )
    expected_outcome = _first_text(
        section.get("expected_outcome"),
        _nested_text(section, ("expected", "outcome")),
        str(config.get("default_expected_outcome") or ""),
    )
    actual_target = _first_text(
        section.get("actual_target"),
        _public_actual_target(section, config),
    )
    expected_target = _first_text(
        section.get("expected_target"),
        _nested_text(section, ("expected", "target")),
        str(config.get("default_expected_target") or ""),
    )
    frame_relation = _public_frame_relation(
        section=section,
        passive_observation=passive_observation,
        result_status=result_status,
        reference_frame=reference_frame,
        evidence_frame=evidence_frame,
    )
    frame_delta = section.get("frame_delta")
    frame_window = _public_frame_window_fields(section, data_dir=data_dir, out=out)
    case = {
        "case_id": str(config["case_id"]),
        "case_type": str(config["case_type"]),
        "title": str(config["title"]),
        "user_message": user_message,
        "reference_frame": reference_frame,
        "evidence_frame": evidence_frame,
        "overlay_source_frame": overlay_source_frame,
        "frame_relation": frame_relation,
        "frame_delta": frame_delta if frame_delta is not None else "not available",
        "expected_target": expected_target,
        "actual_target": actual_target,
        "expected_outcome": expected_outcome,
        "actual_outcome": actual_outcome,
        "verdict": _public_verdict(
            case_id=str(config["case_id"]),
            explicit=_first_text(section.get("verdict")),
            result_status=result_status,
            failure_reason=failure_reason,
            actual_outcome=actual_outcome,
        ),
        "result_status": result_status,
        "result_reason": failure_reason or "ok",
        "failure_reason": failure_reason,
        "selected_target": _public_selected_target(section, config),
        "identity_result": _identity_result(section, event_result),
        "familiar_result": _familiar_result(section, event_result),
        "scene_result": _scene_result(section, event_result),
        "event_result": event_result or "not present",
        "overlay_image": overlay_image,
        "overlay_status": overlay_status,
        **frame_window,
    }
    if str(config.get("case_id") or "") == "familiar_unknown":
        case.update(_public_familiar_metrics(section))
    return {key: value for key, value in case.items() if value is not None}


def _public_frame_window_fields(
    section: dict[str, Any],
    *,
    data_dir: Path,
    out: Path,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    anchor_frame, _anchor_status = _public_frame_ref(
        _first_text(
            section.get("anchor_frame"),
            _nested_value(section, ("candidate_frame_window", "anchor_frame")),
        ),
        data_dir=data_dir,
        out=out,
    )
    if anchor_frame is not None:
        fields["anchor_frame"] = anchor_frame

    candidate_values = _first_list(
        section.get("candidate_frames"),
        _nested_value(section, ("candidate_frame_window", "candidate_frames")),
    )
    if candidate_values is not None:
        candidate_frames: list[str] = []
        for value in candidate_values:
            candidate_frame, _candidate_status = _public_frame_ref(
                value,
                data_dir=data_dir,
                out=out,
            )
            if candidate_frame is not None:
                candidate_frames.append(candidate_frame)
        fields["candidate_frames"] = candidate_frames
        fields["candidate_frame_count"] = len(candidate_frames)

    radius = _first_present(
        section.get("frame_window_radius"),
        _nested_value(section, ("candidate_frame_window", "frame_window_radius")),
    )
    if radius is not None:
        fields["frame_window_radius"] = radius
    return fields


def _selected_evidence_frame(section: dict[str, Any]) -> str | None:
    selected_window = section.get("selected_window")
    if isinstance(selected_window, dict):
        frame = selected_window.get("frame")
        if isinstance(frame, str) and frame:
            return frame
    return _first_text(
        section.get("evidence_frame"),
        section.get("selected_evidence_frame"),
        section.get("selected_frame"),
    ) or None


def _first_scene_image(data_dir: Path, scene_name: str) -> str | None:
    if not scene_name:
        return None
    scene_dir = Path(data_dir) / scene_name
    try:
        paths = sorted(
            path
            for path in scene_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg"}
        )
    except OSError:
        return None
    return str(paths[0]) if paths else None


def _public_frame_ref(
    value: Any,
    *,
    data_dir: Path,
    out: Path,
) -> tuple[str | None, str]:
    text = _first_text(value)
    if not text:
        return None, "missing"
    path = _resolve_public_image_path(text, data_dir=data_dir, out=out)
    if path is None or not path.is_file():
        return None, "missing"
    return _relative_href(path, out), "present"


def _public_overlay_image_ref(
    visual_item: dict[str, Any] | None,
    *,
    fallback_image: str | None,
    out: Path,
) -> tuple[str | None, str]:
    image = None
    if isinstance(visual_item, dict):
        image = _first_text(visual_item.get("image"), visual_item.get("path"))
    if image:
        path = Path(image)
        if not path.is_absolute() and ".." not in path.parts and (Path(out) / path).is_file():
            return path.as_posix(), "present"
        if path.is_absolute() and path.is_file():
            return _relative_href(path, out), "present"
    if fallback_image:
        return fallback_image, "fallback_evidence_frame"
    return None, "missing"


def _resolve_public_image_path(text: str, *, data_dir: Path, out: Path) -> Path | None:
    raw_path = Path(text)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                Path(out) / raw_path,
                Path(data_dir) / raw_path,
                Path.cwd() / raw_path,
                raw_path,
            ]
        )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _relative_href(path: Path, out: Path) -> str:
    try:
        return os.path.relpath(
            Path(path).resolve(strict=False),
            Path(out).resolve(strict=False),
        ).replace(os.sep, "/")
    except ValueError:
        return Path(path).name


def _first_image_failure_reason(
    *,
    reference_status: str,
    evidence_status: str,
    overlay_source_status: str,
    overlay_status: str,
    require_reference: bool,
) -> str | None:
    if require_reference and reference_status != "present":
        return "reference frame missing"
    if evidence_status != "present":
        return "evidence frame missing"
    if overlay_source_status != "present":
        return "overlay source frame missing"
    if overlay_status == "missing":
        return "overlay image missing"
    return None


def _public_frame_relation(
    *,
    section: dict[str, Any],
    passive_observation: bool,
    result_status: str,
    reference_frame: str | None,
    evidence_frame: str | None,
) -> str:
    explicit = _first_text(section.get("frame_relation"))
    if explicit:
        return explicit
    if passive_observation:
        return "no_user_input"
    if result_status != "passed" and evidence_frame is None:
        return "failed"
    if reference_frame and evidence_frame and reference_frame == evidence_frame:
        return "same"
    if reference_frame and evidence_frame:
        return "nearby"
    return "failed"


def _public_verdict(
    *,
    case_id: str,
    explicit: str,
    result_status: str,
    failure_reason: str | None,
    actual_outcome: str,
) -> str:
    if explicit and result_status == "passed":
        return explicit
    if result_status == "passed":
        return f"通过：{_human_public_text(actual_outcome, case_id=case_id)}"
    return (
        "未通过："
        + _human_failure_reason(
            failure_reason or actual_outcome or "insufficient visual evidence",
            case_id=case_id,
        )
    )


def _public_familiar_metrics(section: dict[str, Any]) -> dict[str, Any]:
    event = _first_event(section, "familiar_unknown_present")
    anonymous = (
        event.get("memory_context", {}).get("anonymous_person")
        if isinstance(event.get("memory_context"), dict)
        else None
    )
    if not isinstance(anonymous, dict):
        anonymous = {}
    seen_count = _first_present(section.get("seen_count"), anonymous.get("seen_count"))
    duration_ms = _first_present(
        section.get("observed_duration_ms"),
        anonymous.get("observed_duration_ms"),
    )
    score = _first_present(section.get("familiar_score"), anonymous.get("familiar_score"))
    fields: dict[str, Any] = {}
    if seen_count is not None:
        fields["familiar_seen_count"] = seen_count
    if duration_ms is not None:
        fields["familiar_duration"] = _format_duration_ms(duration_ms)
    if score is not None:
        fields["familiar_score"] = score
    return fields


def _first_event(section: dict[str, Any], event_name: str) -> dict[str, Any]:
    events = section.get("events")
    if not isinstance(events, list):
        return {}
    for event in events:
        if isinstance(event, dict) and event.get("event") == event_name:
            return event
    return {}


def _format_duration_ms(value: Any) -> str:
    try:
        seconds = float(value) / 1000.0
    except (TypeError, ValueError):
        return str(value)
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:.1f}s"


def _status_sentence(section: dict[str, Any]) -> str:
    status = _first_text(section.get("status"), "unknown")
    if status == "passed":
        return "passed"
    return status


def _event_result(section: dict[str, Any], preferred_event: str) -> str:
    events = section.get("events")
    if not isinstance(events, list):
        return ""
    names: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        name = event.get("event")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    if preferred_event and preferred_event in names:
        return preferred_event
    return ", ".join(names)


def _public_actual_target(section: dict[str, Any], config: dict[str, Any]) -> str:
    case_id = str(config.get("case_id") or "")
    if case_id == "scene_teach" and section.get("scene_id"):
        return "已教学场景"
    if case_id == "familiar_unknown":
        return "熟悉的未命名人物"
    if section.get("person_id"):
        return "已记住的人"
    resolve_target = section.get("resolve_target")
    if isinstance(resolve_target, dict):
        status = _first_text(resolve_target.get("status"))
        if status == "resolved":
            return "已选中目标人物"
        if status:
            return status
    return _first_text(section.get("status"), "not present")


def _public_selected_target(section: dict[str, Any], config: dict[str, Any]) -> str:
    explicit = _first_text(section.get("selected_target"))
    if explicit:
        return explicit
    case_id = str(config.get("case_id") or "")
    if case_id == "scene_teach":
        return "scene"
    if case_id == "familiar_unknown":
        return "representative familiar person"
    return _public_actual_target(section, config)


def _identity_result(section: dict[str, Any], event_result: str) -> str:
    if "known_person_present" in event_result:
        return "known_person_present"
    status = _first_text(_nested_text(section, ("identity_result", "status")))
    return status or "not present"


def _familiar_result(section: dict[str, Any], event_result: str) -> str:
    if "familiar_unknown_present" in event_result:
        return "familiar_unknown_present"
    status = _first_text(_nested_text(section, ("familiar_result", "status")))
    return status or "not present"


def _scene_result(section: dict[str, Any], event_result: str) -> str:
    if "scene_activated" in event_result:
        return "scene_activated"
    status = _first_text(_nested_text(section, ("scene_result", "status")))
    return status or "not present"


def _nested_text(value: Any, keys: tuple[str, ...]) -> str:
    return _first_text(_nested_value(value, keys))


def _nested_value(value: Any, keys: tuple[str, ...]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_list(*values: Any) -> list[Any] | None:
    for value in values:
        if isinstance(value, list):
            return value
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value:
                return value
            continue
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, (dict, list)):
            compact = _compact_public_value(value)
            if compact:
                return compact
    return ""


def _compact_public_value(value: Any) -> str:
    sanitized = _sanitize_public_value(value)
    if sanitized in ({}, [], None):
        return ""
    if isinstance(sanitized, str):
        return sanitized
    return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)


def _sanitize_public_value(value: Any) -> Any:
    forbidden_fragments = (
        "track_id",
        "track",
        "bbox",
        "embedding",
        "crop",
        "keypoint",
        "snapshot_ref",
        "target_ref",
        "source_frame_ref",
        "memory_match_id",
    )
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if any(fragment in lowered for fragment in forbidden_fragments):
                continue
            sanitized_child = _sanitize_public_value(child)
            if sanitized_child in ({}, [], None, ""):
                continue
            result[key_text] = sanitized_child
        return result
    if isinstance(value, list):
        return [
            child
            for child in (_sanitize_public_value(item) for item in value)
            if child not in ({}, [], None, "")
        ]
    if isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in forbidden_fragments):
            return ""
        return value
    return value


def _write_public_demo_index_html(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_public_demo_index_html(report), encoding="utf-8")


def _render_public_demo_index_html(report: dict[str, Any]) -> str:
    status = html.escape(str(report.get("status") or "unknown"))
    real_model = html.escape(str(report.get("real_model_evidence")))
    case_count = html.escape(str(report.get("case_count") or 0))
    error_count = html.escape(str(report.get("error_count") or 0))
    summary_sentence = html.escape(_public_demo_summary_sentence(report))
    cases = report.get("cases") if isinstance(report.get("cases"), list) else []
    cards = "\n".join(
        _public_case_card_html(case)
        for case in cases
        if isinstance(case, dict)
    )
    if not cards:
        cards = "<p>No memory demo cases were present in the source report.</p>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Memory Demo</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.45; color: #1f2933; max-width: 1180px; }}
    a {{ color: #0b63ce; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
    .summary {{ margin: 12px 0 20px; }}
    .case {{ border: 1px solid #d9dee7; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .case h3 {{ margin: 0 0 8px; }}
    .status {{ font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 12px 0; }}
    .window-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }}
    .panel {{ background: #f8fafc; border-radius: 6px; padding: 10px; }}
    .wide {{ grid-column: 1 / -1; }}
    .label {{ color: #52606d; font-size: 0.9rem; margin-bottom: 4px; }}
    .image-link {{ display: block; text-decoration: none; }}
    .image-link span {{ display: block; margin-top: 4px; }}
    .thumb {{ display: block; max-height: 180px; object-fit: contain; background: #fff; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9dee7; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d9dee7; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; width: 180px; }}
  </style>
</head>
<body>
  <h1>Memory Demo</h1>
  <h2>Demo Summary</h2>
  <p class="summary"><strong>演示结果：{summary_sentence}</strong><br>机器状态（machine-readable）: <span class="status">{status}</span>; real_model_evidence: {real_model}; cases: {case_count}; failures: {error_count}</p>
  {_public_models_html(report.get("models"))}
  <ul>
    <li><a href="visual-evidence/index.html">Visual evidence page</a></li>
    <li><a href="report.json">report.json</a></li>
  </ul>
  <h2>Cases</h2>
  {cards}
</body>
</html>
"""


def _public_case_card_html(case: dict[str, Any]) -> str:
    case_id = str(case.get("case_id") or "")
    title = html.escape(str(case.get("title") or case.get("case_id") or "case"))
    status = html.escape(_human_status(case.get("result_status")))
    verdict = html.escape(
        _human_public_text(case.get("verdict"), case_id=case_id, field="verdict")
    )
    user_message = html.escape(
        _human_public_text(case.get("user_message"), case_id=case_id)
    )
    reference = _image_link_html(case.get("reference_frame"), "参考帧")
    evidence = _image_link_html(case.get("evidence_frame"), "证据帧")
    overlay = _image_link_html(case.get("overlay_image"), "证据图")
    window = _candidate_window_html(case)
    expected = html.escape(
        _human_public_text(case.get("expected_outcome"), case_id=case_id)
    )
    actual = html.escape(
        _human_public_text(case.get("actual_outcome"), case_id=case_id)
    )
    expected_target = html.escape(
        _human_public_text(case.get("expected_target"), case_id=case_id)
    )
    actual_target = html.escape(
        _human_public_text(case.get("actual_target"), case_id=case_id)
    )
    details = [
        ("帧关系", case.get("frame_relation")),
        ("帧间隔", case.get("frame_delta")),
        ("原因", _case_display_reason(case)),
        ("选中目标", case.get("selected_target")),
        ("人物识别", case.get("identity_result")),
        ("熟悉度结果", case.get("familiar_result")),
        ("场景结果", case.get("scene_result")),
        ("触发结果", case.get("event_result")),
        ("证据图状态", case.get("overlay_status")),
        ("熟悉度", _familiar_metric_text(case)),
    ]
    detail_rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(str(label))}</th>"
        f"<td>{html.escape(_human_public_text(value, case_id=case_id))}</td>"
        "</tr>"
        for label, value in details
        if value is not None
    )
    return f"""<section class="case">
    <h3>{title} <span class="status">{status}</span></h3>
    <div class="grid">
      <div class="panel"><div class="label">用户说了什么</div><div>{user_message}</div></div>
      <div class="panel"><div class="label">系统看的帧</div><div>{evidence}</div></div>
      <div class="panel"><div class="label">一句话 verdict</div><div>{verdict}</div></div>
      <div class="panel"><div class="label">打开证据图</div><div>{overlay}</div></div>
    </div>
    <div class="grid">
      <div class="panel"><div class="label">参考帧</div><div>{reference}</div></div>
      <div class="panel"><div class="label">期望 vs 实际</div><div>期望：{expected_target}; {expected}<br>实际：{actual_target}; {actual}</div></div>
      {window}
    </div>
    <table><tbody>{detail_rows}</tbody></table>
  </section>"""


def _image_link_html(value: Any, label: str) -> str:
    text = _first_text(value)
    if not text:
        return "未提供"
    escaped = html.escape(text, quote=True)
    escaped_label = html.escape(label)
    return (
        f"<a class=\"image-link\" href=\"{escaped}\">"
        f"<img class=\"thumb\" src=\"{escaped}\" alt=\"{escaped_label}\">"
        f"<span>{escaped_label}: <code>{escaped}</code></span>"
        "</a>"
    )


def _candidate_window_html(case: dict[str, Any]) -> str:
    links: list[str] = []
    anchor = _image_link_html(case.get("anchor_frame"), "锚点帧")
    if anchor != "未提供":
        links.append(anchor)
    candidate_frames = case.get("candidate_frames")
    if isinstance(candidate_frames, list):
        for index, frame in enumerate(candidate_frames, start=1):
            link = _image_link_html(frame, f"候选帧 {index}")
            if link != "未提供":
                links.append(link)
    if not links:
        return ""
    return (
        '<div class="panel wide"><div class="label">候选窗口</div>'
        '<div class="window-grid">'
        + "\n".join(links)
        + "</div></div>"
    )


def _public_demo_summary_sentence(report: dict[str, Any]) -> str:
    cases = report.get("cases") if isinstance(report.get("cases"), list) else []
    public_cases = [case for case in cases if isinstance(case, dict)]
    passed_count = sum(
        1 for case in public_cases if case.get("result_status") == "passed"
    )
    failed_cases = [
        case for case in public_cases if case.get("result_status") != "passed"
    ]
    if not failed_cases:
        return f"{passed_count} 个用例通过"
    unbound_count = sum(1 for case in failed_cases if _case_is_target_unbound(case))
    if unbound_count == len(failed_cases):
        return f"{passed_count} 个通过，{unbound_count} 个因目标不明确未绑定"
    if unbound_count:
        other_count = len(failed_cases) - unbound_count
        return (
            f"{passed_count} 个通过，{unbound_count} 个因目标不明确未绑定，"
            f"{other_count} 个未通过"
        )
    return f"{passed_count} 个通过，{len(failed_cases)} 个未通过"


def _case_is_target_unbound(case: dict[str, Any]) -> bool:
    if str(case.get("case_id") or "") == "third_person_pointing_teach":
        return case.get("result_status") != "passed"
    text = " ".join(
        str(case.get(key) or "")
        for key in ("failure_reason", "result_reason", "actual_target", "actual_outcome")
    ).lower()
    return any(
        marker in text
        for marker in (
            "multiple_candidates",
            "ambiguous",
            "insufficient visual evidence",
            "未能稳定选中",
            "目标不明确",
        )
    )


def _case_display_reason(case: dict[str, Any]) -> str | None:
    reason = case.get("result_reason")
    if reason is None:
        return None
    if case.get("result_status") != "passed" and _case_is_target_unbound(case):
        return "目标不明确，未绑定到任何人"
    return str(reason)


def _familiar_metric_text(case: dict[str, Any]) -> str | None:
    seen_count = case.get("familiar_seen_count")
    duration = case.get("familiar_duration")
    score = case.get("familiar_score")
    parts: list[str] = []
    if seen_count is not None:
        parts.append(f"见过 {seen_count} 次")
    if duration is not None:
        parts.append(f"累计 {duration}")
    if score is not None:
        parts.append(f"score {score}")
    return " / ".join(parts) if parts else None


def _human_status(value: Any) -> str:
    text = _first_text(value)
    return {
        "passed": "通过",
        "failed": "未通过",
        "present": "已生成",
        "missing": "缺失",
        "unknown": "未知",
        "insufficient_sample": "样本不足",
    }.get(text, _human_public_text(text))


def _human_failure_reason(value: Any, *, case_id: str = "") -> str:
    text = _first_text(value)
    lowered = text.lower()
    if case_id == "third_person_pointing_teach" and any(
        marker in lowered
        for marker in (
            "multiple_candidates",
            "ambiguous",
            "insufficient visual evidence",
            "insufficient_sample",
            "not_found",
        )
    ):
        return "目标不明确，未绑定到任何人"
    if "multiple_candidates" in lowered or "ambiguous" in lowered:
        return "候选目标不唯一"
    if "insufficient visual evidence" in lowered:
        return "视觉证据不足"
    return _human_public_text(text, case_id=case_id)


def _human_public_text(value: Any, *, case_id: str = "", field: str = "") -> str:
    text = _first_text(value)
    if not text:
        return "未提供"
    exact = {
        "known_person_present": "已识别为记住的人",
        "familiar_unknown_present": "识别为见过但还不知道名字的人",
        "scene_activated": "已识别为记住的场景",
        "not present": "未提供",
        "not_present": "未提供",
        "missing": "缺失",
        "present": "已生成",
        "passed": "通过",
        "failed": "未通过",
        "ok": "正常",
        "fallback_evidence_frame": "使用原始证据帧",
        "no_user_input": "无用户输入",
        "nearby": "相邻帧",
        "same": "同一帧",
        "different": "不同帧",
        "insufficient_sample": "样本不足",
        "not available": "不可用",
        "multiple_candidates": "候选目标不唯一",
        "ambiguous": "候选目标不唯一",
        "representative familiar person": "熟悉的未命名人物",
        "scene": "场景",
    }
    if text in exact:
        return exact[text]
    if (
        case_id == "third_person_pointing_teach"
        and field == "verdict"
        and "insufficient visual evidence" in text.lower()
    ):
        return "未通过：目标不明确，未绑定到任何人"
    replacements = (
        ("no_user_input -> familiar result", "无需用户输入，识别为熟悉的未命名人物"),
        ("teach -> recall known person", "教学后再次看到能认出这个人"),
        ("write -> activation scene_activated", "写入场景后再次看到能激活该场景"),
        ("known_person_present", "已识别为记住的人"),
        ("familiar_unknown_present", "识别为见过但还不知道名字的人"),
        ("scene_activated", "已识别为记住的场景"),
        ("multiple_candidates", "候选目标不唯一"),
        ("ambiguous", "候选目标不唯一"),
        ("insufficient visual evidence", "视觉证据不足"),
        ("not available", "不可用"),
        ("not present", "未提供"),
        ("not_present", "未提供"),
        ("fallback_evidence_frame", "使用原始证据帧"),
        ("no_user_input", "无用户输入"),
    )
    result = text
    for source, replacement in replacements:
        result = result.replace(source, replacement)
    return result


def _public_models_html(models: Any) -> str:
    if not isinstance(models, dict) or not models:
        return "<h2>Models</h2><p>未提供</p>"
    rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(str(key))}</th>"
        f"<td><code>{html.escape(str(value))}</code></td>"
        "</tr>"
        for key, value in sorted(models.items())
    )
    return f"<h2>Models</h2><table><tbody>{rows}</tbody></table>"


def _public_models(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(path)
        for key, path in value.items()
        if key is not None and path is not None
    }


def _public_scene_count(report: dict[str, Any], *, data_dir: Path) -> int:
    value = report.get("scene_count")
    if isinstance(value, int) and value > 0:
        return value
    scenes = report.get("scenes")
    if isinstance(scenes, list):
        scene_count = sum(1 for item in scenes if isinstance(item, dict))
        if scene_count > 0:
            return scene_count
    return _count_jpeg_scene_dirs(data_dir)


def _count_jpeg_scene_dirs(data_dir: Path) -> int:
    try:
        children = list(Path(data_dir).iterdir())
    except OSError:
        return 0
    count = 0
    for child in children:
        if not child.is_dir():
            continue
        try:
            has_jpeg = any(
                path.is_file() and path.suffix.lower() in {".jpeg", ".jpg"}
                for path in child.iterdir()
            )
        except OSError:
            has_jpeg = False
        if has_jpeg:
            count += 1
    return count


def _section_passed(section: dict[str, Any]) -> bool:
    if "passed" in section:
        return section.get("passed") is True
    return section.get("status") == "passed"


def _public_checks(value: Any, *, scene_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = []
    checks_by_name: dict[str, bool] = {}
    checks_by_name["scenes_loaded"] = scene_count > 0
    for item in value:
        if not isinstance(item, dict):
            continue
        internal_name = item.get("name")
        if not isinstance(internal_name, str) or not internal_name:
            continue
        public_name = PUBLIC_CHECK_NAMES.get(internal_name)
        if public_name is None:
            continue
        checks_by_name[public_name] = (
            checks_by_name.get(public_name, True) and item.get("passed") is True
        )
    return [
        {"name": name, "passed": checks_by_name[name]}
        for name in (
            "real_models_loaded",
            "scenes_loaded",
            "self_introduction_remembered",
            "scene_teaching_remembered",
            "pointing_teaching_resolved",
            "familiar_unknown_detected",
            "outputs_written",
        )
        if name in checks_by_name
    ]


if __name__ == "__main__":
    raise SystemExit(main())
