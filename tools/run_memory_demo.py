from __future__ import annotations

import argparse
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
            try:
                summary = memory_teaching_evidence.render_memory_teaching_evidence(
                    artifact=internal_artifact,
                    out=out,
                    public_demo=True,
                )
            except memory_teaching_evidence.MemoryTeachingEvidenceError as exc:
                raise MemoryDemoError(str(exc)) from exc
        internal_report = _read_report(internal_artifact / "report.json")
    visual_evidence_index = summary.get("visual_evidence_index")
    public_report = _build_public_demo_report(
        internal_report,
        data_dir=data_dir,
        out=out,
        camera=args.camera,
        visual_evidence_index=(
            visual_evidence_index if isinstance(visual_evidence_index, list) else []
        ),
    )
    _write_json(out / "report.json", public_report)
    ok = public_report.get("status") == "passed"
    return {
        **summary,
        "ok": ok,
        "report_path": str(out / "report.json"),
    }


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


def _build_public_demo_report(
    report: dict[str, Any],
    *,
    data_dir: Path,
    out: Path,
    camera: str,
    visual_evidence_index: list[Any],
) -> dict[str, Any]:
    del out, camera, visual_evidence_index
    scene_count = _public_scene_count(report, data_dir=data_dir)
    return {
        "demo": "memory",
        "status": str(report.get("status") or "unknown"),
        "real_model_evidence": report.get("real_model_evidence") is True,
        "models": _public_models(report.get("models")),
        "scene_count": scene_count,
        "demo_items": _public_demo_items(report),
        "familiar_unknown": _public_familiar_unknown_result(
            report.get("familiar_unknown")
        ),
        "checks": _public_checks(report.get("checks"), scene_count=scene_count),
        "artifacts": dict(PUBLIC_ARTIFACTS),
    }


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


def _public_demo_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        _public_section_item(
            report,
            source_key="self_introduction",
            item_id="self_introduction",
            label="记住自我介绍",
        ),
        _public_section_item(
            report,
            source_key="teach_scene",
            item_id="teach_scene",
            label="记住场景",
        ),
        _public_section_item(
            report,
            source_key="third_person_introduction",
            item_id="pointing_teaching",
            label="第三人称指向示教",
        ),
        {
            "id": "familiar_unknown",
            "label": "匿名熟客出现",
            **_public_familiar_unknown_result(report.get("familiar_unknown")),
        },
    ]
    return [item for item in items if item.get("passed") is True]


def _public_section_item(
    report: dict[str, Any],
    *,
    source_key: str,
    item_id: str,
    label: str,
) -> dict[str, Any]:
    section = report.get(source_key)
    if not isinstance(section, dict):
        return {"id": item_id, "label": label, "status": "not_present", "passed": False}
    item = {
        "id": item_id,
        "label": label,
        "status": str(section.get("status") or "unknown"),
        "passed": _section_passed(section),
        "scene": section.get("scene"),
    }
    for key in (
        "scene_id",
        "seen_count",
        "observed_duration_ms",
        "familiar_score",
    ):
        if section.get(key) is not None:
            item[key] = section.get(key)
    return item


def _public_familiar_unknown_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"status": "not_present", "passed": False}
    result = {
        "status": str(value.get("status") or "unknown"),
        "passed": _section_passed(value),
        "scene": value.get("scene"),
        "seen_count": value.get("seen_count"),
        "observed_duration_ms": value.get("observed_duration_ms"),
        "familiar_score": value.get("familiar_score"),
        "observation_count": value.get("observation_count"),
        "required_seen_count": value.get("required_seen_count"),
        "required_observed_duration_ms": value.get("required_observed_duration_ms"),
    }
    return {key: child for key, child in result.items() if child is not None}


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
