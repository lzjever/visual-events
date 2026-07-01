from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import memory_teaching_evidence
from tools import run_memory_demo as module


PUBLIC_FORBIDDEN_TEXT = (
    "track_id",
    "bbox_xyxy",
    "request_snapshot_ref",
    "resolver_target_ref",
    "embedding_id",
    "memory_match_id",
    "Debug JSON",
    "assertion_id",
    "source-artifact",
    "artifact_skeleton",
    "crop_path_or_artifact_ref",
    "embedding_backend",
    "inference_backend",
    "local-smoke",
)


@pytest.fixture(autouse=True)
def _public_demo_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)


def _make_model_paths(root: Path) -> dict[str, Path]:
    face = root / "runtime" / "models" / "face-buffalo-s"
    scene = root / "runtime" / "models" / "scene-mobileclip2-s0"
    pose = root / "runtime" / "models" / "yolov8n-pose.pt"
    face.mkdir(parents=True)
    scene.mkdir(parents=True)
    pose.parent.mkdir(parents=True, exist_ok=True)
    pose.write_bytes(b"pose")
    return {"face": face, "scene": scene, "pose": pose}


def _patch_default_models(monkeypatch: pytest.MonkeyPatch, paths: dict[str, Path]) -> None:
    monkeypatch.setattr(module, "DEFAULT_FACE_MODEL_PATH", paths["face"])
    monkeypatch.setattr(module, "DEFAULT_SCENE_MODEL_PATH", paths["scene"])
    monkeypatch.setattr(module, "DEFAULT_POSE_MODEL_PATH", paths["pose"])


def _write_minimal_runner_report(out: Path, **overrides: Any) -> dict[str, Any]:
    report = {
        "ok": True,
        "status": "passed",
        "gate": "memory_demo_real_model",
        "mode": "memory-demo",
        "backend": "local",
        "embedding_backend": "local",
        "inference_backend": "ultralytics",
        "real_model_evidence": True,
        "models": {
            "pose": str(module.DEFAULT_POSE_MODEL_PATH),
            "face": str(module.DEFAULT_FACE_MODEL_PATH),
            "scene": str(module.DEFAULT_SCENE_MODEL_PATH),
        },
        "scene_count": 0,
        "scenes": [],
        "manifest": {"matches_actual_scene_dirs": True},
        "teach_requests": [],
        "checks": [],
        "artifacts": {"report_json": "report.json"},
    }
    report.update(overrides)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "teach_payloads.json").write_text(
        json.dumps({"schema_version": 1, "mode": "memory-demo", "payloads": []}),
        encoding="utf-8",
    )
    return report


def _familiar_unknown_result() -> dict[str, Any]:
    return {
        "status": "passed",
        "passed": True,
        "scene": "pic_familiar_face",
        "anonymous_id": "anon-demo-1",
        "seen_count": 2,
        "observed_duration_ms": 3000,
        "familiar_score": 1.0,
        "events": [
            {
                "event": "familiar_unknown_present",
                "event_id": "evt-familiar-1",
                "evidence": {
                    "memory_match_id": "anon-demo-1",
                    "crop_path_or_artifact_ref": "internal-crop-ref",
                },
                "memory_context": {
                    "anonymous_person": {
                        "anonymous_id": "anon-demo-1",
                        "seen_count": 2,
                        "observed_duration_ms": 3000,
                        "familiar_score": 1.0,
                    }
                },
            }
        ],
        "selected_window": {"frame": "runtime/memory-demo-familiar-unknown/frame.jpg"},
    }


def _patch_runtime_modules(
    monkeypatch: pytest.MonkeyPatch,
    fake_runner: Any,
) -> None:
    monkeypatch.setattr(
        module,
        "_load_runtime_modules",
        lambda: (memory_teaching_evidence, fake_runner),
    )


def test_help_is_clean_and_only_lists_public_args() -> None:
    result = subprocess.run(
        [sys.executable, "tools/run_memory_demo.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "--data-dir" in result.stdout
    assert "--out" in result.stdout
    assert "--camera" in result.stdout
    for hidden_arg in [
        "--artifact",
        "--backend",
        "--embedding-backend",
        "--inference-backend",
        "--run-local-smoke",
        "--local-smoke",
        "--pose-model-path",
        "--person-model-path",
        "--scene-model-path",
    ]:
        assert hidden_arg not in result.stdout


def test_parse_args_keeps_public_surface_minimal() -> None:
    defaults = module.parse_args([])
    assert defaults.data_dir == Path("val-data")
    assert defaults.out == Path("artifacts/demo/memory")
    assert defaults.camera == module.DEFAULT_CAMERA

    args = module.parse_args(["--data-dir", "val-data", "--out", "out", "--camera", "front"])

    assert args.data_dir == Path("val-data")
    assert args.out == Path("out")
    assert args.camera == "front"

    for hidden_arg in [
        "--artifact",
        "--backend",
        "--embedding-backend",
        "--inference-backend",
        "--run-local-smoke",
        "--local-smoke",
        "--pose-model-path",
        "--person-model-path",
        "--scene-model-path",
    ]:
        with pytest.raises(SystemExit):
            module.parse_args([hidden_arg, "x"])


def test_main_uses_default_real_model_paths_and_renders_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    out = tmp_path / "artifacts" / "demo" / "memory"
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        internal_out = kwargs["out"]
        assert internal_out != out
        assert internal_out.name == "artifact"
        assert os.environ[module.PUBLIC_INFERENCE_CACHE_ENV] == str(
            internal_out.parent / module.PUBLIC_INFERENCE_CACHE_RELATIVE_PATH
        )
        assert (
            internal_out.parent
            / module.PUBLIC_INFERENCE_CACHE_RELATIVE_PATH
            / "yolo"
            / "Ultralytics"
            / "settings.json"
        ).is_file()
        return _write_minimal_runner_report(kwargs["out"])

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(data_dir), "--out", str(out)]) == 0

    assert len(calls) == 1
    assert calls[0]["data_dir"] == data_dir
    assert calls[0]["camera"] == module.DEFAULT_CAMERA
    assert calls[0]["person_model_path"] == paths["face"]
    assert calls[0]["scene_model_path"] == paths["scene"]
    assert calls[0]["pose_model_path"] == paths["pose"]
    assert calls[0]["out"] != out
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["demo"] == "memory"
    assert set(report) == {
        "artifacts",
        "checks",
        "demo",
        "demo_items",
        "familiar_unknown",
        "models",
        "real_model_evidence",
        "scene_count",
        "status",
    }
    assert report["real_model_evidence"] is True
    assert "gate" not in report
    assert "mode" not in report
    assert report["models"] == {
        "pose": str(paths["pose"]),
        "face": str(paths["face"]),
        "scene": str(paths["scene"]),
    }
    assert "fake" not in json.dumps(report["models"], ensure_ascii=False).lower()
    assert (out / "index.html").is_file()
    root_html = (out / "index.html").read_text(encoding="utf-8")
    assert "Memory Demo" in root_html
    assert "real_model_evidence" in root_html
    assert "source-artifact" not in root_html
    assert "visual_evidence_index.json" not in root_html
    assert {path.name for path in out.iterdir()} == {
        "index.html",
        "report.json",
        "visual-evidence",
    }
    assert not (out / "visual-evidence" / "crops").exists()
    public_file_names = [path.name for path in out.rglob("*") if path.is_file()]
    assert all("person_" not in name for name in public_file_names)
    assert all(re.search(r"[0-9a-f]{12,}", name) is None for name in public_file_names)


def test_public_output_does_not_keep_internal_runner_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    out = tmp_path / "artifacts" / "demo" / "memory"

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        internal_out = kwargs["out"]
        internal_out.mkdir(parents=True, exist_ok=True)
        for name in (
            "timeline.jsonl",
            "teach_payloads.json",
            "api_responses.jsonl",
            "botified_frames.jsonl",
            "visual_states.jsonl",
        ):
            (internal_out / name).write_text("{}\n", encoding="utf-8")
        runtime_db = internal_out / "runtime" / "memory.sqlite3"
        runtime_db.parent.mkdir(parents=True)
        runtime_db.write_bytes(b"sqlite")
        cache_marker = internal_out / ".cache" / "inference" / "marker.txt"
        cache_marker.parent.mkdir(parents=True)
        cache_marker.write_text("cache", encoding="utf-8")
        return _write_minimal_runner_report(internal_out)

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(data_dir), "--out", str(out)]) == 0

    public_paths = {path.relative_to(out).as_posix() for path in out.rglob("*")}
    assert "index.html" in public_paths
    assert "report.json" in public_paths
    assert "visual-evidence/index.html" in public_paths
    assert "visual-evidence/crops" not in public_paths
    for internal_path in (
        "timeline.jsonl",
        "teach_payloads.json",
        "api_responses.jsonl",
        "botified_frames.jsonl",
        "visual_states.jsonl",
        "runtime/memory.sqlite3",
        ".cache/inference/marker.txt",
    ):
        assert internal_path not in public_paths


def test_runner_wrapper_normalizes_real_model_demo_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_local_smoke(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        out = kwargs["out"]
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "ok": True,
            "status": "passed",
            "gate": "memory_teaching_ga_runner_local_smoke",
            "mode": "local-smoke",
            "backend": "local",
            "real_model_evidence": True,
            "self_smoke": {"status": "passed"},
            "scene_smoke": {"status": "passed"},
            "third_person_probe": {"status": "passed"},
            "familiar_unknown": _familiar_unknown_result(),
            "checks": [
                {"name": "local_smoke_explicit_real_backends", "passed": True},
                {"name": "self_local_smoke", "passed": True},
                {"name": "scene_local_smoke", "passed": True},
                {"name": "third_person_local_probe", "passed": True},
                {"name": "familiar_unknown_local_demo", "passed": True},
                {"name": "artifact_skeleton", "passed": True},
            ],
            "visual_evidence_index": [
                {"assertion_id": "memory_teaching_ga_local_smoke"}
            ],
        }
        (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (out / "teach_payloads.json").write_text(
            json.dumps({"schema_version": 1, "mode": "local-smoke"}),
            encoding="utf-8",
        )
        return report

    with module._suppress_known_starlette_testclient_warning():
        from tools import run_memory_teaching_ga_e2e as runner

    monkeypatch.setattr(runner, "run_local_smoke", fake_run_local_smoke)
    model_paths = _make_model_paths(tmp_path)
    out = tmp_path / "out"

    report = runner.run_real_model_memory_demo(
        data_dir=tmp_path / "val-data",
        out=out,
        camera="front",
        person_model_path=model_paths["face"],
        scene_model_path=model_paths["scene"],
        pose_model_path=model_paths["pose"],
    )

    assert calls[0]["embedding_backend"] == "local"
    assert calls[0]["inference_backend"] == "ultralytics"
    assert calls[0]["case_names"] == runner.REAL_MODEL_MEMORY_DEMO_CASES
    assert calls[0]["include_familiar_unknown"] is True
    assert calls[0]["familiar_unknown_scene"] == runner.FAMILIAR_UNKNOWN_SCENE
    assert report["gate"] == "memory_demo_real_model"
    assert report["mode"] == "memory-demo"
    assert report["embedding_backend"] == "local"
    assert report["inference_backend"] == "ultralytics"
    assert report["real_model_evidence"] is True
    assert report["models"] == {
        "pose": str(model_paths["pose"]),
        "face": str(model_paths["face"]),
        "scene": str(model_paths["scene"]),
    }
    assert "self_smoke" not in report
    assert "scene_smoke" not in report
    assert "third_person_probe" not in report
    assert report["familiar_unknown"]["status"] == "passed"
    assert {"self_introduction", "teach_scene", "third_person_introduction"} <= set(report)
    assert [check["name"] for check in report["checks"]] == [
        "real_model_paths",
        "self_introduction_known_person_present",
        "teach_scene_scene_activated",
        "third_person_pose_pointing_known_person",
        "familiar_unknown_present",
        "demo_outputs",
    ]
    payloads = json.loads((out / "teach_payloads.json").read_text(encoding="utf-8"))
    assert payloads["mode"] == "memory-demo"
    assert payloads["backend"] == "local"


def test_runner_wrapper_does_not_mark_failed_local_run_as_real_model_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_local_smoke(**kwargs: Any) -> dict[str, Any]:
        out = kwargs["out"]
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "ok": False,
            "status": "failed",
            "gate": "memory_teaching_ga_runner_local_smoke",
            "mode": "local-smoke",
            "backend": "local",
            "real_model_evidence": True,
            "checks": [{"name": "local_smoke_explicit_real_backends", "passed": False}],
        }
        (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    with module._suppress_known_starlette_testclient_warning():
        from tools import run_memory_teaching_ga_e2e as runner

    monkeypatch.setattr(runner, "run_local_smoke", fake_run_local_smoke)
    model_paths = _make_model_paths(tmp_path)

    report = runner.run_real_model_memory_demo(
        data_dir=tmp_path / "val-data",
        out=tmp_path / "out",
        camera="front",
        person_model_path=model_paths["face"],
        scene_model_path=model_paths["scene"],
        pose_model_path=model_paths["pose"],
    )

    assert report["ok"] is False
    assert report["real_model_evidence"] is False


def test_missing_data_dir_fails_before_cleaning_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    missing_data_dir = tmp_path / "missing-val-data"
    out = tmp_path / "artifacts" / "demo" / "memory"
    stale_index = out / "index.html"
    stale_index.parent.mkdir(parents=True)
    stale_index.write_text("keep", encoding="utf-8")
    called = False

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(missing_data_dir), "--out", str(out)]) == 2

    assert called is False
    assert stale_index.read_text(encoding="utf-8") == "keep"
    assert "data-dir does not exist" in capsys.readouterr().err


def test_file_data_dir_fails_before_cleaning_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.write_text("not a dir", encoding="utf-8")
    out = tmp_path / "artifacts" / "demo" / "memory"
    stale_report = out / "report.json"
    stale_report.parent.mkdir(parents=True)
    stale_report.write_text("keep", encoding="utf-8")

    assert module.main(["--data-dir", str(data_dir), "--out", str(out)]) == 2

    assert stale_report.read_text(encoding="utf-8") == "keep"
    assert "data-dir is not a directory" in capsys.readouterr().err


def test_missing_model_paths_fail_fast_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_paths = {
        "face": tmp_path / "runtime" / "models" / "face-buffalo-s",
        "scene": tmp_path / "runtime" / "models" / "scene-mobileclip2-s0",
        "pose": tmp_path / "runtime" / "models" / "yolov8n-pose.pt",
    }
    _patch_default_models(monkeypatch, missing_paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    out = tmp_path / "artifacts" / "demo" / "memory"
    called = False

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(data_dir), "--out", str(out)]) == 2

    assert called is False
    stderr = capsys.readouterr().err
    assert "missing required real model path" in stderr
    for path in missing_paths.values():
        assert str(path) in stderr
    assert not (out / "index.html").exists()


def test_output_cleanup_does_not_clear_parent_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    artifacts = tmp_path / "artifacts"
    sibling = artifacts / "visual" / "index.html"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("keep", encoding="utf-8")
    out = artifacts / "demo" / "memory"
    (out / "visual-evidence").mkdir(parents=True)
    (out / "visual-evidence" / "stale.html").write_text("stale", encoding="utf-8")
    (out / "unmanaged-note.txt").write_text("keep", encoding="utf-8")
    cache_marker = out / ".cache" / "inference" / "yolo" / "Ultralytics" / "marker.txt"
    cache_marker.parent.mkdir(parents=True)
    cache_marker.write_text("keep-cache", encoding="utf-8")

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["out"] != out
        assert not (out / "visual-evidence" / "stale.html").exists()
        assert not (out / "unmanaged-note.txt").exists()
        assert not (out / ".cache").exists()
        return _write_minimal_runner_report(kwargs["out"])

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(data_dir), "--out", str(out)]) == 0

    assert sibling.read_text(encoding="utf-8") == "keep"
    assert not (out / "unmanaged-note.txt").exists()
    assert not cache_marker.exists()


def test_out_repo_root_is_rejected_before_cleaning_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    runtime_marker = tmp_path / "runtime" / "keep.txt"
    runtime_marker.write_text("keep-runtime", encoding="utf-8")
    called = False

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(data_dir), "--out", "."]) == 2

    assert called is False
    assert runtime_marker.read_text(encoding="utf-8") == "keep-runtime"
    assert "artifacts/demo" in capsys.readouterr().err


def test_out_runtime_is_rejected_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    runtime_marker = tmp_path / "runtime" / "keep.txt"
    runtime_marker.write_text("keep-runtime", encoding="utf-8")

    assert module.main(
        ["--data-dir", str(data_dir), "--out", "runtime/memory-demo"]
    ) == 2

    assert runtime_marker.read_text(encoding="utf-8") == "keep-runtime"
    assert not (tmp_path / "runtime" / "memory-demo").exists()
    assert "artifacts/demo" in capsys.readouterr().err


def test_parent_symlink_in_public_out_is_rejected_before_cleaning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    target_marker = symlink_target / "keep.txt"
    target_marker.write_text("keep-target", encoding="utf-8")
    (artifacts / "demo").symlink_to(symlink_target, target_is_directory=True)
    called = False

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert (
        module.main(
            [
                "--data-dir",
                str(data_dir),
                "--out",
                str(tmp_path / "artifacts" / "demo" / "memory"),
            ]
        )
        == 2
    )

    assert called is False
    assert "symlink" in capsys.readouterr().err
    assert target_marker.read_text(encoding="utf-8") == "keep-target"
    assert not (symlink_target / "memory").exists()


def test_default_output_uses_public_demo_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    calls: list[Path] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["out"])
        return _write_minimal_runner_report(kwargs["out"])

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(data_dir)]) == 0

    expected = tmp_path / "artifacts" / "demo" / "memory"
    assert len(calls) == 1
    assert calls[0] != expected
    assert calls[0].name == "artifact"
    assert (expected / "report.json").is_file()


def test_output_cleanup_removes_managed_symlink_without_touching_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    data_dir.mkdir()
    out = tmp_path / "artifacts" / "demo" / "memory"
    target = tmp_path / "external-visual-evidence"
    target.mkdir()
    target_marker = target / "keep.txt"
    target_marker.write_text("keep-target", encoding="utf-8")
    out.mkdir(parents=True)
    (out / "visual-evidence").symlink_to(target, target_is_directory=True)

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["out"] != out
        assert not (out / "visual-evidence").exists()
        assert target_marker.read_text(encoding="utf-8") == "keep-target"
        return _write_minimal_runner_report(kwargs["out"])

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(data_dir), "--out", str(out)]) == 0

    assert target_marker.read_text(encoding="utf-8") == "keep-target"
    assert (out / "visual-evidence").is_dir()
    assert not (out / "visual-evidence").is_symlink()


def test_public_demo_pages_cover_familiar_unknown_without_legacy_main_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_model_paths(tmp_path)
    _patch_default_models(monkeypatch, paths)
    data_dir = tmp_path / "val-data"
    (data_dir / "pic_familiar_face").mkdir(parents=True)
    (data_dir / "pic_familiar_face" / "img_000.jpeg").write_bytes(b"jpeg")
    (data_dir / "pic_teach_me").mkdir()
    (data_dir / "pic_teach_me" / "img_000.jpg").write_bytes(b"jpeg")
    out = tmp_path / "artifacts" / "demo" / "memory"

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        return _write_minimal_runner_report(
            kwargs["out"],
            scene_count=None,
            familiar_unknown=_familiar_unknown_result(),
            checks=[
                {"name": "real_model_paths", "passed": True},
                {"name": "familiar_unknown_present", "passed": True},
                {"name": "demo_outputs", "passed": True},
            ],
        )

    _patch_runtime_modules(
        monkeypatch,
        SimpleNamespace(run_real_model_memory_demo=fake_runner),
    )

    assert module.main(["--data-dir", str(data_dir), "--out", str(out)]) == 0

    root_html = (out / "index.html").read_text(encoding="utf-8")
    visual_html = (out / "visual-evidence" / "index.html").read_text(encoding="utf-8")
    report_text = (out / "report.json").read_text(encoding="utf-8")
    public_text = "\n".join([root_html, visual_html, report_text])
    report = json.loads(report_text)

    assert "Memory Demo" in root_html
    assert "Demo Summary" in root_html
    assert "Models" in root_html
    assert "Demo Items" in root_html
    assert report["demo"] == "memory"
    assert report["real_model_evidence"] is True
    assert report["scene_count"] == 2
    assert isinstance(report["scene_count"], int)
    assert report["scene_count"] > 0
    assert report["models"] == {
        "pose": str(paths["pose"]),
        "face": str(paths["face"]),
        "scene": str(paths["scene"]),
    }
    assert report["familiar_unknown"]["status"] == "passed"
    assert report["demo_items"][-1]["id"] == "familiar_unknown"
    assert report["demo_items"][-1]["status"] == "passed"
    assert "匿名熟客出现" in public_text
    assert report["checks"] == [
        {"name": "real_models_loaded", "passed": True},
        {"name": "scenes_loaded", "passed": True},
        {"name": "familiar_unknown_detected", "passed": True},
        {"name": "outputs_written", "passed": True},
    ]
    assert set(report) == {
        "artifacts",
        "checks",
        "demo",
        "demo_items",
        "familiar_unknown",
        "models",
        "real_model_evidence",
        "scene_count",
        "status",
    }
    for legacy_text in [
        "Memory Teaching Evidence",
        "Source Report",
        "source gate",
        "gate",
        "local-smoke",
        "source-artifact",
        "artifact_skeleton",
        "crop_path_or_artifact_ref",
    ]:
        assert legacy_text not in public_text
    for forbidden_text in PUBLIC_FORBIDDEN_TEXT:
        assert forbidden_text not in public_text
    assert not (out / "visual_evidence_index.json").exists()
    assert {path.name for path in out.iterdir()} == {
        "index.html",
        "report.json",
        "visual-evidence",
    }


def test_public_inference_cache_env_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "out"
    previous = tmp_path / "previous-cache"
    monkeypatch.setenv(module.PUBLIC_INFERENCE_CACHE_ENV, str(previous))

    with module._public_inference_cache_env(out):
        expected = out / module.PUBLIC_INFERENCE_CACHE_RELATIVE_PATH
        assert os.environ[module.PUBLIC_INFERENCE_CACHE_ENV] == str(expected)
        settings_path = expected / "yolo" / "Ultralytics" / "settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings["settings_version"] == module.ULTRALYTICS_SETTINGS_VERSION
        assert set(settings) == set(module._ultralytics_settings_defaults(Path.cwd()))

    assert os.environ[module.PUBLIC_INFERENCE_CACHE_ENV] == str(previous)


def test_inference_factory_uses_public_demo_cache_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visual_events_server.inference.factory import configure_inference_cache

    stable_cache = tmp_path / "stable-inference-cache"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv(module.PUBLIC_INFERENCE_CACHE_ENV, str(stable_cache))

    configure_inference_cache(runtime_dir)

    assert Path(os.environ["YOLO_CONFIG_DIR"]) == stable_cache / "yolo"
    assert Path(os.environ["TORCH_HOME"]) == stable_cache / "torch"
    assert Path(os.environ["XDG_CACHE_HOME"]) == stable_cache / "xdg"
    assert Path(os.environ["MPLCONFIGDIR"]) == stable_cache / "matplotlib"
