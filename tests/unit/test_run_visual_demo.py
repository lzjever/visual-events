from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import generate_visual_evidence as visual_evidence
from tools import run_visual_demo as module


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _public_demo_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)


class FakeImage:
    def __init__(self, source: Path) -> None:
        self.source = source


class FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if self.returncode is None:
            self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class FakeRunner:
    def __init__(self, *, health: module.HealthCheckResult | None = None) -> None:
        self.process = FakeProcess()
        self.health = health or module.HealthCheckResult(
            passed=True,
            healthz_pid=self.process.pid,
            healthz_identity_verified=True,
        )
        self.started: list[dict[str, Any]] = []
        self.health_checks: list[dict[str, Any]] = []
        self.stopped: list[FakeProcess] = []

    def start_server(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> FakeProcess:
        self.started.append({"command": command, "cwd": cwd, "env": env})
        return self.process

    def wait_healthz(
        self,
        url: str,
        process: FakeProcess,
        *,
        timeout_s: float,
        interval_s: float,
    ) -> module.HealthCheckResult:
        self.health_checks.append(
            {
                "url": url,
                "process": process,
                "timeout_s": timeout_s,
                "interval_s": interval_s,
            }
        )
        return self.health

    def stop_server(self, process: FakeProcess) -> None:
        self.stopped.append(process)
        process.terminate()


def _make_scene(data_dir: Path, scene: str, frame_count: int) -> None:
    scene_dir = data_dir / scene
    scene_dir.mkdir(parents=True)
    for frame_id in range(frame_count):
        (scene_dir / f"frame_{frame_id:03d}.jpg").write_bytes(b"jpeg")


def _state(frame_id: int, *, person_count: int = 0) -> dict[str, Any]:
    return {
        "type": "visual_state",
        "camera": "front",
        "frame_id": frame_id,
        "scene_flags": {
            "has_person": person_count > 0,
            "person_count": person_count,
        },
        "tracks": [],
        "attention": None,
        "semantic_events": [],
    }


def _wrapped(scene: str, frame_id: int, response: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": scene,
        "frame_id": frame_id,
        "latency_ms": 10.0 + frame_id,
        "response": response,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _patch_image_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_decode(path: Path) -> FakeImage:
        return FakeImage(path)

    def fake_draw(
        image: FakeImage,
        response: dict[str, Any],
        *,
        scene: str,
        frame_id: int,
        public_labels: bool = False,
    ) -> dict[str, Any]:
        del response
        return {
            "source": image.source,
            "scene": scene,
            "frame_id": frame_id,
            "public_labels": public_labels,
        }

    def fake_write(path: Path, image: Any) -> None:
        del image
        path.write_bytes(b"annotated")

    monkeypatch.setattr(visual_evidence, "_decode_image", fake_decode)
    monkeypatch.setattr(visual_evidence, "draw_visual_state", fake_draw)
    monkeypatch.setattr(visual_evidence, "_write_jpeg", fake_write)


def test_direct_script_help_exposes_only_visual_demo_public_args() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_visual_demo.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run the real-model visual demo" in result.stdout
    for flag in ("--data-dir", "--out", "--camera", "--fps", "--no-realtime"):
        assert flag in result.stdout
    for flag in (
        "--server",
        "--visual-state-jsonl",
        "--replay-artifact",
        "--run-replay",
        "--head-motion",
        "--host",
        "--port",
    ):
        assert flag not in result.stdout


def test_parse_args_defaults_to_real_visual_demo_contract() -> None:
    args = module.parse_args(["--data-dir", "val-data"])

    assert args.data_dir == Path("val-data")
    assert args.out == Path("artifacts/demo/visual")
    assert args.camera == "front"
    assert args.fps == 10.0
    assert args.no_realtime is False
    assert not hasattr(args, "host")
    assert not hasattr(args, "port")


async def test_missing_pose_model_fails_before_starting_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _make_scene(data_dir, "lobby", 1)
    missing_model = tmp_path / "runtime" / "models" / "yolov8n-pose.pt"
    runner = FakeRunner()
    monkeypatch.setattr(module, "POSE_MODEL_PATH", missing_model)

    with pytest.raises(SystemExit) as exc:
        await module.run_visual_demo_from_args(
            argparse.Namespace(
                data_dir=data_dir,
                out=tmp_path / "artifacts" / "demo" / "visual",
                camera="front",
                fps=10.0,
                no_realtime=False,
            ),
            runner=runner,
        )

    assert "required real model is missing" in str(exc.value)
    assert str(missing_model) in str(exc.value)
    assert runner.started == []


async def test_visual_demo_starts_temp_server_replays_renders_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    out = tmp_path / "artifacts" / "demo" / "visual"
    model_path = tmp_path / "runtime" / "models" / "yolov8n-pose.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    _make_scene(data_dir, "lobby", 2)
    _patch_image_io(monkeypatch)
    monkeypatch.setattr(module, "POSE_MODEL_PATH", model_path)
    monkeypatch.setattr(module, "_choose_available_loopback_port", lambda: 9876)
    replay_calls: list[dict[str, Any]] = []

    async def fake_replay_data_dir(**kwargs: Any) -> list[Any]:
        replay_calls.append(kwargs)
        _write_jsonl(
            kwargs["save_jsonl"],
            [
                _wrapped("lobby", 0, _state(0, person_count=1)),
                _wrapped("lobby", 1, {"type": "error", "code": "response_timeout"}),
            ],
        )
        return []

    monkeypatch.setattr(module, "replay_data_dir", fake_replay_data_dir)
    runner = FakeRunner()

    report = await module.run_visual_demo_from_args(
        argparse.Namespace(
            data_dir=data_dir,
            out=out,
            camera="front",
            fps=10.0,
            no_realtime=True,
        ),
        runner=runner,
    )

    assert runner.started == [
        {
            "command": [
                "visual-events-server",
                "--config",
                "configs/pc-ga-server.toml",
                "--host",
                "127.0.0.1",
                "--port",
                "9876",
            ],
            "cwd": module.REPO_ROOT,
            "env": runner.started[0]["env"],
        }
    ]
    assert runner.health_checks[0]["url"] == "http://127.0.0.1:9876/healthz"
    assert runner.stopped == [runner.process]
    assert replay_calls == [
        {
            "server": "ws://127.0.0.1:9876/v1/stream",
            "data_dir": data_dir,
            "camera": "front",
            "fps": 10.0,
            "head_motion": "stationary",
            "save_jsonl": replay_calls[0]["save_jsonl"],
            "realtime": False,
            "response_timeout_ms": None,
            "continue_on_timeout": True,
        }
    ]
    assert replay_calls[0]["save_jsonl"].name == "visual_state.jsonl"
    assert out not in replay_calls[0]["save_jsonl"].parents

    assert (out / "index.html").is_file()
    assert (out / "scenes" / "lobby" / "index.html").is_file()
    assert not (out / "visual_state.jsonl").exists()
    assert not (out / "scenes" / "lobby" / "visual_state.jsonl").exists()
    assert not (out / "scenes" / "lobby" / "states").exists()
    report_json = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report_json == report
    assert report["real_model_evidence"] is True
    assert report["demo"] == "visual"
    assert report["status"] == "completed"
    assert report["inference_backend"] == "ultralytics"
    assert report["models"] == {"pose": str(model_path)}
    assert report["scene_count"] == 1
    assert report["frame_count"] == 2
    assert report["error_count"] == 1
    assert report["person_frame_count"] == 1
    assert report["event_count"] == 0
    assert report["artifacts"] == {
        "index_html": "index.html",
        "summary_json": "summary.json",
        "scenes": "scenes/",
    }
    assert set(report) == {
        "artifacts",
        "demo",
        "error_count",
        "event_count",
        "frame_count",
        "inference_backend",
        "models",
        "person_frame_count",
        "real_model_evidence",
        "scene_count",
        "status",
    }
    forbidden_report_keys = {
        "server_command",
        "server_url",
        "server_pid",
        "config_path",
        "out",
        "index_path",
        "summary_path",
    }
    assert forbidden_report_keys.isdisjoint(report)
    root_html = (out / "index.html").read_text(encoding="utf-8")
    assert "data-visual-demo-summary=\"true\"" in root_html
    assert "real_model_evidence=true" in root_html
    assert "pose runtime: ultralytics" in root_html
    assert f"pose model: {model_path}" in root_html
    assert "person_frames: 1" in root_html
    assert root_html.index("real_model_evidence=true") < root_html.index("<h2>Scenes</h2>")


async def test_out_repo_root_is_rejected_before_cleaning_runtime(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runtime_marker = tmp_path / "runtime" / "keep.txt"
    runtime_marker.parent.mkdir()
    runtime_marker.write_text("keep-runtime", encoding="utf-8")
    runner = FakeRunner()

    with pytest.raises(SystemExit) as exc:
        await module.run_visual_demo_from_args(
            argparse.Namespace(
                data_dir=data_dir,
                out=Path("."),
                camera="front",
                fps=10.0,
                no_realtime=True,
            ),
            runner=runner,
        )

    assert "artifacts/demo" in str(exc.value)
    assert runner.started == []
    assert runtime_marker.read_text(encoding="utf-8") == "keep-runtime"


async def test_out_runtime_is_rejected_before_creating_output(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runtime_marker = tmp_path / "runtime" / "keep.txt"
    runtime_marker.parent.mkdir()
    runtime_marker.write_text("keep-runtime", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        await module.run_visual_demo_from_args(
            argparse.Namespace(
                data_dir=data_dir,
                out=Path("runtime/visual-demo"),
                camera="front",
                fps=10.0,
                no_realtime=True,
            ),
            runner=FakeRunner(),
        )

    assert "artifacts/demo" in str(exc.value)
    assert runtime_marker.read_text(encoding="utf-8") == "keep-runtime"
    assert not (tmp_path / "runtime" / "visual-demo").exists()


async def test_parent_symlink_in_public_out_is_rejected_before_cleaning(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    target_marker = symlink_target / "keep.txt"
    target_marker.write_text("keep-target", encoding="utf-8")
    (artifacts / "demo").symlink_to(symlink_target, target_is_directory=True)
    runner = FakeRunner()

    with pytest.raises(SystemExit) as exc:
        await module.run_visual_demo_from_args(
            argparse.Namespace(
                data_dir=data_dir,
                out=tmp_path / "artifacts" / "demo" / "visual",
                camera="front",
                fps=10.0,
                no_realtime=True,
            ),
            runner=runner,
        )

    assert "symlink" in str(exc.value)
    assert runner.started == []
    assert target_marker.read_text(encoding="utf-8") == "keep-target"
    assert not (symlink_target / "visual").exists()


async def test_default_output_cleanup_does_not_clear_artifacts_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    default_out = artifacts / "demo" / "visual"
    model_path = tmp_path / "runtime" / "models" / "yolov8n-pose.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    _make_scene(data_dir, "lobby", 1)
    (default_out / "scenes" / "stale").mkdir(parents=True)
    (default_out / "scenes" / "stale" / "old.txt").write_text("old", encoding="utf-8")
    (artifacts / "keep.txt").parent.mkdir(parents=True, exist_ok=True)
    (artifacts / "keep.txt").write_text("keep", encoding="utf-8")
    _patch_image_io(monkeypatch)
    monkeypatch.setattr(module, "POSE_MODEL_PATH", model_path)
    monkeypatch.setattr(module, "DEFAULT_OUT", default_out)

    async def fake_replay_data_dir(**kwargs: Any) -> list[Any]:
        _write_jsonl(kwargs["save_jsonl"], [_wrapped("lobby", 0, _state(0))])
        return []

    monkeypatch.setattr(module, "replay_data_dir", fake_replay_data_dir)

    await module.run_visual_demo_from_args(
        module.parse_args(["--data-dir", str(data_dir)]),
        runner=FakeRunner(),
    )

    assert not (default_out / "scenes" / "stale" / "old.txt").exists()
    assert (artifacts / "keep.txt").read_text(encoding="utf-8") == "keep"


async def test_output_cleanup_removes_managed_symlink_without_touching_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    out = tmp_path / "artifacts" / "demo" / "visual"
    model_path = tmp_path / "runtime" / "models" / "yolov8n-pose.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    _make_scene(data_dir, "lobby", 1)
    target = tmp_path / "external-scenes"
    target.mkdir()
    target_marker = target / "keep.txt"
    target_marker.write_text("keep-target", encoding="utf-8")
    out.mkdir(parents=True)
    (out / "scenes").symlink_to(target, target_is_directory=True)
    _patch_image_io(monkeypatch)
    monkeypatch.setattr(module, "POSE_MODEL_PATH", model_path)

    async def fake_replay_data_dir(**kwargs: Any) -> list[Any]:
        assert not (out / "scenes").exists()
        assert target_marker.read_text(encoding="utf-8") == "keep-target"
        _write_jsonl(kwargs["save_jsonl"], [_wrapped("lobby", 0, _state(0))])
        return []

    monkeypatch.setattr(module, "replay_data_dir", fake_replay_data_dir)

    await module.run_visual_demo_from_args(
        argparse.Namespace(
            data_dir=data_dir,
            out=out,
            camera="front",
            fps=10.0,
            no_realtime=True,
        ),
        runner=FakeRunner(),
    )

    assert target_marker.read_text(encoding="utf-8") == "keep-target"
    assert (out / "scenes").is_dir()
    assert not (out / "scenes").is_symlink()
