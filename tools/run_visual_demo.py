from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import generate_visual_evidence as visual_evidence
from tools.public_demo_output import PublicDemoOutputError, resolve_public_demo_out
from tools.replay_val_data import replay_data_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("artifacts/demo/visual")
DEFAULT_CONFIG = Path("configs/pc-ga-server.toml")
POSE_MODEL_PATH = Path("runtime/models/yolov8n-pose.pt")
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_CAMERA = "front"
DEFAULT_FPS = 10.0
DEFAULT_HEALTH_TIMEOUT_S = 30.0
DEFAULT_HEALTH_INTERVAL_S = 0.2
SERVER_STOP_TIMEOUT_S = 5.0
HEAD_MOTION = "stationary"

@dataclass(frozen=True)
class HealthCheckResult:
    passed: bool
    failure_reason: str | None = None
    healthz_pid: int | None = None
    healthz_identity_verified: bool = False


@dataclass(frozen=True)
class HealthzResponse:
    status: int
    payload: dict[str, object] | None
    failure_reason: str | None = None


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None:
        ...

    def terminate(self) -> None:
        ...

    def wait(self, timeout: float | None = None) -> int:
        ...

    def kill(self) -> None:
        ...


class VisualDemoRunner(Protocol):
    def start_server(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> ProcessLike:
        ...

    def wait_healthz(
        self,
        url: str,
        process: ProcessLike,
        *,
        timeout_s: float,
        interval_s: float,
    ) -> HealthCheckResult:
        ...

    def stop_server(self, process: ProcessLike) -> None:
        ...


class LocalVisualDemoRunner:
    def start_server(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> ProcessLike:
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_healthz(
        self,
        url: str,
        process: ProcessLike,
        *,
        timeout_s: float,
        interval_s: float,
    ) -> HealthCheckResult:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return HealthCheckResult(False, "server_exited_early")

            response = _request_healthz(url, timeout_s=interval_s)
            if response is None:
                time.sleep(interval_s)
                continue

            healthz_pid = _healthz_pid(response.payload)
            if response.status != 200 or response.payload is None:
                return HealthCheckResult(False, "healthz_unhealthy", healthz_pid)
            if response.payload.get("ok") is not True:
                return HealthCheckResult(False, "healthz_unhealthy", healthz_pid)
            if healthz_pid != process.pid:
                return HealthCheckResult(
                    False,
                    "healthz_identity_mismatch",
                    healthz_pid,
                )
            if process.poll() is not None:
                return HealthCheckResult(
                    False,
                    "server_exited_early",
                    healthz_pid,
                    True,
                )
            return HealthCheckResult(
                True,
                healthz_pid=healthz_pid,
                healthz_identity_verified=True,
            )

        if process.poll() is not None:
            return HealthCheckResult(False, "server_exited_early")
        return HealthCheckResult(False, "healthz_timeout")

    def stop_server(self, process: ProcessLike) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=SERVER_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=SERVER_STOP_TIMEOUT_S)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real-model visual demo against val-data."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send replay frames as fast as responses arrive.",
    )
    return parser.parse_args(argv)


async def async_main(
    argv: list[str] | None = None,
    *,
    runner: VisualDemoRunner | None = None,
) -> int:
    args = parse_args(argv)
    await run_visual_demo_from_args(args, runner=runner)
    print(Path(args.out) / "index.html")
    return 0


async def run_visual_demo_from_args(
    args: argparse.Namespace,
    *,
    runner: VisualDemoRunner | None = None,
) -> dict[str, Any]:
    args.out = _resolve_public_demo_out(Path(args.out))
    _preflight(args)
    _prepare_output_dir(Path(args.out))

    active_runner = runner or LocalVisualDemoRunner()
    server_process: ProcessLike | None = None
    health_result: HealthCheckResult | None = None
    host = LOOPBACK_HOST
    port = _choose_available_loopback_port()
    server_command = _server_command(host=host, port=port)
    server_url = _server_url(host=host, port=port)

    try:
        server_process = active_runner.start_server(
            server_command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
        )
        health_result = active_runner.wait_healthz(
            _healthz_url(host=host, port=port),
            server_process,
            timeout_s=DEFAULT_HEALTH_TIMEOUT_S,
            interval_s=DEFAULT_HEALTH_INTERVAL_S,
        )
        if not health_result.passed:
            raise SystemExit(
                "visual-events-server health check failed: "
                f"{health_result.failure_reason or 'unknown'}"
            )

        with tempfile.TemporaryDirectory(prefix="visual-demo-replay-") as replay_dir:
            jsonl_path = Path(replay_dir) / "visual_state.jsonl"
            await replay_data_dir(
                server=server_url,
                data_dir=Path(args.data_dir),
                camera=str(args.camera),
                fps=float(args.fps),
                head_motion=HEAD_MOTION,
                save_jsonl=jsonl_path,
                realtime=not bool(args.no_realtime),
                response_timeout_ms=None,
                continue_on_timeout=True,
            )
            records = visual_evidence.read_wrapped_visual_state_jsonl(jsonl_path)
            source_images = visual_evidence.map_source_images(
                Path(args.data_dir),
                records,
                camera=str(args.camera),
                fps=float(args.fps),
                head_motion=HEAD_MOTION,
            )
            summary = visual_evidence.generate_visual_evidence(
                records=records,
                source_images=source_images,
                out=Path(args.out),
                input_jsonl=jsonl_path,
                public_demo=True,
            )
        report = _build_report(
            args,
            summary=summary,
        )
        _write_json(Path(args.out) / "report.json", report)
        _inject_public_demo_summary(Path(args.out) / "index.html", report)
        return report
    finally:
        if server_process is not None:
            active_runner.stop_server(server_process)


def _preflight(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    if not data_dir.exists():
        raise SystemExit(f"data-dir does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"data-dir is not a directory: {data_dir}")
    if out.resolve(strict=False) == data_dir.resolve() or out.resolve(
        strict=False
    ).is_relative_to(data_dir.resolve()):
        raise SystemExit("--out must not be inside --data-dir")
    if float(args.fps) <= 0.0:
        raise SystemExit("fps must be positive")
    _require_file(DEFAULT_CONFIG, label="server config")
    _require_file(POSE_MODEL_PATH, label="required real model")


def _resolve_public_demo_out(out: Path) -> Path:
    try:
        return resolve_public_demo_out(out, repo_root=REPO_ROOT)
    except PublicDemoOutputError as exc:
        raise SystemExit(str(exc)) from exc


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} is missing: {path}")


def _prepare_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        _remove_path(child)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _server_command(*, host: str, port: int) -> list[str]:
    return [
        "visual-events-server",
        "--config",
        os.fspath(DEFAULT_CONFIG),
        "--host",
        str(host),
        "--port",
        str(port),
    ]


def _choose_available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK_HOST, 0))
        return int(sock.getsockname()[1])


def _server_url(*, host: str, port: int) -> str:
    return f"ws://{host}:{port}/v1/stream"


def _healthz_url(*, host: str, port: int) -> str:
    return f"http://{host}:{port}/healthz"


def _build_report(
    args: argparse.Namespace,
    *,
    summary: dict[str, Any],
) -> dict[str, Any]:
    del args
    person = summary.get("person")
    semantic_events = summary.get("semantic_events")
    return {
        "demo": "visual",
        "status": "completed",
        "real_model_evidence": True,
        "inference_backend": "ultralytics",
        "models": {
            "pose": os.fspath(POSE_MODEL_PATH),
        },
        "scene_count": len(summary.get("scenes", {})),
        "frame_count": int(summary.get("frames_total", 0)),
        "error_count": int(summary.get("errors", 0)),
        "person_frame_count": int(
            person.get("frames_with_person", 0) if isinstance(person, dict) else 0
        ),
        "event_count": int(
            semantic_events.get("total", 0)
            if isinstance(semantic_events, dict)
            else 0
        ),
        "artifacts": {
            "index_html": "index.html",
            "summary_json": "summary.json",
            "scenes": "scenes/",
        },
    }


def _inject_public_demo_summary(index_path: Path, report: dict[str, Any]) -> None:
    if not index_path.is_file():
        raise SystemExit(f"visual demo index.html not found: {index_path}")
    document = index_path.read_text(encoding="utf-8")
    if 'data-visual-demo-summary="true"' in document:
        return

    block = _public_demo_summary_html(report)
    marker = "  <h1>visual evidence</h1>"
    if marker in document:
        updated = document.replace(marker, marker + "\n" + block, 1)
    else:
        updated = document.replace("<body>", "<body>\n" + block, 1)
    index_path.write_text(updated, encoding="utf-8")


def _public_demo_summary_html(report: dict[str, Any]) -> str:
    models = report.get("models")
    pose_model = models.get("pose") if isinstance(models, dict) else None
    return f"""  <section class="demo-summary" data-visual-demo-summary="true">
    <h2>Real Model Demo</h2>
    <p class="meta">
      real_model_evidence={html.escape(str(report.get("real_model_evidence")).lower())}<br>
      pose runtime: {html.escape(str(report.get("inference_backend", "-")))}<br>
      pose model: {html.escape(str(pose_model or "-"))}<br>
      frames: {html.escape(str(report.get("frame_count", "-")))} |
      person_frames: {html.escape(str(report.get("person_frame_count", "-")))} |
      events: {html.escape(str(report.get("event_count", "-")))} |
      errors: {html.escape(str(report.get("error_count", "-")))}
    </p>
  </section>"""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _request_healthz(url: str, *, timeout_s: float) -> HealthzResponse | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read()
            status = int(response.status)
    except (OSError, TimeoutError, urllib.error.URLError):
        return None

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return HealthzResponse(status, None, "healthz_invalid_json")
    if not isinstance(payload, dict):
        return HealthzResponse(status, None, "healthz_unhealthy")
    return HealthzResponse(status, payload)


def _healthz_pid(payload: dict[str, object] | None) -> int | None:
    if payload is None:
        return None
    pid = payload.get("pid")
    if isinstance(pid, bool):
        return None
    return pid if isinstance(pid, int) else None


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
