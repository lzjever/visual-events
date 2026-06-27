from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    from tools import cli_local_e2e_manifest
    from tools.dds_pc_tools import (
        DEFAULT_BUILD_DIR,
        DEFAULT_CAMERA_TOPIC,
        DEFAULT_GAZE_TOPIC,
        DEFAULT_HEAD_STATE_TOPIC,
        PcDdsToolError,
        validate_domain_network,
    )
except ModuleNotFoundError:
    import cli_local_e2e_manifest  # type: ignore[no-redef]
    from dds_pc_tools import (  # type: ignore[no-redef]
        DEFAULT_BUILD_DIR,
        DEFAULT_CAMERA_TOPIC,
        DEFAULT_GAZE_TOPIC,
        DEFAULT_HEAD_STATE_TOPIC,
        PcDdsToolError,
        validate_domain_network,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_SERVER_URL = "ws://127.0.0.1:8767/v1/stream"
DEFAULT_DDS_BRIDGE_BINARY = "visual_events_dds_bridge"
STOP_TIMEOUT_S = 5.0
VALID_GAZE_STATES = {"tracking", "lost", "stale", "disabled"}
NOT_COVERED = [
    "full_scene_matrix",
    "oracle",
    "latency_p95_p99",
    "soak",
    "fault_matrix",
    "release_report",
]
MANIFEST_REPORT_KEYS = [
    "data_dir",
    "manifest_source",
    "manifest_path",
    "manifest_sha256",
    "scene_count",
    "frame_count",
    "effective_manifest",
]


class PreflightError(Exception):
    pass


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class HealthCheckResult:
    passed: bool
    failure_reason: str | None = None
    healthz_pid: int | None = None
    healthz_identity_verified: bool = False


@dataclass(frozen=True)
class HealthzResponse:
    status: int
    payload: dict[str, Any] | None
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


class ProcessRunner(Protocol):
    def start_process(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        name: str,
    ) -> ProcessLike:
        ...

    def run_sync(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        name: str,
        timeout_s: float | None = None,
    ) -> CommandResult:
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

    def wait_process(
        self,
        process: ProcessLike,
        *,
        name: str,
        timeout_s: float | None = None,
    ) -> CommandResult:
        ...

    def sleep(self, seconds: float) -> None:
        ...

    def stop_process(self, process: ProcessLike) -> None:
        ...


@dataclass
class ManagedProcess:
    name: str
    popen: subprocess.Popen[str]
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def pid(self) -> int:
        return int(self.popen.pid)

    def poll(self) -> int | None:
        return self.popen.poll()

    def terminate(self) -> None:
        self.popen.terminate()

    def wait(self, timeout: float | None = None) -> int:
        return int(self.popen.wait(timeout=timeout))

    def kill(self) -> None:
        self.popen.kill()

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        stdout, stderr = self.popen.communicate(timeout=timeout)
        self.stdout_tail = _tail(stdout)
        self.stderr_tail = _tail(stderr)
        return stdout or "", stderr or ""


@dataclass(frozen=True)
class ParsedServerUrl:
    original: str
    host: str
    port: int
    healthz_url: str


@dataclass(frozen=True)
class CliLocalE2EConfig:
    data_dir: Path
    out: Path
    manifest_report: dict[str, Any]
    server_bin: Path
    cli_bin: Path
    server_url: ParsedServerUrl
    server_config: Path | None
    build_dir: Path
    dds_bridge_bin: Path
    dds_domain: int
    dds_network: str
    allow_non_loopback_dds: bool
    image_topic: str
    head_state_topic: str
    gaze_topic: str
    dds_source_camera_name: str
    logical_camera_name: str
    selected_scene: str
    scene_dir: Path
    frame_count: int
    image_hz: float
    head_state: str
    head_state_hz: float
    gaze_count: int
    gaze_timeout_ms: int
    health_timeout_s: float
    health_interval_s: float
    startup_grace_s: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a partial real-server DDS plumbing smoke for the CLI."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--server-bin", required=True, type=Path)
    parser.add_argument("--cli-bin", required=True, type=Path)
    parser.add_argument("--server", default=DEFAULT_SERVER_URL)
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--dds-bridge-bin", type=Path, default=None)
    parser.add_argument("--dds-domain", required=True, type=_non_negative_int)
    parser.add_argument("--dds-network", required=True, type=_non_empty_str)
    parser.add_argument("--allow-non-loopback-dds", action="store_true")
    parser.add_argument("--image-topic", default=DEFAULT_CAMERA_TOPIC)
    parser.add_argument("--head-state-topic", default=DEFAULT_HEAD_STATE_TOPIC)
    parser.add_argument("--gaze-topic", default=DEFAULT_GAZE_TOPIC)
    parser.add_argument("--dds-source-camera-name", default="image")
    parser.add_argument("--logical-camera-name", default="front")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frame-count", type=_positive_int, default=5)
    parser.add_argument("--image-hz", type=_positive_float, default=10.0)
    parser.add_argument(
        "--head-state",
        choices=("stationary", "moving", "unknown"),
        default="stationary",
    )
    parser.add_argument("--head-state-hz", type=_positive_float, default=10.0)
    parser.add_argument("--gaze-count", type=_positive_int, default=1)
    parser.add_argument("--gaze-timeout-ms", type=_positive_int, default=5000)
    parser.add_argument("--health-timeout-s", type=_positive_float, default=15.0)
    parser.add_argument("--health-interval-s", type=_positive_float, default=0.1)
    parser.add_argument("--startup-grace-s", type=_non_negative_float, default=0.5)
    return parser.parse_args(argv)


class LocalProcessRunner:
    def start_process(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        name: str,
    ) -> ManagedProcess:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return ManagedProcess(name=name, popen=process)

    def run_sync(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        name: str,
        timeout_s: float | None = None,
    ) -> CommandResult:
        del name
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return CommandResult(
            returncode=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
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
                return HealthCheckResult(False, "healthz_identity_mismatch", healthz_pid)
            if process.poll() is not None:
                return HealthCheckResult(False, "server_exited_early", healthz_pid, True)
            return HealthCheckResult(True, healthz_pid=healthz_pid, healthz_identity_verified=True)

        if process.poll() is not None:
            return HealthCheckResult(False, "server_exited_early")
        return HealthCheckResult(False, "healthz_timeout")

    def wait_process(
        self,
        process: ProcessLike,
        *,
        name: str,
        timeout_s: float | None = None,
    ) -> CommandResult:
        del name
        if isinstance(process, ManagedProcess):
            try:
                stdout, stderr = process.communicate(timeout=timeout_s)
                return CommandResult(
                    returncode=process.poll() if process.poll() is not None else 0,
                    stdout=stdout,
                    stderr=stderr,
                )
            except subprocess.TimeoutExpired:
                self.stop_process(process)
                return CommandResult(
                    returncode=process.poll() if process.poll() is not None else -9,
                    stdout=process.stdout_tail,
                    stderr=process.stderr_tail,
                )
        returncode = process.wait(timeout=timeout_s)
        return CommandResult(returncode=int(returncode))

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def stop_process(self, process: ProcessLike) -> None:
        if process.poll() is not None:
            if isinstance(process, ManagedProcess):
                _capture_finished_process(process)
            return
        process.terminate()
        try:
            if isinstance(process, ManagedProcess):
                process.communicate(timeout=STOP_TIMEOUT_S)
            else:
                process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            if isinstance(process, ManagedProcess):
                process.communicate(timeout=STOP_TIMEOUT_S)
            else:
                process.wait(timeout=STOP_TIMEOUT_S)


def main(
    argv: list[str] | None = None,
    *,
    runner: ProcessRunner | None = None,
) -> int:
    args = parse_args(argv)
    out_for_failure: Path | None = None
    manifest_report_for_failure: dict[str, Any] | None = None

    try:
        out, manifest_report = cli_local_e2e_manifest.build_report(
            data_dir=args.data_dir,
            out=args.out,
            manifest=args.manifest,
        )
        out_for_failure = out
        manifest_report_for_failure = manifest_report
        config = _build_config(args, out=out, manifest_report=manifest_report)
    except (OSError, PreflightError, cli_local_e2e_manifest.PreflightError, PcDdsToolError) as exc:
        failure_reason = str(exc)
        out_for_failure = out_for_failure or _safe_out_for_preflight_failure(args)
        if out_for_failure is not None:
            report = _build_preflight_failed_report(
                args=args,
                out=out_for_failure,
                manifest_report=manifest_report_for_failure,
                failure_reason=failure_reason,
            )
            _write_report(out_for_failure, report)
        print(f"cli local e2e preflight failed: {failure_reason}", file=sys.stderr)
        return 2

    active_runner = runner or LocalProcessRunner()
    return _run_smoke(config, active_runner)


def _build_config(
    args: argparse.Namespace,
    *,
    out: Path,
    manifest_report: dict[str, Any],
) -> CliLocalE2EConfig:
    data_dir = Path(manifest_report["data_dir"])
    server_bin = _preflight_executable(args.server_bin, name="server-bin")
    cli_bin = _preflight_executable(args.cli_bin, name="cli-bin")
    build_dir = _resolve_path(args.build_dir)
    if not build_dir.exists():
        raise PreflightError(f"build-dir not found: {args.build_dir}")
    if not build_dir.is_dir():
        raise PreflightError(f"build-dir is not a directory: {args.build_dir}")
    dds_bridge_bin = args.dds_bridge_bin
    if dds_bridge_bin is None:
        dds_bridge_bin = build_dir / DEFAULT_DDS_BRIDGE_BINARY
    dds_bridge_bin = _preflight_executable(dds_bridge_bin, name="dds-bridge-bin")
    validate_domain_network(
        dds_domain=args.dds_domain,
        dds_network=args.dds_network,
        allow_non_loopback_dds=args.allow_non_loopback_dds,
    )

    server_url = _parse_server_url(args.server)
    selected_scene = _select_scene(
        data_dir=data_dir,
        scene=args.scene,
        effective_manifest=manifest_report.get("effective_manifest"),
    )
    scene_dir = data_dir / selected_scene
    if not scene_dir.exists():
        raise PreflightError(f"scene directory not found: {scene_dir}")
    if not scene_dir.is_dir():
        raise PreflightError(f"scene is not a directory: {scene_dir}")
    if not cli_local_e2e_manifest.jpeg_files(scene_dir):
        raise PreflightError(f"scene has no JPEG frames: {selected_scene}")

    return CliLocalE2EConfig(
        data_dir=data_dir,
        out=out,
        manifest_report=manifest_report,
        server_bin=server_bin,
        cli_bin=cli_bin,
        server_url=server_url,
        server_config=_resolve_path(args.server_config) if args.server_config else None,
        build_dir=build_dir,
        dds_bridge_bin=dds_bridge_bin,
        dds_domain=args.dds_domain,
        dds_network=args.dds_network,
        allow_non_loopback_dds=args.allow_non_loopback_dds,
        image_topic=args.image_topic,
        head_state_topic=args.head_state_topic,
        gaze_topic=args.gaze_topic,
        dds_source_camera_name=args.dds_source_camera_name,
        logical_camera_name=args.logical_camera_name,
        selected_scene=selected_scene,
        scene_dir=scene_dir,
        frame_count=args.frame_count,
        image_hz=args.image_hz,
        head_state=args.head_state,
        head_state_hz=args.head_state_hz,
        gaze_count=args.gaze_count,
        gaze_timeout_ms=args.gaze_timeout_ms,
        health_timeout_s=args.health_timeout_s,
        health_interval_s=args.health_interval_s,
        startup_grace_s=args.startup_grace_s,
    )


def _run_smoke(config: CliLocalE2EConfig, runner: ProcessRunner) -> int:
    started_at = _utc_now()
    started_s = time.perf_counter()
    env = os.environ.copy()
    server_command = _server_command(config)
    cli_command = _cli_command(config)
    gaze_command = _gaze_subscriber_command(config)
    image_command = _image_publisher_command(config)
    head_command = _head_publisher_command(config)
    commands = {
        "server": server_command,
        "cli": cli_command,
        "gaze_subscriber": gaze_command,
        "head_publisher": head_command,
        "image_publisher": image_command,
    }
    failure_reasons: list[str] = []
    server_process: ProcessLike | None = None
    cli_process: ProcessLike | None = None
    gaze_process: ProcessLike | None = None
    health_result: HealthCheckResult | None = None
    head_result: CommandResult | None = None
    image_result: CommandResult | None = None
    gaze_result: CommandResult | None = None
    gaze_summary = _summarize_gaze_jsonl("", config.logical_camera_name, config.gaze_count)

    try:
        server_process = runner.start_process(
            server_command,
            cwd=REPO_ROOT,
            env=env,
            name="server",
        )
        health_result = runner.wait_healthz(
            config.server_url.healthz_url,
            server_process,
            timeout_s=config.health_timeout_s,
            interval_s=config.health_interval_s,
        )
        if not health_result.passed:
            failure_reasons.append(health_result.failure_reason or "healthz_failed")
        else:
            gaze_process = runner.start_process(
                gaze_command,
                cwd=REPO_ROOT,
                env=env,
                name="gaze_subscriber",
            )
            cli_process = runner.start_process(
                cli_command,
                cwd=REPO_ROOT,
                env=env,
                name="cli",
            )
            runner.sleep(config.startup_grace_s)
            head_result = runner.run_sync(
                head_command,
                cwd=REPO_ROOT,
                env=env,
                name="head_publisher",
            )
            if head_result.returncode != 0:
                failure_reasons.append("head_publisher_failed")
            image_result = runner.run_sync(
                image_command,
                cwd=REPO_ROOT,
                env=env,
                name="image_publisher",
            )
            if image_result.returncode != 0:
                failure_reasons.append("image_publisher_failed")
            gaze_result = runner.wait_process(
                gaze_process,
                name="gaze_subscriber",
                timeout_s=(config.gaze_timeout_ms / 1000.0) + STOP_TIMEOUT_S,
            )
            if gaze_result.returncode != 0:
                failure_reasons.append("gaze_subscriber_failed")
            gaze_summary = _summarize_gaze_jsonl(
                gaze_result.stdout,
                config.logical_camera_name,
                config.gaze_count,
            )
            if gaze_summary["parse_errors"]:
                failure_reasons.append("gaze_json_parse_errors")
            if gaze_summary["accepted_count"] < config.gaze_count:
                failure_reasons.append("gaze_target_count_shortfall")
    except Exception as exc:
        failure_reasons.append(f"runner_exception:{type(exc).__name__}")
    finally:
        if cli_process is not None:
            runner.stop_process(cli_process)
        if gaze_process is not None and gaze_process.poll() is None:
            runner.stop_process(gaze_process)
        if server_process is not None:
            runner.stop_process(server_process)

    failure_reasons = _dedupe(failure_reasons)
    slice_pass = _slice_pass(
        failure_reasons=failure_reasons,
        health_result=health_result,
        head_result=head_result,
        image_result=image_result,
        gaze_result=gaze_result,
        gaze_summary=gaze_summary,
        expected_gaze_count=config.gaze_count,
    )
    report = _build_smoke_report(
        config,
        commands=commands,
        started_at=started_at,
        elapsed_s=max(0.0, time.perf_counter() - started_s),
        slice_pass=slice_pass,
        failure_reasons=[] if slice_pass else failure_reasons,
        health_result=health_result,
        server_process=server_process,
        cli_process=cli_process,
        gaze_process=gaze_process,
        head_result=head_result,
        image_result=image_result,
        gaze_result=gaze_result,
        gaze_summary=gaze_summary,
    )
    _write_report(config.out, report)
    print(str(config.out))
    return 0 if slice_pass else 1


def _server_command(config: CliLocalE2EConfig) -> list[str]:
    command = [
        os.fspath(config.server_bin),
        "--host",
        config.server_url.host,
        "--port",
        str(config.server_url.port),
    ]
    if config.server_config is not None:
        command.extend(["--config", os.fspath(config.server_config)])
    return command


def _cli_command(config: CliLocalE2EConfig) -> list[str]:
    return [
        os.fspath(config.cli_bin),
        "--dds-runtime",
        "bridge",
        "--dds-bridge-bin",
        os.fspath(config.dds_bridge_bin),
        "--server",
        config.server_url.original,
        "--camera",
        config.logical_camera_name,
        "--dds-domain",
        str(config.dds_domain),
        "--dds-network",
        config.dds_network,
        "--image-topic",
        config.image_topic,
        "--head-state-topic",
        config.head_state_topic,
        "--gaze-topic",
        config.gaze_topic,
    ]


def _wrapper_common_command(config: CliLocalE2EConfig, script_name: str) -> list[str]:
    command = [
        os.fspath(sys.executable),
        os.fspath(TOOLS_DIR / script_name),
        "--build-dir",
        os.fspath(config.build_dir),
        "--dds-domain",
        str(config.dds_domain),
        "--dds-network",
        config.dds_network,
    ]
    if config.allow_non_loopback_dds:
        command.append("--allow-non-loopback-dds")
    return command


def _gaze_subscriber_command(config: CliLocalE2EConfig) -> list[str]:
    return [
        *_wrapper_common_command(config, "subscribe_test_gaze_targets.py"),
        "--count",
        str(config.gaze_count),
        "--timeout-ms",
        str(config.gaze_timeout_ms),
        "--gaze-topic",
        config.gaze_topic,
    ]


def _image_publisher_command(config: CliLocalE2EConfig) -> list[str]:
    return [
        *_wrapper_common_command(config, "publish_test_dds_images.py"),
        "--input",
        os.fspath(config.scene_dir),
        "--count",
        str(config.frame_count),
        "--hz",
        _format_number(config.image_hz),
        "--camera-name",
        config.dds_source_camera_name,
        "--camera-topic",
        config.image_topic,
    ]


def _head_publisher_command(config: CliLocalE2EConfig) -> list[str]:
    return [
        *_wrapper_common_command(config, "publish_test_head_state.py"),
        "--state",
        config.head_state,
        "--count",
        str(config.frame_count),
        "--hz",
        _format_number(config.head_state_hz),
        "--head-state-topic",
        config.head_state_topic,
    ]


def _build_smoke_report(
    config: CliLocalE2EConfig,
    *,
    commands: dict[str, list[str]],
    started_at: str,
    elapsed_s: float,
    slice_pass: bool,
    failure_reasons: list[str],
    health_result: HealthCheckResult | None,
    server_process: ProcessLike | None,
    cli_process: ProcessLike | None,
    gaze_process: ProcessLike | None,
    head_result: CommandResult | None,
    image_result: CommandResult | None,
    gaze_result: CommandResult | None,
    gaze_summary: dict[str, Any],
) -> dict[str, Any]:
    report = _base_report(
        manifest_report=config.manifest_report,
        status="partial_smoke_pass" if slice_pass else "partial_smoke_failed",
        failure_reasons=failure_reasons,
    )
    report.update(
        {
            "slice_pass": slice_pass,
            "server_bin": os.fspath(config.server_bin),
            "cli_bin": os.fspath(config.cli_bin),
            "build_dir": os.fspath(config.build_dir),
            "dds_bridge_bin": os.fspath(config.dds_bridge_bin),
            "server_url": config.server_url.original,
            "healthz_url": config.server_url.healthz_url,
            "commands": commands,
            "processes": {
                "server": _process_report(server_process),
                "cli": _process_report(cli_process),
                "gaze_subscriber": _process_report(gaze_process),
            },
            "returncodes": {
                "head_publisher": None if head_result is None else head_result.returncode,
                "image_publisher": None if image_result is None else image_result.returncode,
                "gaze_subscriber": None if gaze_result is None else gaze_result.returncode,
            },
            "dds": {
                "domain": config.dds_domain,
                "network": config.dds_network,
                "allow_non_loopback_dds": config.allow_non_loopback_dds,
                "image_topic": config.image_topic,
                "head_state_topic": config.head_state_topic,
                "gaze_topic": config.gaze_topic,
            },
            "camera_names": {
                "dds_source_camera_name": config.dds_source_camera_name,
                "logical_camera_name": config.logical_camera_name,
            },
            "selected_scene": config.selected_scene,
            "selected_scene_dir": os.fspath(config.scene_dir),
            "published_frames": config.frame_count,
            "image_hz": config.image_hz,
            "head_state": {
                "required": True,
                "publisher_mode": "required",
                "state": config.head_state,
                "hz": config.head_state_hz,
                "count": config.frame_count,
            },
            "gaze": gaze_summary,
            "health": {
                "ok": False if health_result is None else health_result.passed,
                "failure_reason": None if health_result is None else health_result.failure_reason,
                "pid": None if health_result is None else health_result.healthz_pid,
                "identity_verified": False
                if health_result is None
                else health_result.healthz_identity_verified,
                "timeout_s": config.health_timeout_s,
                "interval_s": config.health_interval_s,
            },
            "stdout_stderr_tails": {
                "server": _tail_report(server_process),
                "cli": _tail_report(cli_process),
                "gaze_subscriber": {
                    "stdout_tail": _tail("" if gaze_result is None else gaze_result.stdout),
                    "stderr_tail": _tail("" if gaze_result is None else gaze_result.stderr),
                },
                "head_publisher": {
                    "stdout_tail": _tail("" if head_result is None else head_result.stdout),
                    "stderr_tail": _tail("" if head_result is None else head_result.stderr),
                },
                "image_publisher": {
                    "stdout_tail": _tail("" if image_result is None else image_result.stdout),
                    "stderr_tail": _tail("" if image_result is None else image_result.stderr),
                },
            },
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_s": float(elapsed_s),
        }
    )
    return report


def _build_preflight_failed_report(
    *,
    args: argparse.Namespace,
    out: Path,
    manifest_report: dict[str, Any] | None,
    failure_reason: str,
) -> dict[str, Any]:
    if manifest_report is None:
        data_dir = cli_local_e2e_manifest.resolve_path(args.data_dir)
        manifest_report = {
            "data_dir": os.fspath(data_dir),
            "manifest_source": None,
            "manifest_path": None,
            "manifest_sha256": None,
            "scene_count": None,
            "frame_count": None,
            "effective_manifest": None,
        }
    report = _base_report(
        manifest_report=manifest_report,
        status="preflight_failed",
        failure_reasons=[failure_reason],
    )
    report.update(
        {
            "slice_pass": False,
            "out": os.fspath(out),
            "server_url": getattr(args, "server", DEFAULT_SERVER_URL),
            "commands": {},
        }
    )
    return report


def _base_report(
    *,
    manifest_report: dict[str, Any],
    status: str,
    failure_reasons: list[str],
) -> dict[str, Any]:
    report = {
        "report_type": "cli_local_e2e_smoke_v1",
        "slice_pass": False,
        "overall_pass": False,
        "ga_gate_pass": False,
        "pc_local_e2e_status": status,
        "failure_reasons": failure_reasons,
        "not_covered": NOT_COVERED,
    }
    for key in MANIFEST_REPORT_KEYS:
        report[key] = manifest_report.get(key)
    return report


def _summarize_gaze_jsonl(
    stdout: str,
    logical_camera_name: str,
    expected_count: int,
) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    accepted_samples: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    rejected_count = 0
    valid_count = 0
    invalid_count = 0
    state_counts: dict[str, int] = {}

    for index, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            rejected_count += 1
            parse_errors.append(f"line {index}: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            rejected_count += 1
            continue
        if payload.get("type") != "gaze_target":
            rejected_count += 1
            continue
        if payload.get("camera") != logical_camera_name:
            rejected_count += 1
            continue
        state = payload.get("state")
        if state not in VALID_GAZE_STATES:
            rejected_count += 1
            continue
        accepted_samples.append(payload)
        state_counts[state] = state_counts.get(state, 0) + 1
        if payload.get("valid") is True:
            valid_count += 1
        elif payload.get("valid") is False:
            invalid_count += 1

    return {
        "expected_count": expected_count,
        "received_lines": len(lines),
        "state_counts": dict(sorted(state_counts.items())),
        "accepted_count": len(accepted_samples),
        "rejected_count": rejected_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "parse_errors": parse_errors,
        "first_sample": accepted_samples[0] if accepted_samples else None,
        "last_sample": accepted_samples[-1] if accepted_samples else None,
    }


def _slice_pass(
    *,
    failure_reasons: list[str],
    health_result: HealthCheckResult | None,
    head_result: CommandResult | None,
    image_result: CommandResult | None,
    gaze_result: CommandResult | None,
    gaze_summary: dict[str, Any],
    expected_gaze_count: int,
) -> bool:
    return (
        not failure_reasons
        and health_result is not None
        and health_result.passed
        and head_result is not None
        and head_result.returncode == 0
        and image_result is not None
        and image_result.returncode == 0
        and gaze_result is not None
        and gaze_result.returncode == 0
        and gaze_summary["accepted_count"] >= expected_gaze_count
    )


def _select_scene(
    *,
    data_dir: Path,
    scene: str | None,
    effective_manifest: Any,
) -> str:
    if scene is not None:
        scene_path = Path(scene)
        if scene_path.is_absolute() or len(scene_path.parts) != 1:
            raise PreflightError("--scene must be a scene directory name")
        return scene

    if not isinstance(effective_manifest, dict):
        raise PreflightError("effective manifest is not an object")
    scenes = effective_manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise PreflightError("effective manifest has no scenes")
    first = scenes[0]
    if not isinstance(first, dict):
        raise PreflightError("effective manifest first scene is invalid")
    name = first.get("scene_name")
    if not isinstance(name, str):
        name = first.get("name")
    if not isinstance(name, str) or name == "":
        raise PreflightError("effective manifest first scene has no scene name")
    scene_path = data_dir / name
    if not scene_path.exists():
        raise PreflightError(f"selected scene directory not found: {scene_path}")
    return name


def _parse_server_url(value: str) -> ParsedServerUrl:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"ws", "wss"}:
        raise PreflightError("--server must be a ws:// or wss:// URL")
    if parsed.path != "/v1/stream":
        raise PreflightError("--server URL path must be /v1/stream")
    if parsed.query or parsed.fragment:
        raise PreflightError("--server URL must not include query or fragment")
    if parsed.hostname is None:
        raise PreflightError("--server URL must include a host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PreflightError("--server URL port is invalid") from exc
    if port is None:
        port = 443 if parsed.scheme == "wss" else 80
    if port <= 0:
        raise PreflightError("--server URL port must be positive")
    health_scheme = "https" if parsed.scheme == "wss" else "http"
    health_host = _url_host(parsed.hostname)
    return ParsedServerUrl(
        original=value,
        host=parsed.hostname,
        port=port,
        healthz_url=f"{health_scheme}://{health_host}:{port}/healthz",
    )


def _preflight_executable(path: Path, *, name: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise PreflightError(f"{name} not found: {path}")
    if not resolved.is_file():
        raise PreflightError(f"{name} is not a file: {path}")
    if not os.access(resolved, os.X_OK):
        raise PreflightError(f"{name} is not executable: {path}")
    return resolved


def _safe_out_for_preflight_failure(args: argparse.Namespace) -> Path | None:
    try:
        data_dir = cli_local_e2e_manifest.resolve_path(args.data_dir)
        return cli_local_e2e_manifest.preflight_out_path(args.out, data_dir=data_dir)
    except (OSError, cli_local_e2e_manifest.PreflightError):
        return None


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


def _healthz_pid(payload: dict[str, Any] | None) -> int | None:
    if payload is None:
        return None
    pid = payload.get("pid")
    if isinstance(pid, bool):
        return None
    return pid if isinstance(pid, int) else None


def _process_report(process: ProcessLike | None) -> dict[str, Any]:
    return {
        "pid": None if process is None else process.pid,
        "returncode": None if process is None else process.poll(),
    }


def _tail_report(process: ProcessLike | None) -> dict[str, str]:
    return {
        "stdout_tail": _tail(str(getattr(process, "stdout_tail", ""))),
        "stderr_tail": _tail(str(getattr(process, "stderr_tail", ""))),
    }


def _capture_finished_process(process: ManagedProcess) -> None:
    if process.stdout_tail or process.stderr_tail:
        return
    try:
        process.communicate(timeout=0)
    except (ValueError, subprocess.TimeoutExpired):
        return


def _tail(value: str, *, limit: int = 4096) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _write_report(out: Path, report: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _url_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be positive") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be non-negative") from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_empty_str(value: str) -> str:
    if value == "":
        raise argparse.ArgumentTypeError("must be non-empty")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
