from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    from tools import runtime_provenance
    from tools import cli_local_e2e_manifest
    from tools import replay_val_data
    from tools.dds_pc_tools import (
        DEFAULT_BUILD_DIR,
        DEFAULT_CAMERA_TOPIC,
        DEFAULT_GAZE_TOPIC,
        DEFAULT_HEAD_STATE_TOPIC,
        PcDdsToolError,
        validate_domain_network,
    )
except ModuleNotFoundError:
    import runtime_provenance  # type: ignore[no-redef]
    import cli_local_e2e_manifest  # type: ignore[no-redef]
    import replay_val_data  # type: ignore[no-redef]
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
VALID_HEAD_STATES = ("stationary", "moving", "unknown")
DEFAULT_HEAD_STATE_SEGMENTS = VALID_HEAD_STATES
MIN_HEAD_STATE_HZ = 9.0
MIN_GAZE_PUBLISH_HZ = 9.0
MAX_GAZE_PUBLISH_HZ = 10.5
MIN_PC_GA_ATTENTION_SWITCH_CONFIRM_MS = 750.0
MAX_PC_GA_TARGET_NORM_JITTER_P95 = 0.04
TARGET_NORM_CONSISTENCY_TOLERANCE = 1e-3
MAX_CAPTURE_TO_GAZE_LATENCY_MS = 60_000.0
PROCESS_COLLECTED_LINE_LIMIT = 4096
PROCESS_READER_JOIN_TIMEOUT_S = 1.0
BOTIFIED_OPEN = "<botified>"
BOTIFIED_CLOSE = "</botified>"
BOTIFIED_ALLOWED_EVENTS = {
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
}
BOTIFIED_EVENT_RE = re.compile(r"\bevent=([A-Za-z0-9_]+)\b")
BOTIFIED_TRACK_ID_RE = re.compile(r"\btrack_id=(\d+)\b")
BOTIFIED_OUTPUT_LIMIT_PER_TRACK_EVENT_60S_THRESHOLD = 1
BOTIFIED_OUTPUT_LIMIT_GLOBAL_60S_THRESHOLD = 12
BOTIFIED_OUTPUT_LIMIT_BURST_1S_THRESHOLD = 3
BOTIFIED_OUTPUT_LIMIT_60S_WINDOW_MS = 60_000
BOTIFIED_OUTPUT_LIMIT_1S_WINDOW_MS = 1_000
NOT_COVERED = [
    "full_scene_matrix",
    "oracle",
    "latency_p95_p99",
    "soak",
    "fault_matrix",
    "release_report",
]
FULL_MATRIX_NON_BLOCKING_GAPS = [
    "latency_p95_p99",
    "soak",
    "fault_matrix",
    "release_report",
]
CURRENT_PC_CORE_GATE_SCOPE = "current_pc_core_gate"
GA_GATE_STATUS_NOT_EVALUATED = "not_evaluated"
GA_GATE_STATUS_PC_SIMULATED_PASS = "pc_simulated_ga_pass"
GA_GATE_STATUS_PC_SIMULATED_FAIL = "pc_simulated_ga_fail"
GA_GATE_STATUS = GA_GATE_STATUS_NOT_EVALUATED
POST_GA_NOT_COVERED = [
    "real_robot_validation",
    "rk3588_board_validation",
    "field_validation",
    "release_audit",
]
MANIFEST_UNREADABLE_OR_INVALID = "manifest_unreadable_or_invalid"
MANIFEST_REPORT_KEYS = [
    "data_dir",
    "manifest_source",
    "manifest_path",
    "manifest_sha256",
    "manifest_schema_version",
    "manifest_authoritative",
    "manifest_validation_errors",
    "oracle_schema_present",
    "oracle_schema_valid",
    "oracle_summary",
    "manifest_contract_required",
    "manifest_contract_satisfied",
    "manifest_contract_failure_reasons",
    "oracle_evaluated",
    "oracle_evaluation_passed",
    "scene_count",
    "frame_count",
    "effective_manifest",
]
SERVER_SCRIPT_NAME = "visual-events-server"
CLI_SCRIPT_NAME = "visual-events-cli"
EXPECTED_RUNTIME_EXIT_CODES = {0, -15}


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


@dataclass(frozen=True)
class HeadStateSegment:
    state: str
    frame_count: int
    head_state_hz: float


@dataclass(frozen=True)
class CliFrameRequestLogRead:
    records: tuple[dict[str, Any], ...]
    parse_error_count: int
    parse_errors: tuple[dict[str, Any], ...]
    line_count: int
    next_line_offset: int


@dataclass(frozen=True)
class SegmentSmokeResult:
    segment: HeadStateSegment
    head_result: CommandResult | None
    image_result: CommandResult | None
    gaze_result: CommandResult | None
    gaze_summary: dict[str, Any]
    gaze_latency_samples_ms: tuple[float, ...]
    cli_frame_request_records: tuple[dict[str, Any], ...]
    cli_frame_request_log_parse_error_count: int
    cli_frame_request_log_parse_errors: tuple[dict[str, Any], ...]
    cli_frame_request_log_line_count: int
    head_stdout_json: dict[str, Any] | None
    head_stdout_parse_error: str | None
    failure_reasons: list[str]


@dataclass(frozen=True)
class SmokeRunResult:
    report: dict[str, Any]
    rc: int


@dataclass(frozen=True)
class WarmupResult:
    enabled: bool
    passed: bool | None
    failure_reason: str | None = None
    count: int = 0
    image_dir: Path | None = None
    image_result: CommandResult | None = None
    head_result: CommandResult | None = None


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
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    collection_incomplete: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stdout_lines: deque[str] = field(init=False, repr=False)
    _stdout_line_timestamps_ms: deque[int] = field(init=False, repr=False)
    _stderr_lines: deque[str] = field(init=False, repr=False)
    _stdout_line_count: int = field(default=0, init=False, repr=False)
    _stderr_line_count: int = field(default=0, init=False, repr=False)
    _reader_threads: list[threading.Thread] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._stdout_lines = deque(maxlen=PROCESS_COLLECTED_LINE_LIMIT)
        self._stdout_line_timestamps_ms = deque(maxlen=PROCESS_COLLECTED_LINE_LIMIT)
        self._stderr_lines = deque(maxlen=PROCESS_COLLECTED_LINE_LIMIT)
        self._start_reader("stdout", self.popen.stdout)
        self._start_reader("stderr", self.popen.stderr)

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
        self.wait(timeout=timeout)
        self.join_readers(PROCESS_READER_JOIN_TIMEOUT_S)
        return self.collected_stdout(), self.collected_stderr()

    @property
    def stdout_lines(self) -> list[str]:
        with self._lock:
            return list(self._stdout_lines)

    @property
    def stdout_line_timestamps_ms(self) -> list[int]:
        with self._lock:
            return list(self._stdout_line_timestamps_ms)

    @property
    def stderr_lines(self) -> list[str]:
        with self._lock:
            return list(self._stderr_lines)

    @property
    def stdout_line_count(self) -> int:
        with self._lock:
            return self._stdout_line_count

    @property
    def stderr_line_count(self) -> int:
        with self._lock:
            return self._stderr_line_count

    def collected_stdout(self) -> str:
        return _lines_to_text(self.stdout_lines)

    def collected_stderr(self) -> str:
        return _lines_to_text(self.stderr_lines)

    def join_readers(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        for thread in self._reader_threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive():
                with self._lock:
                    self.collection_incomplete = True

    def _start_reader(self, stream_name: str, stream: Any | None) -> None:
        if stream is None:
            return
        thread = threading.Thread(
            target=self._drain_stream,
            args=(stream_name, stream),
            name=f"{self.name}-{stream_name}-reader",
            daemon=True,
        )
        self._reader_threads.append(thread)
        thread.start()

    def _drain_stream(self, stream_name: str, stream: Any) -> None:
        try:
            for chunk in stream:
                self._record_stream_line(stream_name, chunk)
        except Exception:
            with self._lock:
                self.collection_incomplete = True
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _record_stream_line(self, stream_name: str, chunk: str) -> None:
        line = chunk.rstrip("\n")
        if line.endswith("\r"):
            line = line[:-1]
        with self._lock:
            if stream_name == "stdout":
                self.stdout_tail = _tail(self.stdout_tail + chunk)
                self._stdout_line_count += 1
                if len(self._stdout_lines) >= PROCESS_COLLECTED_LINE_LIMIT:
                    self.stdout_truncated = True
                received_monotonic_ms = time.monotonic_ns() // 1_000_000
                self._stdout_lines.append(line)
                self._stdout_line_timestamps_ms.append(received_monotonic_ms)
            else:
                self.stderr_tail = _tail(self.stderr_tail + chunk)
                self._stderr_line_count += 1
                if len(self._stderr_lines) >= PROCESS_COLLECTED_LINE_LIMIT:
                    self.stderr_truncated = True
                self._stderr_lines.append(line)


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
    runtime_provenance: dict[str, Any]
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
    scene_replay_mode: str
    selected_scene_frame_count: int
    frame_count: int
    image_hz: float
    head_state_mode: str
    head_state_segments: tuple[HeadStateSegment, ...]
    head_state_hz: float
    gaze_collection_mode: str
    gaze_count: int
    gaze_duration_ms: int | None
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
    parser.add_argument(
        "--require-authoritative-manifest",
        action="store_true",
        help="Fail preflight unless a valid authoritative manifest/oracle contract is present.",
    )
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
    parser.add_argument("--full-scene", action="store_true")
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--frame-count", type=_positive_int, default=None)
    parser.add_argument("--image-hz", type=_positive_float, default=10.0)
    parser.add_argument(
        "--head-state",
        choices=VALID_HEAD_STATES,
        default=None,
        help="Compatibility shortcut for a single required head-state segment.",
    )
    parser.add_argument("--head-state-mode", choices=("required",), default="required")
    parser.add_argument("--head-state-segments", default=None)
    parser.add_argument("--head-state-hz", type=_positive_float, default=10.0)
    parser.add_argument("--gaze-count", type=_positive_int, default=None)
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
                returncode = process.wait(timeout=timeout_s)
                process.join_readers(PROCESS_READER_JOIN_TIMEOUT_S)
                return CommandResult(
                    returncode=int(returncode),
                    stdout=process.collected_stdout(),
                    stderr=process.collected_stderr(),
                )
            except subprocess.TimeoutExpired:
                self.stop_process(process)
                return CommandResult(
                    returncode=process.poll() if process.poll() is not None else -9,
                    stdout=process.collected_stdout(),
                    stderr=process.collected_stderr(),
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
                process.wait(timeout=STOP_TIMEOUT_S)
            else:
                process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            if isinstance(process, ManagedProcess):
                process.wait(timeout=STOP_TIMEOUT_S)
            else:
                process.wait(timeout=STOP_TIMEOUT_S)
        finally:
            if isinstance(process, ManagedProcess):
                process.join_readers(PROCESS_READER_JOIN_TIMEOUT_S)


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
        manifest_report = _manifest_report_with_contract(
            manifest_report,
            contract_required=args.require_authoritative_manifest,
        )
        out_for_failure = out
        manifest_report_for_failure = manifest_report
        _preflight_required_manifest_contract(manifest_report)
        _preflight_manifest_contract(manifest_report)
        config = _build_config(args, out=out, manifest_report=manifest_report)
    except (OSError, PreflightError, cli_local_e2e_manifest.PreflightError, PcDdsToolError) as exc:
        failure_reason = str(exc)
        if manifest_report_for_failure is None:
            manifest_report_for_failure = _manifest_read_failure_report(args, exc)
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
    if args.all_scenes:
        return _run_full_scene_matrix(config, active_runner)
    return _run_smoke(config, active_runner)


def _manifest_read_failure_report(
    args: argparse.Namespace,
    exc: BaseException,
) -> dict[str, Any] | None:
    try:
        data_dir = cli_local_e2e_manifest.resolve_path(args.data_dir)
        if not data_dir.exists() or not data_dir.is_dir():
            return None
        cli_local_e2e_manifest.preflight_out_path(args.out, data_dir=data_dir)
        manifest_path = _selected_manifest_read_path(args, data_dir=data_dir)
    except (OSError, cli_local_e2e_manifest.PreflightError):
        return None

    if manifest_path is None or not _is_manifest_read_or_parse_error(
        exc, manifest_path=manifest_path
    ):
        return None

    return {
        "data_dir": os.fspath(data_dir),
        "manifest_source": "file",
        "manifest_path": os.fspath(manifest_path),
        "manifest_sha256": None,
        "manifest_schema_version": None,
        "manifest_authoritative": False,
        "manifest_validation_errors": [MANIFEST_UNREADABLE_OR_INVALID],
        "oracle_schema_present": False,
        "oracle_schema_valid": False,
        "oracle_summary": None,
        "scene_count": None,
        "frame_count": None,
        "effective_manifest": None,
    }


def _selected_manifest_read_path(
    args: argparse.Namespace,
    *,
    data_dir: Path,
) -> Path | None:
    manifest = getattr(args, "manifest", None)
    if manifest is not None:
        manifest_path = cli_local_e2e_manifest.resolve_path(manifest)
        if manifest_path.exists() and manifest_path.is_file():
            return manifest_path
        return None

    default_manifest = data_dir / "manifest.json"
    return default_manifest if default_manifest.exists() else None


def _is_manifest_read_or_parse_error(
    exc: BaseException,
    *,
    manifest_path: Path,
) -> bool:
    if isinstance(exc, cli_local_e2e_manifest.PreflightError):
        return str(exc).startswith("manifest JSON is invalid:")
    if isinstance(exc, OSError):
        filenames = [
            value
            for value in (
                getattr(exc, "filename", None),
                getattr(exc, "filename2", None),
            )
            if value is not None
        ]
        for filename in filenames:
            if cli_local_e2e_manifest.resolve_path(Path(filename)) == manifest_path:
                return True
        return os.fspath(manifest_path) in str(exc)
    return False


def _manifest_report_with_contract(
    manifest_report: dict[str, Any],
    *,
    contract_required: bool,
) -> dict[str, Any]:
    report = dict(manifest_report)
    validation_errors = report.get("manifest_validation_errors")
    if isinstance(validation_errors, list):
        report["manifest_validation_errors"] = list(validation_errors)

    contract_satisfied = cli_local_e2e_manifest.manifest_contract_satisfied(report)
    report["manifest_contract_required"] = bool(contract_required)
    report["manifest_contract_satisfied"] = contract_satisfied
    report["manifest_contract_failure_reasons"] = (
        []
        if contract_satisfied
        else cli_local_e2e_manifest.manifest_contract_failure_reasons(report)
    )
    report["oracle_evaluated"] = False
    report["oracle_evaluation_passed"] = None
    return report


def _build_config(
    args: argparse.Namespace,
    *,
    out: Path,
    manifest_report: dict[str, Any],
) -> CliLocalE2EConfig:
    if args.all_scenes:
        if not args.full_scene:
            raise PreflightError("--all-scenes requires --full-scene")
        if args.scene is not None:
            raise PreflightError("--all-scenes cannot be combined with --scene")
        if args.frame_count is not None:
            raise PreflightError("--all-scenes cannot be combined with --frame-count")
    data_dir = Path(manifest_report["data_dir"])
    if args.all_scenes:
        _all_scene_names_from_effective_manifest(
            data_dir=data_dir,
            effective_manifest=manifest_report.get("effective_manifest"),
        )
    server_bin = _preflight_executable(args.server_bin, name="server-bin")
    cli_bin = _preflight_executable(args.cli_bin, name="cli-bin")
    server_config = _resolve_path(args.server_config) if args.server_config else None
    _preflight_pc_ga_server_config(server_config, required=args.all_scenes)
    runtime_provenance = _preflight_runtime_provenance(
        server_bin=server_bin,
        cli_bin=cli_bin,
        server_config=server_config,
    )
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
    selected_scene_frames = cli_local_e2e_manifest.jpeg_files(scene_dir)
    if not selected_scene_frames:
        raise PreflightError(f"scene has no JPEG frames: {selected_scene}")
    selected_scene_frame_count = len(selected_scene_frames)
    if args.full_scene and args.frame_count is not None:
        raise PreflightError("--full-scene cannot be combined with --frame-count")
    scene_replay_mode = "full_scene" if args.full_scene else "partial"
    if args.full_scene:
        frame_count = selected_scene_frame_count
        if args.gaze_count is None:
            gaze_collection_mode = "duration"
            gaze_count = 1
        else:
            gaze_collection_mode = "count"
            gaze_count = args.gaze_count
    else:
        frame_count = args.frame_count or 5
        gaze_collection_mode = "count"
        gaze_count = args.gaze_count if args.gaze_count is not None else 1
    gaze_duration_ms = (
        _full_scene_gaze_duration_ms(
            frame_count=frame_count,
            image_hz=args.image_hz,
            settle_grace_s=args.startup_grace_s,
        )
        if gaze_collection_mode == "duration"
        else None
    )
    head_state_segment_names = _parse_head_state_segment_names(
        head_state=args.head_state,
        head_state_segments=args.head_state_segments,
    )

    return CliLocalE2EConfig(
        data_dir=data_dir,
        out=out,
        manifest_report=manifest_report,
        server_bin=server_bin,
        cli_bin=cli_bin,
        runtime_provenance=runtime_provenance,
        server_url=server_url,
        server_config=server_config,
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
        scene_replay_mode=scene_replay_mode,
        selected_scene_frame_count=selected_scene_frame_count,
        frame_count=frame_count,
        image_hz=args.image_hz,
        head_state_mode=args.head_state_mode,
        head_state_segments=tuple(
            HeadStateSegment(
                state=state,
                frame_count=frame_count,
                head_state_hz=args.head_state_hz,
            )
            for state in head_state_segment_names
        ),
        head_state_hz=args.head_state_hz,
        gaze_collection_mode=gaze_collection_mode,
        gaze_count=gaze_count,
        gaze_duration_ms=gaze_duration_ms,
        gaze_timeout_ms=args.gaze_timeout_ms,
        health_timeout_s=args.health_timeout_s,
        health_interval_s=args.health_interval_s,
        startup_grace_s=args.startup_grace_s,
    )


def _preflight_pc_ga_server_config(
    server_config: Path | None,
    *,
    required: bool,
) -> None:
    if server_config is None:
        if required:
            raise PreflightError("pc_ga_server_config_missing")
        return
    if not server_config.exists():
        raise PreflightError("pc_ga_server_config_missing")
    if not server_config.is_file():
        raise PreflightError("pc_ga_server_config_invalid")
    try:
        with server_config.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PreflightError("pc_ga_server_config_invalid") from exc

    attention = payload.get("attention")
    if not isinstance(attention, dict):
        raise PreflightError("pc_ga_attention_switch_confirm_ms_missing")
    switch_confirm_ms = attention.get("switch_confirm_ms")
    if isinstance(switch_confirm_ms, bool) or not isinstance(
        switch_confirm_ms,
        int | float,
    ):
        raise PreflightError("pc_ga_attention_switch_confirm_ms_missing")
    if not math.isfinite(float(switch_confirm_ms)):
        raise PreflightError("pc_ga_server_config_invalid")
    if float(switch_confirm_ms) < MIN_PC_GA_ATTENTION_SWITCH_CONFIRM_MS:
        raise PreflightError("pc_ga_attention_switch_confirm_ms_below_min")


def _preflight_required_manifest_contract(manifest_report: dict[str, Any]) -> None:
    if manifest_report.get("manifest_contract_required") is not True:
        return
    if manifest_report.get("manifest_contract_satisfied") is True:
        return

    reasons = manifest_report.get("manifest_contract_failure_reasons")
    if not isinstance(reasons, list) or not reasons:
        reasons = cli_local_e2e_manifest.manifest_contract_failure_reasons(
            manifest_report
        )
    joined = ", ".join(str(reason) for reason in reasons)
    raise PreflightError(
        "authoritative manifest contract required but not satisfied: " + joined
    )


def _preflight_manifest_contract(manifest_report: dict[str, Any]) -> None:
    errors = manifest_report.get("manifest_validation_errors")
    if not errors:
        return
    if not isinstance(errors, list):
        raise PreflightError("manifest_validation_failed")
    joined = ",".join(str(error) for error in errors)
    raise PreflightError(f"manifest_validation_failed:{joined}")


def _run_smoke(config: CliLocalE2EConfig, runner: ProcessRunner) -> int:
    result = _run_smoke_result(config, runner)
    _write_report(config.out, result.report)
    print(str(config.out))
    return result.rc


def _run_smoke_result(
    config: CliLocalE2EConfig,
    runner: ProcessRunner,
) -> SmokeRunResult:
    started_at = _utc_now()
    started_s = time.perf_counter()
    orchestration_env = os.environ.copy()
    runtime_env = runtime_provenance.runtime_execution_env(orchestration_env)
    cli_request_log_path = _cli_frame_request_log_path(config)
    server_command = _server_command(config)
    cli_command = _cli_command(config)
    gaze_command = _gaze_subscriber_command(config)
    image_command = _image_publisher_command(config)
    head_commands = {
        segment.state: _head_publisher_command(config, segment)
        for segment in config.head_state_segments
    }
    first_segment = config.head_state_segments[0]
    commands = {
        "server": server_command,
        "cli": cli_command,
        "gaze_subscriber": gaze_command,
        "head_publisher": head_commands[first_segment.state],
        "head_publishers": head_commands,
        "image_publisher": image_command,
        "image_publishers": {
            segment.state: image_command for segment in config.head_state_segments
        },
    }
    warmup_result = WarmupResult(enabled=False, passed=None)
    failure_reasons: list[str] = []
    server_process: ProcessLike | None = None
    cli_process: ProcessLike | None = None
    gaze_process: ProcessLike | None = None
    head_process: ProcessLike | None = None
    health_result: HealthCheckResult | None = None
    segment_results: list[SegmentSmokeResult] = []
    cli_request_line_offset = 0
    gaze_summary = _summarize_gaze_jsonl("", config.logical_camera_name, config.gaze_count)
    botified_stdout = _summarize_botified_stdout_from_process(None)

    if config.head_state_hz < MIN_HEAD_STATE_HZ:
        failure_reasons.append("head_state_hz_below_min")

    try:
        _reset_cli_frame_request_log(cli_request_log_path)
        server_process = runner.start_process(
            server_command,
            cwd=REPO_ROOT,
            env=runtime_env,
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
            cli_process = runner.start_process(
                cli_command,
                cwd=REPO_ROOT,
                env=runtime_env,
                name="cli",
            )
            runner.sleep(config.startup_grace_s)
            warmup_result, warmup_commands = _run_full_scene_warmup(
                config,
                runner,
                orchestration_env=orchestration_env,
            )
            if warmup_commands is not None:
                commands["warmup"] = warmup_commands
            if warmup_result.enabled and not warmup_result.passed:
                failure_reasons.append(
                    f"warmup_failed:{warmup_result.failure_reason or 'unknown'}"
                )
            cli_request_line_offset = _read_cli_frame_request_log(
                cli_request_log_path
            ).next_line_offset
            for segment in config.head_state_segments:
                segment_suffix = f":segment={segment.state}"
                segment_start_request_line_offset = cli_request_line_offset
                gaze_process = runner.start_process(
                    gaze_command,
                    cwd=REPO_ROOT,
                    env=orchestration_env,
                    name="gaze_subscriber",
                )
                head_name = f"head_publisher:{segment.state}"
                head_process = runner.start_process(
                    head_commands[segment.state],
                    cwd=REPO_ROOT,
                    env=orchestration_env,
                    name=head_name,
                )
                image_result = runner.run_sync(
                    image_command,
                    cwd=REPO_ROOT,
                    env=orchestration_env,
                    name=f"image_publisher:{segment.state}",
                )
                segment_failure_reasons: list[str] = []
                if image_result.returncode != 0:
                    segment_failure_reasons.append(
                        f"image_publisher_failed{segment_suffix}"
                    )

                head_result = runner.wait_process(
                    head_process,
                    name=head_name,
                    timeout_s=_publisher_timeout_s(segment.frame_count, segment.head_state_hz),
                )
                head_stdout_json, head_stdout_parse_error = _parse_head_publisher_stdout(
                    head_result.stdout
                )
                segment_failure_reasons.extend(
                    _head_segment_failure_reasons(
                        segment=segment,
                        result=head_result,
                        stdout_json=head_stdout_json,
                        stdout_parse_error=head_stdout_parse_error,
                    )
                )
                head_process = None

                gaze_result = runner.wait_process(
                    gaze_process,
                    name="gaze_subscriber",
                    timeout_s=_gaze_subscriber_timeout_s(config),
                )
                if gaze_result.returncode != 0:
                    segment_failure_reasons.append(
                        f"gaze_subscriber_failed{segment_suffix}"
                    )
                (
                    gaze_summary,
                    gaze_latency_samples_ms,
                ) = _summarize_gaze_jsonl_with_samples(
                    gaze_result.stdout,
                    config.logical_camera_name,
                    config.gaze_count,
                )
                if gaze_summary["parse_errors"]:
                    segment_failure_reasons.append(
                        f"gaze_json_parse_errors{segment_suffix}"
                    )
                if gaze_summary["accepted_count"] < config.gaze_count:
                    segment_failure_reasons.append(
                        f"gaze_target_count_shortfall{segment_suffix}"
                    )
                segment_failure_reasons.extend(
                    _capture_to_gaze_publish_latency_failure_reasons(
                        gaze_summary,
                        segment_suffix=segment_suffix,
                    )
                )
                gaze_publish_hz = gaze_summary["fresh_gaze_publish_hz"]
                if (
                    config.gaze_collection_mode == "duration"
                    and not gaze_publish_hz["available"]
                ):
                    segment_failure_reasons.append(
                        f"fresh_gaze_publish_hz_unavailable{segment_suffix}"
                    )
                elif (
                    gaze_publish_hz["available"]
                    and gaze_publish_hz["pass"] is False
                ):
                    hz = _finite_float(gaze_publish_hz.get("hz"))
                    if hz is not None and hz > MAX_GAZE_PUBLISH_HZ:
                        segment_failure_reasons.append(
                            f"fresh_gaze_publish_hz_above_max{segment_suffix}"
                        )
                    else:
                        segment_failure_reasons.append(
                            f"gaze_publish_hz_below_min{segment_suffix}"
                        )
                segment_cli_request_log = _read_cli_frame_request_log(
                    cli_request_log_path,
                    line_offset=segment_start_request_line_offset,
                )
                segment_cli_request_records = segment_cli_request_log.records
                cli_request_line_offset = segment_cli_request_log.next_line_offset
                segment_failure_reasons.extend(
                    _head_cli_request_failure_reasons(
                        segment=segment,
                        records=segment_cli_request_records,
                        parse_error_count=(
                            segment_cli_request_log.parse_error_count
                        ),
                    )
                )
                segment_results.append(
                    SegmentSmokeResult(
                        segment=segment,
                        head_result=head_result,
                        image_result=image_result,
                        gaze_result=gaze_result,
                        gaze_summary=gaze_summary,
                        gaze_latency_samples_ms=gaze_latency_samples_ms,
                        cli_frame_request_records=segment_cli_request_records,
                        cli_frame_request_log_parse_error_count=(
                            segment_cli_request_log.parse_error_count
                        ),
                        cli_frame_request_log_parse_errors=(
                            segment_cli_request_log.parse_errors
                        ),
                        cli_frame_request_log_line_count=(
                            segment_cli_request_log.line_count
                        ),
                        head_stdout_json=head_stdout_json,
                        head_stdout_parse_error=head_stdout_parse_error,
                        failure_reasons=segment_failure_reasons,
                    )
                )
                failure_reasons.extend(segment_failure_reasons)
    except Exception as exc:
        failure_reasons.append(f"runner_exception:{type(exc).__name__}")
    finally:
        if cli_process is not None:
            runner.stop_process(cli_process)
        if head_process is not None and head_process.poll() is None:
            runner.stop_process(head_process)
        if gaze_process is not None and gaze_process.poll() is None:
            runner.stop_process(gaze_process)
        if server_process is not None:
            runner.stop_process(server_process)

    server_exit_code = _process_exit_code(server_process)
    cli_exit_code = _process_exit_code(cli_process)
    failure_reasons.extend(
        _runtime_exit_code_failure_reasons(
            server_exit_code=server_exit_code,
            cli_exit_code=cli_exit_code,
        )
    )
    botified_stdout = _summarize_botified_stdout_from_process(cli_process)
    failure_reasons.extend(_botified_stdout_failure_reasons(botified_stdout))
    failure_reasons = _dedupe(failure_reasons)
    slice_pass = _slice_pass(
        failure_reasons=failure_reasons,
        health_result=health_result,
        segment_results=segment_results,
        expected_segment_count=len(config.head_state_segments),
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
        segment_results=segment_results,
        gaze_summary=gaze_summary,
        botified_stdout=botified_stdout,
        warmup_result=warmup_result,
    )
    return SmokeRunResult(report=report, rc=0 if slice_pass else 1)


def _run_full_scene_warmup(
    config: CliLocalE2EConfig,
    runner: ProcessRunner,
    *,
    orchestration_env: dict[str, str],
) -> tuple[WarmupResult, dict[str, list[str]] | None]:
    if config.scene_replay_mode != "full_scene":
        return WarmupResult(enabled=False, passed=None), None

    head_process: ProcessLike | None = None
    head_result: CommandResult | None = None
    image_result: CommandResult | None = None
    image_dir: Path | None = None
    try:
        image_dir = _ensure_warmup_jpeg_dir(config)
        warmup_config = replace(config, scene_dir=image_dir, frame_count=1)
        head_segment = HeadStateSegment(
            state="stationary",
            frame_count=1,
            head_state_hz=max(config.head_state_hz, MIN_HEAD_STATE_HZ),
        )
        head_command = _head_publisher_command(warmup_config, head_segment)
        image_command = _image_publisher_command(warmup_config)
        commands = {
            "head_publisher": head_command,
            "image_publisher": image_command,
        }

        head_process = runner.start_process(
            head_command,
            cwd=REPO_ROOT,
            env=orchestration_env,
            name="head_publisher:warmup",
        )
        image_result = runner.run_sync(
            image_command,
            cwd=REPO_ROOT,
            env=orchestration_env,
            name="image_publisher:warmup",
        )
        head_result = runner.wait_process(
            head_process,
            name="head_publisher:warmup",
            timeout_s=_publisher_timeout_s(1, head_segment.head_state_hz),
        )
        head_process = None

        failure_reason = _warmup_failure_reason(
            image_result=image_result,
            head_result=head_result,
        )
        return (
            WarmupResult(
                enabled=True,
                passed=failure_reason is None,
                failure_reason=failure_reason,
                count=1,
                image_dir=image_dir,
                image_result=image_result,
                head_result=head_result,
            ),
            commands,
        )
    except Exception as exc:
        return (
            WarmupResult(
                enabled=True,
                passed=False,
                failure_reason=f"warmup_exception:{type(exc).__name__}",
                count=1,
                image_dir=image_dir,
                image_result=image_result,
                head_result=head_result,
            ),
            None,
        )
    finally:
        if head_process is not None and head_process.poll() is None:
            runner.stop_process(head_process)


def _warmup_failure_reason(
    *,
    image_result: CommandResult,
    head_result: CommandResult,
) -> str | None:
    if image_result.returncode != 0:
        return "warmup_image_publisher_failed"
    if head_result.returncode != 0:
        return "warmup_head_publisher_failed"
    head_stdout_json, head_stdout_parse_error = _parse_head_publisher_stdout(
        head_result.stdout
    )
    if head_stdout_parse_error is not None or head_stdout_json is None:
        return "warmup_head_publisher_malformed_json"
    if head_stdout_json.get("state") != "stationary":
        return "warmup_head_publisher_state_mismatch"
    published = head_stdout_json.get("published")
    if not isinstance(published, int) or isinstance(published, bool) or published < 1:
        return "warmup_head_publisher_count_mismatch"
    return None


def _ensure_warmup_jpeg_dir(config: CliLocalE2EConfig) -> Path:
    scene_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.selected_scene) or "scene"
    image_dir = config.out.parent / "tmp" / "cli-local-e2e-warmup" / scene_name
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "000001.jpg"
    _write_warmup_jpeg(image_path)
    return image_dir


def _write_warmup_jpeg(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to generate the DDS E2E warmup JPEG") from exc

    # DDS E2E warmup blank JPEG; must be decodable by server inference.
    image = Image.new("RGB", (1280, 720), (0, 0, 0))
    image.save(path, format="JPEG", quality=75)


def _run_full_scene_matrix(config: CliLocalE2EConfig, runner: ProcessRunner) -> int:
    scene_names = _all_scene_names_from_effective_manifest(
        data_dir=config.data_dir,
        effective_manifest=config.manifest_report.get("effective_manifest"),
    )
    scene_reports: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    oracle_evaluated = _should_evaluate_botified_event_oracle(config)

    for scene_name in scene_names:
        scene_config = _config_for_full_scene(config, scene_name)
        result = _run_smoke_result(scene_config, runner)
        scene_report = result.report
        scene_reports.append(_matrix_scene_result(scene_report))
        if result.rc != 0:
            failure_reasons.append(f"scene_failed:{scene_name}")
            failure_reasons.extend(
                f"{scene_name}:{reason}"
                for reason in scene_report.get("failure_reasons", [])
            )

    slice_matrix_pass = bool(scene_reports) and all(
        result.get("slice_pass") is True for result in scene_reports
    )
    if oracle_evaluated:
        botified_event_oracle, oracle_failure_reasons = (
            _evaluate_botified_event_oracle(scene_reports, head_motion="stationary")
        )
        failure_reasons.extend(oracle_failure_reasons)
    else:
        botified_event_oracle = _unevaluated_botified_event_oracle()
    if not oracle_evaluated:
        failure_reasons.append("botified_event_oracle_not_evaluated")
    attention_oracle_contract = _attention_oracle_contract(
        config.manifest_report.get("effective_manifest")
    )
    gaze_attention_oracle_evaluated = (
        oracle_evaluated and attention_oracle_contract is not None
    )
    if gaze_attention_oracle_evaluated:
        gaze_attention_oracle, attention_failure_reasons = (
            _evaluate_gaze_attention_oracle(
                scene_reports,
                attention_oracle_contract,
            )
        )
        failure_reasons.extend(attention_failure_reasons)
    else:
        gaze_attention_oracle = _unevaluated_gaze_attention_oracle()
    if not gaze_attention_oracle_evaluated:
        failure_reasons.append("gaze_attention_oracle_not_evaluated")
    oracle_contracts_present = _botified_event_oracle_contracts_present(
        botified_event_oracle
    )
    gaze_attention_oracle_contracts_present = (
        _gaze_attention_oracle_contracts_present(gaze_attention_oracle)
    )
    oracle_pass = (
        oracle_evaluated
        and botified_event_oracle["passed"] is True
        and oracle_contracts_present
    )
    gaze_attention_oracle_pass = (
        gaze_attention_oracle_evaluated
        and gaze_attention_oracle["passed"] is True
        and gaze_attention_oracle_contracts_present
    )
    current_pc_core_gate_pass = (
        slice_matrix_pass and oracle_pass and gaze_attention_oracle_pass
    )
    ga_gate_status = (
        GA_GATE_STATUS_PC_SIMULATED_PASS
        if current_pc_core_gate_pass
        else GA_GATE_STATUS_PC_SIMULATED_FAIL
    )
    report = _base_report(
        manifest_report=config.manifest_report,
        status="full_scene_matrix_pass"
        if current_pc_core_gate_pass
        else "full_scene_matrix_failed",
        failure_reasons=[]
        if current_pc_core_gate_pass
        else _dedupe(failure_reasons),
    )
    report.update(
        {
            "slice_pass": current_pc_core_gate_pass,
            "slice_matrix_pass": slice_matrix_pass,
            "overall_pass": current_pc_core_gate_pass,
            "current_pc_core_gate_pass": current_pc_core_gate_pass,
            "ga_gate_pass": current_pc_core_gate_pass,
            "ga_gate_status": ga_gate_status,
            "report_scope": CURRENT_PC_CORE_GATE_SCOPE,
            "overall_scope": CURRENT_PC_CORE_GATE_SCOPE,
            "scene_replay_mode": "full_scene_matrix",
            "scene_results": scene_reports,
            "botified_event_oracle": botified_event_oracle,
            "gaze_attention_oracle": gaze_attention_oracle,
            "gaze_attention_oracle_evaluated": gaze_attention_oracle_evaluated,
            "gaze_attention_oracle_pass": gaze_attention_oracle_pass,
            "oracle_evaluated": oracle_evaluated,
            "oracle_evaluation_passed": botified_event_oracle["passed"],
            "not_covered": _full_scene_matrix_not_covered(
                slice_matrix_pass=slice_matrix_pass,
                oracle_evaluated=oracle_evaluated,
                oracle_contracts_present=oracle_contracts_present,
                gaze_attention_oracle_evaluated=gaze_attention_oracle_evaluated,
                gaze_attention_oracle_contracts_present=(
                    gaze_attention_oracle_contracts_present
                ),
            ),
            "non_blocking_gaps": list(FULL_MATRIX_NON_BLOCKING_GAPS),
            "gates": _full_scene_matrix_gates(
                scene_reports=scene_reports,
                current_pc_core_gate_pass=current_pc_core_gate_pass,
                slice_matrix_pass=slice_matrix_pass,
                oracle_evaluated=oracle_evaluated,
                oracle_pass=oracle_pass,
                gaze_attention_oracle_evaluated=gaze_attention_oracle_evaluated,
                gaze_attention_oracle_pass=gaze_attention_oracle_pass,
                ga_gate_status=ga_gate_status,
            ),
        }
    )
    _write_report(config.out, report)
    print(str(config.out))
    return 0 if current_pc_core_gate_pass else 1


def _full_scene_matrix_not_covered(
    *,
    slice_matrix_pass: bool,
    oracle_evaluated: bool,
    oracle_contracts_present: bool,
    gaze_attention_oracle_evaluated: bool,
    gaze_attention_oracle_contracts_present: bool,
) -> list[str]:
    gaps: list[str] = []
    if not slice_matrix_pass:
        gaps.append("full_scene_matrix")
    if not oracle_evaluated or not oracle_contracts_present:
        gaps.append("oracle")
    if (
        not gaze_attention_oracle_evaluated
        or not gaze_attention_oracle_contracts_present
    ):
        gaps.append("gaze_attention_oracle")
    return gaps


def _full_scene_matrix_gates(
    *,
    scene_reports: list[dict[str, Any]],
    current_pc_core_gate_pass: bool,
    slice_matrix_pass: bool,
    oracle_evaluated: bool,
    oracle_pass: bool,
    gaze_attention_oracle_evaluated: bool,
    gaze_attention_oracle_pass: bool,
    ga_gate_status: str,
) -> dict[str, Any]:
    return {
        "current_pc_core": {
            "scope": "full_scene_all_scenes_stationary_oracle",
            "pass": current_pc_core_gate_pass,
            "scene_count": len(scene_reports),
            "frame_count": _full_scene_matrix_frame_count(scene_reports),
            "slice_matrix_pass": slice_matrix_pass,
            "oracle_evaluated": oracle_evaluated,
            "oracle_pass": oracle_pass,
            "gaze_attention_oracle_evaluated": gaze_attention_oracle_evaluated,
            "gaze_attention_oracle_pass": gaze_attention_oracle_pass,
            "stdout_pollution_count": _full_scene_matrix_stdout_pollution_count(
                scene_reports
            ),
            "fresh_gaze_hz_pass": _full_scene_matrix_fresh_gaze_hz_pass(
                scene_reports
            ),
        },
        "ga": {
            "scope": "pc_simulated_ga",
            "pass": current_pc_core_gate_pass,
            "status": ga_gate_status,
        },
    }


def _full_scene_matrix_frame_count(scene_reports: list[dict[str, Any]]) -> int:
    return sum(
        _int_count(result.get("selected_scene_frame_count")) for result in scene_reports
    )


def _full_scene_matrix_stdout_pollution_count(
    scene_reports: list[dict[str, Any]],
) -> int:
    total = 0
    for result in scene_reports:
        botified_stdout = result.get("botified_stdout")
        if isinstance(botified_stdout, dict):
            total += _int_count(botified_stdout.get("pollution_count"))
    return total


def _full_scene_matrix_fresh_gaze_hz_pass(
    scene_reports: list[dict[str, Any]],
) -> bool | None:
    if not scene_reports:
        return False

    saw_rate = False
    for result in scene_reports:
        gaze = result.get("gaze")
        if not isinstance(gaze, dict):
            return None
        fresh_hz = gaze.get("fresh_gaze_publish_hz")
        if not isinstance(fresh_hz, dict):
            return None
        rate_pass = fresh_hz.get("pass")
        if rate_pass is not True:
            return False
        saw_rate = True
    return True if saw_rate else None


def _config_for_full_scene(
    base_config: CliLocalE2EConfig,
    scene_name: str,
) -> CliLocalE2EConfig:
    scene_dir = base_config.data_dir / scene_name
    if not scene_dir.exists():
        raise PreflightError(f"scene directory not found: {scene_dir}")
    if not scene_dir.is_dir():
        raise PreflightError(f"scene is not a directory: {scene_dir}")
    frame_count = len(cli_local_e2e_manifest.jpeg_files(scene_dir))
    if frame_count <= 0:
        raise PreflightError(f"scene has no JPEG frames: {scene_name}")
    head_state_segments = tuple(
        replace(segment, frame_count=frame_count)
        for segment in base_config.head_state_segments
    )
    return replace(
        base_config,
        selected_scene=scene_name,
        scene_dir=scene_dir,
        scene_replay_mode="full_scene",
        selected_scene_frame_count=frame_count,
        frame_count=frame_count,
        gaze_duration_ms=_full_scene_gaze_duration_ms(
            frame_count=frame_count,
            image_hz=base_config.image_hz,
            settle_grace_s=base_config.startup_grace_s,
        )
        if base_config.gaze_collection_mode == "duration"
        else base_config.gaze_duration_ms,
        head_state_segments=head_state_segments,
    )


def _matrix_scene_result(scene_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": scene_report.get("selected_scene"),
        "selected_scene_frame_count": scene_report.get("selected_scene_frame_count"),
        "published_frames": scene_report.get("published_frames"),
        "slice_pass": scene_report.get("slice_pass"),
        "failure_reasons": scene_report.get("failure_reasons", []),
        "gaze": scene_report.get("gaze"),
        "head_state": scene_report.get("head_state"),
        "warmup": scene_report.get("warmup"),
        "botified_stdout": scene_report.get("botified_stdout"),
        "report": scene_report,
    }


def _should_evaluate_botified_event_oracle(config: CliLocalE2EConfig) -> bool:
    return (
        len(config.head_state_segments) == 1
        and config.head_state_segments[0].state == "stationary"
    )


def _unevaluated_gaze_attention_oracle() -> dict[str, Any]:
    return {
        "evaluated": False,
        "passed": None,
        "scenes": [],
    }


def _attention_oracle_contract(effective_manifest: Any) -> dict[str, Any] | None:
    if not isinstance(effective_manifest, dict):
        return None
    oracle = effective_manifest.get("oracle")
    if not isinstance(oracle, dict):
        return None
    timeline = oracle.get("expected_attention_target_timeline")
    if not isinstance(timeline, dict):
        return None
    scenes = timeline.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        return None
    return scenes


def _evaluate_gaze_attention_oracle(
    scene_reports: list[dict[str, Any]],
    scene_contracts: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    scenes: list[dict[str, Any]] = []
    failure_reasons: list[str] = []

    for scene_report in scene_reports:
        scene = str(scene_report.get("scene") or "")
        raw_windows = scene_contracts.get(scene)
        contract_present = isinstance(raw_windows, list) and bool(raw_windows)
        if not contract_present:
            failure_reasons.append(
                f"gaze_attention_oracle_missing_scene_contract:{scene}"
            )
            scenes.append(
                {
                    "scene": scene,
                    "contract_present": False,
                    "windows": 0,
                    "matched_windows": 0,
                    "mismatches": [],
                }
            )
            continue

        gaze = scene_report.get("gaze")
        if not isinstance(gaze, dict):
            gaze = {}
        accepted_samples = _gaze_sample_dicts(gaze.get("accepted_samples"))
        fresh_samples = _fresh_gaze_samples(accepted_samples)
        scene_diagnostics = _evaluate_gaze_attention_scene(
            scene=scene,
            windows=raw_windows,
            accepted_samples=fresh_samples,
            dwell_samples=accepted_samples,
        )
        if scene_diagnostics["mismatches"]:
            failure_reasons.append(f"gaze_attention_oracle_mismatch:{scene}")
        scenes.append(scene_diagnostics)

    return (
        {
            "evaluated": True,
            "passed": not failure_reasons,
            "scenes": scenes,
        },
        failure_reasons,
    )


def _evaluate_gaze_attention_scene(
    *,
    scene: str,
    windows: list[Any],
    accepted_samples: list[dict[str, Any]],
    dwell_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    window_diagnostics: list[dict[str, Any]] = []
    matched_windows = 0

    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            mismatches.append({"window_index": index, "reason": "invalid_window"})
            continue
        start_ms = _finite_float(window.get("start_frame_timestamp_ms"))
        end_ms = _finite_float(window.get("end_frame_timestamp_ms"))
        if start_ms is None or end_ms is None or end_ms < start_ms:
            mismatches.append({"window_index": index, "reason": "invalid_window"})
            continue

        samples = [
            sample
            for sample in accepted_samples
            if _sample_frame_timestamp_in_window(sample, start_ms, end_ms)
        ]
        if window.get("no_target") is True:
            tracking_samples = [
                sample for sample in samples if _is_tracking_gaze_sample(sample)
            ]
            if tracking_samples:
                mismatches.append(
                    {
                        "window_index": index,
                        "reason": "tracking_during_no_target",
                        "sample_count": len(samples),
                        "tracking_count": len(tracking_samples),
                    }
                )
            else:
                matched_windows += 1
            continue

        allowed_track_ids = _allowed_attention_track_ids(window)
        if not allowed_track_ids:
            mismatches.append(
                {"window_index": index, "reason": "missing_expected_target"}
            )
            continue

        matching_samples = [
            sample
            for sample in samples
            if _is_tracking_gaze_sample(sample)
            and _sample_track_id(sample) in allowed_track_ids
            and _sample_matches_optional_attention_point(sample, window)
        ]
        if matching_samples:
            matched_windows += 1
            norm_diagnostics, norm_mismatches = _target_norm_window_diagnostics(
                window_index=index,
                matching_samples=matching_samples,
            )
            window_diagnostics.append(norm_diagnostics)
            mismatches.extend(norm_mismatches)
        else:
            window_diagnostics.append(
                _empty_target_norm_window_diagnostics(window_index=index)
            )
            mismatches.append(
                {
                    "window_index": index,
                    "reason": "target_not_observed",
                    "sample_count": len(samples),
                    "allowed_target_track_ids": sorted(allowed_track_ids),
                }
            )

    switch_dwell_diagnostics, switch_dwell_mismatches = (
        _target_switch_dwell_diagnostics(
            accepted_samples if dwell_samples is None else dwell_samples
        )
    )
    mismatches.extend(switch_dwell_mismatches)

    return {
        "scene": scene,
        "contract_present": True,
        "windows": len(windows),
        "matched_windows": matched_windows,
        "window_diagnostics": window_diagnostics,
        "diagnostics": {
            "target_switch_dwell": switch_dwell_diagnostics,
        },
        "mismatches": mismatches,
    }


def _empty_target_norm_window_diagnostics(*, window_index: int) -> dict[str, Any]:
    return {
        "window_index": window_index,
        "matching_sample_count": 0,
        "target_norm_consistency": {
            "checked": False,
            "passed": None,
            "invalid_count": 0,
            "tolerance": TARGET_NORM_CONSISTENCY_TOLERANCE,
        },
        "target_norm_jitter": {
            "available": False,
            "sample_count": 0,
            "p95": None,
            "max_p95": MAX_PC_GA_TARGET_NORM_JITTER_P95,
            "passed": None,
        },
    }


def _target_norm_window_diagnostics(
    *,
    window_index: int,
    matching_samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    invalid_norms: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(matching_samples):
        invalid_reason = _target_norm_inconsistency_reason(sample)
        if invalid_reason is not None:
            invalid_norms.append(
                {
                    "sample_index": sample_index,
                    "reason": invalid_reason,
                }
            )

    if invalid_norms:
        mismatches.append(
            {
                "window_index": window_index,
                "reason": "target_norm_inconsistent",
                "sample_count": len(matching_samples),
                "invalid_count": len(invalid_norms),
                "invalid_samples": invalid_norms,
            }
        )

    jitter_summary = _target_norm_jitter_summary(matching_samples)
    if jitter_summary["passed"] is False:
        mismatches.append(
            {
                "window_index": window_index,
                "reason": "target_norm_jitter_above_max",
                "sample_count": jitter_summary["sample_count"],
                "p95": jitter_summary["p95"],
                "max_p95": jitter_summary["max_p95"],
            }
        )

    diagnostics = {
        "window_index": window_index,
        "matching_sample_count": len(matching_samples),
        "target_norm_consistency": {
            "checked": True,
            "passed": not invalid_norms,
            "invalid_count": len(invalid_norms),
            "tolerance": TARGET_NORM_CONSISTENCY_TOLERANCE,
        },
        "target_norm_jitter": jitter_summary,
    }
    return diagnostics, mismatches


def _target_norm_inconsistency_reason(sample: dict[str, Any]) -> str | None:
    target_u = _finite_float(sample.get("target_u"))
    target_v = _finite_float(sample.get("target_v"))
    target_norm_x = _finite_float(sample.get("target_norm_x"))
    target_norm_y = _finite_float(sample.get("target_norm_y"))
    image_width = _finite_float(sample.get("image_width"))
    image_height = _finite_float(sample.get("image_height"))
    if (
        target_u is None
        or target_v is None
        or target_norm_x is None
        or target_norm_y is None
        or image_width is None
        or image_height is None
    ):
        return "missing_or_non_finite_fields"
    if image_width <= 0 or image_height <= 0:
        return "invalid_image_size"
    if not (0 <= target_u <= image_width) or not (0 <= target_v <= image_height):
        return "target_pixel_out_of_range"
    if not (-0.5 <= target_norm_x <= 0.5) or not (-0.5 <= target_norm_y <= 0.5):
        return "target_norm_out_of_range"
    expected_norm_x = target_u / image_width - 0.5
    expected_norm_y = target_v / image_height - 0.5
    if (
        abs(target_norm_x - expected_norm_x) > TARGET_NORM_CONSISTENCY_TOLERANCE
        or abs(target_norm_y - expected_norm_y) > TARGET_NORM_CONSISTENCY_TOLERANCE
    ):
        return "target_norm_formula_mismatch"
    return None


def _target_norm_jitter_summary(
    matching_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_count = len(matching_samples)
    if sample_count < 2:
        return {
            "available": False,
            "sample_count": sample_count,
            "p95": None,
            "max_p95": MAX_PC_GA_TARGET_NORM_JITTER_P95,
            "passed": None,
        }
    first_norm_x = _finite_float(matching_samples[0].get("target_norm_x"))
    first_norm_y = _finite_float(matching_samples[0].get("target_norm_y"))
    if first_norm_x is None or first_norm_y is None:
        return {
            "available": False,
            "sample_count": sample_count,
            "p95": None,
            "max_p95": MAX_PC_GA_TARGET_NORM_JITTER_P95,
            "passed": None,
        }
    distances: list[float] = []
    for sample in matching_samples:
        target_norm_x = _finite_float(sample.get("target_norm_x"))
        target_norm_y = _finite_float(sample.get("target_norm_y"))
        if target_norm_x is None or target_norm_y is None:
            return {
                "available": False,
                "sample_count": sample_count,
                "p95": None,
                "max_p95": MAX_PC_GA_TARGET_NORM_JITTER_P95,
                "passed": None,
            }
        distances.append(
            max(
                abs(target_norm_x - first_norm_x),
                abs(target_norm_y - first_norm_y),
            )
        )
    p95 = _percentile_nearest_rank(distances, 95)
    return {
        "available": True,
        "sample_count": sample_count,
        "p95": p95,
        "max_p95": MAX_PC_GA_TARGET_NORM_JITTER_P95,
        "passed": p95 <= MAX_PC_GA_TARGET_NORM_JITTER_P95,
    }


def _target_switch_dwell_diagnostics(
    accepted_samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    unavailable_transition_count = 0
    transition_count = 0
    current_run: dict[str, Any] | None = None

    for sample in accepted_samples:
        track_id = _sample_track_id(sample) if _is_tracking_gaze_sample(sample) else None
        timestamp_ms = _sample_frame_or_publish_timestamp_ms(sample)
        if track_id is None:
            if current_run is not None:
                runs.append(current_run)
                current_run = None
            continue

        if current_run is None:
            current_run = _new_target_switch_run(track_id, timestamp_ms)
            continue

        if current_run["target_track_id"] == track_id:
            current_run["sample_count"] += 1
            if timestamp_ms is None:
                current_run["timestamp_unavailable"] = True
            else:
                current_run["last_timestamp_ms"] = timestamp_ms
            continue

        runs.append(current_run)
        transition_count += 1
        duration_ms = _target_switch_duration_ms(current_run, timestamp_ms)
        if duration_ms is None:
            unavailable_transition_count += 1
        elif duration_ms < MIN_PC_GA_ATTENTION_SWITCH_CONFIRM_MS:
            violations.append(
                {
                    "reason": "target_switch_dwell_below_min",
                    "from_target_track_id": current_run["target_track_id"],
                    "to_target_track_id": track_id,
                    "duration_ms": duration_ms,
                    "min_duration_ms": MIN_PC_GA_ATTENTION_SWITCH_CONFIRM_MS,
                }
            )
        current_run = _new_target_switch_run(track_id, timestamp_ms)

    if current_run is not None:
        runs.append(current_run)

    diagnostics = {
        "available": transition_count > 0 and unavailable_transition_count == 0,
        "run_count": len(runs),
        "transition_count": transition_count,
        "unavailable_transition_count": unavailable_transition_count,
        "min_duration_ms": MIN_PC_GA_ATTENTION_SWITCH_CONFIRM_MS,
        "violations": violations,
    }
    return diagnostics, violations


def _new_target_switch_run(
    target_track_id: int,
    timestamp_ms: float | None,
) -> dict[str, Any]:
    return {
        "target_track_id": target_track_id,
        "sample_count": 1,
        "first_timestamp_ms": timestamp_ms,
        "last_timestamp_ms": timestamp_ms,
        "timestamp_unavailable": timestamp_ms is None,
    }


def _target_switch_duration_ms(
    previous_run: dict[str, Any],
    next_first_timestamp_ms: float | None,
) -> float | None:
    if previous_run.get("timestamp_unavailable") is True:
        return None
    previous_first_timestamp_ms = _finite_float(previous_run.get("first_timestamp_ms"))
    if previous_first_timestamp_ms is None or next_first_timestamp_ms is None:
        return None
    if next_first_timestamp_ms < previous_first_timestamp_ms:
        return None
    return next_first_timestamp_ms - previous_first_timestamp_ms


def _sample_frame_or_publish_timestamp_ms(sample: dict[str, Any]) -> float | None:
    frame_timestamp_ms = _finite_float(sample.get("frame_timestamp_ms"))
    if frame_timestamp_ms is not None:
        return frame_timestamp_ms
    return _finite_float(sample.get("publish_timestamp_ms"))


def _gaze_sample_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [sample for sample in value if isinstance(sample, dict)]


def _fresh_gaze_samples(value: Any) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample in value:
        if sample.get("state") != "stale":
            samples.append(sample)
    return samples


def _sample_frame_timestamp_in_window(
    sample: dict[str, Any],
    start_ms: float,
    end_ms: float,
) -> bool:
    timestamp_ms = _finite_float(sample.get("frame_timestamp_ms"))
    return timestamp_ms is not None and start_ms <= timestamp_ms <= end_ms


def _is_tracking_gaze_sample(sample: dict[str, Any]) -> bool:
    return sample.get("valid") is True and sample.get("state") == "tracking"


def _sample_track_id(sample: dict[str, Any]) -> int | None:
    value = sample.get("target_track_id")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _allowed_attention_track_ids(window: dict[str, Any]) -> set[int]:
    allowed: set[int] = set()
    target_track_id = window.get("target_track_id")
    if isinstance(target_track_id, int) and not isinstance(target_track_id, bool):
        allowed.add(target_track_id)
    raw_allowed = window.get("allowed_target_track_ids")
    if isinstance(raw_allowed, list):
        for item in raw_allowed:
            if isinstance(item, int) and not isinstance(item, bool):
                allowed.add(item)
    return allowed


def _sample_matches_optional_attention_point(
    sample: dict[str, Any],
    window: dict[str, Any],
) -> bool:
    expected_u = _finite_float(window.get("target_u"))
    expected_v = _finite_float(window.get("target_v"))
    tolerance_px = _finite_float(window.get("tolerance_px"))
    if expected_u is None and expected_v is None:
        return True
    if tolerance_px is None or tolerance_px < 0:
        return False
    sample_u = _finite_float(sample.get("target_u"))
    sample_v = _finite_float(sample.get("target_v"))
    if expected_u is not None and (sample_u is None or abs(sample_u - expected_u) > tolerance_px):
        return False
    if expected_v is not None and (sample_v is None or abs(sample_v - expected_v) > tolerance_px):
        return False
    return True


def _gaze_attention_oracle_contracts_present(
    gaze_attention_oracle: dict[str, Any],
) -> bool:
    scenes = gaze_attention_oracle.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return False
    return all(
        isinstance(scene, dict) and scene.get("contract_present") is True
        for scene in scenes
    )


def _unevaluated_botified_event_oracle() -> dict[str, Any]:
    return {
        "evaluated": False,
        "passed": None,
        "scenes": [],
    }


def _evaluate_botified_event_oracle(
    scene_reports: list[dict[str, Any]],
    *,
    head_motion: str,
) -> tuple[dict[str, Any], list[str]]:
    scenes: list[dict[str, Any]] = []
    failure_reasons: list[str] = []

    for scene_report in scene_reports:
        scene = str(scene_report.get("scene") or "")
        botified_stdout = scene_report.get("botified_stdout")
        if not isinstance(botified_stdout, dict):
            botified_stdout = {}
        observed = _int_event_counts(botified_stdout.get("event_counts"))
        sequence = _event_sequence(botified_stdout.get("event_sequence"))
        facts = replay_val_data.botified_event_oracle_facts(scene, head_motion)
        required = list(facts["required_events"])
        forbidden = list(facts["forbidden_events"])
        order_requirements = list(facts["order_requirements"])
        duplicate_greeting_contracts = list(facts["duplicate_greeting_contracts"])
        contract_present = bool(
            required
            or forbidden
            or order_requirements
            or duplicate_greeting_contracts
        )

        missing = [event for event in required if observed.get(event, 0) <= 0]
        forbidden_present = {
            event: observed[event]
            for event in forbidden
            if observed.get(event, 0) > 0
        }
        order_violations = _botified_event_order_violations(
            sequence,
            order_requirements,
        )
        duplicate_greeting_violations = replay_val_data.duplicate_greeting_violations(
            scene=scene,
            head_motion=head_motion,
            event_records=sequence,
        )

        for event in missing:
            failure_reasons.append(f"botified_event_oracle_missing:{scene}:{event}")
        for event in forbidden_present:
            failure_reasons.append(f"botified_event_oracle_forbidden:{scene}:{event}")
        for violation in order_violations:
            failure_reasons.append(
                "botified_event_oracle_order:"
                f"{scene}:{violation['before_event']}_before_{violation['after_event']}"
            )
        for violation in duplicate_greeting_violations:
            failure_reasons.append(
                "botified_event_oracle_duplicate_greeting:"
                f"{scene}:{violation['person_label']}:{violation['event']}"
            )
        if not contract_present:
            failure_reasons.append(f"botified_event_oracle_missing_scene_contract:{scene}")

        scenes.append(
            {
                "scene": scene,
                "contract_present": contract_present,
                "observed": observed,
                "required": required,
                "missing": missing,
                "forbidden_present": forbidden_present,
                "order_violations": order_violations,
                "duplicate_greeting_violations": duplicate_greeting_violations,
            }
        )

    return (
        {
            "evaluated": True,
            "passed": not failure_reasons,
            "scenes": scenes,
        },
        failure_reasons,
    )


def _botified_event_oracle_contracts_present(
    botified_event_oracle: dict[str, Any],
) -> bool:
    scenes = botified_event_oracle.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return False
    return all(
        isinstance(scene, dict) and scene.get("contract_present") is True
        for scene in scenes
    )


def _int_event_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for event, count in value.items():
        if isinstance(event, str) and isinstance(count, int) and not isinstance(count, bool):
            counts[event] = count
    return dict(sorted(counts.items()))


def _event_sequence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sequence: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        line = item.get("line")
        event = item.get("event")
        if isinstance(line, int) and not isinstance(line, bool) and isinstance(event, str):
            sequence_item: dict[str, Any] = {"line": line, "event": event}
            event_id = item.get("event_id")
            if isinstance(event_id, str):
                sequence_item["event_id"] = event_id
            track_id = item.get("track_id")
            if isinstance(track_id, int) and not isinstance(track_id, bool):
                sequence_item["track_id"] = track_id
            frame = item.get("frame")
            if isinstance(frame, int) and not isinstance(frame, bool):
                sequence_item["frame"] = frame
            sequence.append(sequence_item)
    return sequence


def _botified_event_order_violations(
    sequence: list[dict[str, Any]],
    order_requirements: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    first_line_by_event: dict[str, int] = {}
    for item in sequence:
        first_line_by_event.setdefault(item["event"], item["line"])

    violations: list[dict[str, Any]] = []
    for before_event, after_event in order_requirements:
        before_line = first_line_by_event.get(before_event)
        after_line = first_line_by_event.get(after_event)
        if before_line is None or after_line is None or before_line < after_line:
            continue
        violations.append(
            {
                "before_event": before_event,
                "after_event": after_event,
                "before_line": before_line,
                "after_line": after_line,
            }
        )
    return violations


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
        "--log-jsonl",
        os.fspath(_cli_frame_request_log_path(config)),
    ]


def _cli_frame_request_log_path(config: CliLocalE2EConfig) -> Path:
    return config.out.with_suffix(".cli-frame-requests.jsonl")


def _reset_cli_frame_request_log(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _read_cli_frame_request_log(
    path: Path,
    *,
    line_offset: int = 0,
) -> CliFrameRequestLogRead:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        next_line_offset = max(0, int(line_offset))
        return CliFrameRequestLogRead(
            records=(),
            parse_error_count=0,
            parse_errors=(),
            line_count=0,
            next_line_offset=next_line_offset,
        )

    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    start = max(0, int(line_offset))
    selected_lines = lines[start:]
    for index, line in enumerate(selected_lines, start=start + 1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append({"line": index, "error": exc.msg})
            continue
        if not isinstance(payload, dict):
            parse_errors.append({"line": index, "error": "expected JSON object"})
            continue
        if payload.get("type") == "frame_request":
            records.append(payload)
    return CliFrameRequestLogRead(
        records=tuple(records),
        parse_error_count=len(parse_errors),
        parse_errors=tuple(parse_errors),
        line_count=len(selected_lines),
        next_line_offset=len(lines),
    )


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
    command = _wrapper_common_command(config, "subscribe_test_gaze_targets.py")
    if config.gaze_collection_mode == "duration":
        command.extend(
            [
                "--duration-ms",
                str(config.gaze_duration_ms),
                "--min-count",
                str(config.gaze_count),
            ]
        )
    else:
        command.extend(
            [
                "--count",
                str(config.gaze_count),
                "--timeout-ms",
                str(config.gaze_timeout_ms),
            ]
        )
    command.extend(["--gaze-topic", config.gaze_topic])
    return command


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


def _head_publisher_command(config: CliLocalE2EConfig, segment: HeadStateSegment) -> list[str]:
    return [
        *_wrapper_common_command(config, "publish_test_head_state.py"),
        "--state",
        segment.state,
        "--count",
        str(segment.frame_count),
        "--hz",
        _format_number(segment.head_state_hz),
        "--head-state-topic",
        config.head_state_topic,
    ]


def _parse_head_state_segment_names(
    *,
    head_state: str | None,
    head_state_segments: str | None,
) -> tuple[str, ...]:
    if head_state_segments is None:
        if head_state is not None:
            return (head_state,)
        return DEFAULT_HEAD_STATE_SEGMENTS

    raw_segments = head_state_segments.split(",")
    if not raw_segments or any(segment.strip() == "" for segment in raw_segments):
        raise PreflightError("--head-state-segments must not contain empty values")

    segments = tuple(segment.strip() for segment in raw_segments)
    unknown = [segment for segment in segments if segment not in VALID_HEAD_STATES]
    if unknown:
        allowed = ",".join(VALID_HEAD_STATES)
        raise PreflightError(
            f"--head-state-segments contains unsupported value {unknown[0]!r}; "
            f"allowed values: {allowed}"
        )
    if len(set(segments)) != len(segments):
        raise PreflightError("--head-state-segments must not contain duplicate values")
    return segments


def _publisher_timeout_s(frame_count: int, hz: float) -> float:
    return max(STOP_TIMEOUT_S, (frame_count / hz) + STOP_TIMEOUT_S)


def _full_scene_gaze_duration_ms(
    *,
    frame_count: int,
    image_hz: float,
    settle_grace_s: float,
) -> int:
    duration_s = (frame_count / image_hz) + settle_grace_s
    return max(1, math.ceil(duration_s * 1000.0))


def _gaze_subscriber_timeout_s(config: CliLocalE2EConfig) -> float:
    if config.gaze_collection_mode == "duration":
        duration_ms = config.gaze_duration_ms or 0
        return (duration_ms / 1000.0) + STOP_TIMEOUT_S
    return (config.gaze_timeout_ms / 1000.0) + STOP_TIMEOUT_S


def _parse_head_publisher_stdout(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return None, "expected exactly one head publisher JSON summary line"
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "head publisher JSON summary is not an object"
    return payload, None


def _head_segment_failure_reasons(
    *,
    segment: HeadStateSegment,
    result: CommandResult,
    stdout_json: dict[str, Any] | None,
    stdout_parse_error: str | None,
) -> list[str]:
    reasons: list[str] = []
    suffix = f":segment={segment.state}"
    if result.returncode != 0:
        reasons.append(f"head_publisher_failed{suffix}")
    if stdout_parse_error is not None:
        reasons.append(f"head_publisher_malformed_json{suffix}")
        return reasons
    if stdout_json is None:
        reasons.append(f"head_publisher_malformed_json{suffix}")
        return reasons

    if stdout_json.get("state") != segment.state:
        reasons.append(f"head_publisher_state_mismatch{suffix}")
    published = stdout_json.get("published")
    if isinstance(published, bool) or not isinstance(published, int):
        reasons.append(f"head_publisher_count_mismatch{suffix}")
    elif published != segment.frame_count:
        reasons.append(f"head_publisher_count_mismatch{suffix}")

    mapped_state = stdout_json.get("mapped_state")
    mapped_valid = stdout_json.get("mapped_valid")
    dds_valid = stdout_json.get("dds_valid")
    if segment.state in {"stationary", "moving"}:
        if mapped_state != segment.state:
            reasons.append(f"head_state_segment_state_mismatch{suffix}")
        if mapped_state == "unknown" or mapped_valid is False:
            reasons.append(f"head_state_unknown_ratio_above_max{suffix}")
    elif segment.state == "unknown":
        has_unknown_evidence = (
            mapped_state == "unknown"
            or mapped_valid is False
            or dds_valid is False
        )
        if not has_unknown_evidence:
            reasons.append(f"head_state_unknown_segment_missing_unknown_evidence{suffix}")
    return reasons


def _head_cli_request_failure_reasons(
    *,
    segment: HeadStateSegment,
    records: tuple[dict[str, Any], ...],
    parse_error_count: int,
) -> list[str]:
    if segment.state != "stationary":
        return []

    suffix = f":segment={segment.state}"
    if parse_error_count > 0:
        return [f"head_state_cli_request_log_parse_error{suffix}"]
    if not records:
        return [f"head_state_cli_request_evidence_missing{suffix}"]

    for record in records:
        head_motion = record.get("head_motion")
        state = (
            head_motion.get("state") if isinstance(head_motion, dict) else None
        )
        if state != "stationary":
            return [f"head_state_cli_unknown_or_stale{suffix}"]
    return []


def _capture_to_gaze_publish_latency_failure_reasons(
    gaze_summary: dict[str, Any],
    *,
    segment_suffix: str,
) -> list[str]:
    if _int_count(gaze_summary.get("accepted_count")) == 0:
        return []
    latency_summary = gaze_summary.get("capture_to_gaze_publish_ms")
    if isinstance(latency_summary, dict):
        if _int_count(latency_summary.get("invalid_sample_count")) > 0:
            return [f"capture_to_gaze_publish_latency_invalid{segment_suffix}"]
        if latency_summary.get("available") is True:
            return []
    return [f"capture_to_gaze_publish_latency_unavailable{segment_suffix}"]


def _build_smoke_report(
    config: CliLocalE2EConfig,
    *,
    commands: dict[str, Any],
    started_at: str,
    elapsed_s: float,
    slice_pass: bool,
    failure_reasons: list[str],
    health_result: HealthCheckResult | None,
    server_process: ProcessLike | None,
    cli_process: ProcessLike | None,
    gaze_process: ProcessLike | None,
    segment_results: list[SegmentSmokeResult],
    gaze_summary: dict[str, Any],
    botified_stdout: dict[str, Any],
    warmup_result: WarmupResult,
) -> dict[str, Any]:
    legacy_segment_result = _legacy_segment_result(segment_results)
    head_result = None if legacy_segment_result is None else legacy_segment_result.head_result
    image_result = None if legacy_segment_result is None else legacy_segment_result.image_result
    gaze_result = None if legacy_segment_result is None else legacy_segment_result.gaze_result
    legacy_gaze_summary = (
        gaze_summary if legacy_segment_result is None else legacy_segment_result.gaze_summary
    )
    head_state_segments = _head_state_segment_reports(segment_results)
    head_state_unknown_ratio = _head_state_unknown_ratio(segment_results)
    head_state_stale_count = _head_state_stale_count(segment_results)
    server_exit_code = _process_exit_code(server_process)
    cli_exit_code = _process_exit_code(cli_process)
    runtime_report = _runtime_provenance_report_with_exit_codes(
        config.runtime_provenance,
        server_exit_code=server_exit_code,
        cli_exit_code=cli_exit_code,
    )
    latency_report = {
        "capture_to_gaze_publish_ms": _aggregate_capture_to_gaze_publish_summary(
            segment_results
        ),
        "capture_to_botified_stdout_ms": _unavailable_latency_summary(),
    }
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
            **_runtime_provenance_flat_aliases(
                runtime_report,
                server_exit_code=server_exit_code,
                cli_exit_code=cli_exit_code,
            ),
            "build_dir": os.fspath(config.build_dir),
            "dds_bridge_bin": os.fspath(config.dds_bridge_bin),
            "server_url": config.server_url.original,
            "healthz_url": config.server_url.healthz_url,
            "cli_request_log_path": os.fspath(_cli_frame_request_log_path(config)),
            "commands": commands,
            "warmup": _warmup_report(warmup_result),
            "processes": {
                "server": _process_report(server_process),
                "cli": _process_report(cli_process),
                "gaze_subscriber": _process_report(gaze_process),
            },
            "returncodes": {
                "head_publisher": None if head_result is None else head_result.returncode,
                "head_publishers": {
                    result.segment.state: None
                    if result.head_result is None
                    else result.head_result.returncode
                    for result in segment_results
                },
                "image_publisher": None if image_result is None else image_result.returncode,
                "image_publishers": {
                    result.segment.state: None
                    if result.image_result is None
                    else result.image_result.returncode
                    for result in segment_results
                },
                "gaze_subscriber": None if gaze_result is None else gaze_result.returncode,
                "gaze_subscribers": {
                    result.segment.state: None
                    if result.gaze_result is None
                    else result.gaze_result.returncode
                    for result in segment_results
                },
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
            "scene_replay_mode": config.scene_replay_mode,
            "selected_scene_frame_count": config.selected_scene_frame_count,
            "published_frames": config.frame_count,
            "image_hz": config.image_hz,
            "head_state": {
                "required": True,
                "publisher_mode": config.head_state_mode,
                "state": config.head_state_segments[0].state,
                "hz": config.head_state_hz,
                "count": config.frame_count,
                "segments": head_state_segments,
                "stale_count": head_state_stale_count,
                "unknown_ratio": head_state_unknown_ratio,
                "evidence_source": "synthetic_publisher_stdout",
                "cli_request_evidence_source": "cli_runtime_jsonl",
                "cli_request_log_path": os.fspath(_cli_frame_request_log_path(config)),
                "evidence_scope": config.scene_replay_mode,
                "partial_smoke_only": config.scene_replay_mode != "full_scene",
            },
            "head_state_publisher_mode": config.head_state_mode,
            "head_state_hz": config.head_state_hz,
            "head_state_stale_count": head_state_stale_count,
            "head_state_unknown_ratio": head_state_unknown_ratio,
            "head_state_segments": [
                segment.state for segment in config.head_state_segments
            ],
            "gaze": legacy_gaze_summary,
            "gaze_segments": [
                {
                    "requested_head_state": result.segment.state,
                    "summary": result.gaze_summary,
                }
                for result in segment_results
            ],
            "botified_stdout": botified_stdout,
            "latency": latency_report,
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
                "head_publishers": {
                    result.segment.state: {
                        "stdout_tail": _tail(
                            ""
                            if result.head_result is None
                            else result.head_result.stdout
                        ),
                        "stderr_tail": _tail(
                            ""
                            if result.head_result is None
                            else result.head_result.stderr
                        ),
                    }
                    for result in segment_results
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
    report["evidence_summary"] = _evidence_summary(report)
    return report


def _warmup_report(warmup_result: WarmupResult) -> dict[str, Any]:
    return {
        "enabled": warmup_result.enabled,
        "passed": warmup_result.passed,
        "failure_reason": warmup_result.failure_reason,
        "count": warmup_result.count,
    }


def _evidence_summary(report: dict[str, Any]) -> dict[str, Any]:
    head_state = report["head_state"]
    return {
        "manifest": {
            "manifest_source": report.get("manifest_source"),
            "manifest_sha256": report.get("manifest_sha256"),
            "manifest_authoritative": report.get("manifest_authoritative"),
            "oracle_schema_present": report.get("oracle_schema_present"),
            "oracle_schema_valid": report.get("oracle_schema_valid"),
        },
        "runtime": {
            "server_bin_is_runtime_venv": report.get("server_bin_is_runtime_venv"),
            "cli_bin_is_runtime_venv": report.get("cli_bin_is_runtime_venv"),
            "runtime_hash": report.get("runtime_hash"),
            "config_hash": report.get("config_hash"),
        },
        "head_state": {
            "required": head_state.get("required"),
            "mode": head_state.get("publisher_mode"),
            "hz": head_state.get("hz"),
            "stale_count": head_state.get("stale_count"),
            "unknown_ratio": head_state.get("unknown_ratio"),
            "segments": head_state.get("segments"),
        },
        "gaze": _gaze_evidence_summary(report.get("gaze_segments", [])),
        "botified_stdout": _botified_stdout_evidence_summary(
            report["botified_stdout"]
        ),
        "latency": report["latency"],
    }


def _gaze_evidence_summary(gaze_segments: Any) -> dict[str, Any]:
    total_expected_count = 0
    total_accepted_count = 0
    total_valid_count = 0
    total_invalid_count = 0
    total_rejected_count = 0
    per_segment_accepted_counts: dict[str, int] = {}
    available_hz: list[float] = []
    saw_available_rate = False
    all_available_rates_pass = True

    if not isinstance(gaze_segments, list):
        gaze_segments = []

    for segment in gaze_segments:
        if not isinstance(segment, dict):
            continue
        state = segment.get("requested_head_state")
        summary = segment.get("summary")
        if not isinstance(summary, dict):
            continue

        accepted_count = _int_count(summary.get("accepted_count"))
        if isinstance(state, str):
            per_segment_accepted_counts[state] = accepted_count
        total_expected_count += _int_count(summary.get("expected_count"))
        total_accepted_count += accepted_count
        total_valid_count += _int_count(summary.get("valid_count"))
        total_invalid_count += _int_count(summary.get("invalid_count"))
        total_rejected_count += _int_count(summary.get("rejected_count"))

        publish_hz = summary.get("fresh_gaze_publish_hz")
        if not isinstance(publish_hz, dict) or publish_hz.get("available") is not True:
            continue
        hz = _finite_float(publish_hz.get("hz"))
        if hz is None:
            continue
        saw_available_rate = True
        available_hz.append(hz)
        if publish_hz.get("pass") is not True:
            all_available_rates_pass = False

    rate_pass = None
    if saw_available_rate:
        rate_pass = all_available_rates_pass

    return {
        "total_expected_count": total_expected_count,
        "total_accepted_count": total_accepted_count,
        "total_valid_count": total_valid_count,
        "total_invalid_count": total_invalid_count,
        "total_rejected_count": total_rejected_count,
        "per_segment_accepted_counts": dict(
            sorted(per_segment_accepted_counts.items())
        ),
        "min_available_gaze_hz": min(available_hz) if available_hz else None,
        "max_available_gaze_hz": max(available_hz) if available_hz else None,
        "rate_pass": rate_pass,
    }


def _botified_stdout_evidence_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_frame_count": _int_count(summary.get("allowed_frame_count")),
        "pollution_count": _int_count(summary.get("pollution_count")),
        "forbidden_event_count": _int_count(summary.get("forbidden_event_count")),
        "parse_error_count": _int_count(summary.get("parse_error_count")),
        "output_limits": summary.get("output_limits"),
    }


def _aggregate_capture_to_gaze_publish_summary(
    segment_results: list[SegmentSmokeResult],
) -> dict[str, Any]:
    samples: list[float] = []
    invalid_sample_count = 0
    for result in segment_results:
        samples.extend(result.gaze_latency_samples_ms)
        latency_summary = result.gaze_summary.get("capture_to_gaze_publish_ms")
        if isinstance(latency_summary, dict):
            invalid_sample_count += _int_count(
                latency_summary.get("invalid_sample_count")
            )
    return _capture_to_gaze_publish_summary(
        samples,
        invalid_sample_count=invalid_sample_count,
    )


def _unavailable_latency_summary() -> dict[str, Any]:
    return {
        "available": False,
        "sample_count": 0,
        "p50": None,
        "p95": None,
        "p99": None,
    }


def _int_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _legacy_segment_result(
    segment_results: list[SegmentSmokeResult],
) -> SegmentSmokeResult | None:
    for result in segment_results:
        if result.failure_reasons:
            return result
    return segment_results[-1] if segment_results else None


def _head_state_segment_reports(
    segment_results: list[SegmentSmokeResult],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for result in segment_results:
        summary = result.head_stdout_json or {}
        reports.append(
            {
                "requested_state": result.segment.state,
                "frame_count": result.segment.frame_count,
                "head_state_hz": result.segment.head_state_hz,
                "publisher_mode": "required",
                "returncode": None
                if result.head_result is None
                else result.head_result.returncode,
                "published": summary.get("published"),
                "state": summary.get("state"),
                "mapped_state": summary.get("mapped_state"),
                "mapped_valid": summary.get("mapped_valid"),
                "dds_valid": summary.get("dds_valid"),
                "head_state_topic": summary.get("head_state_topic"),
                "stale_count": summary.get("stale_count"),
                "cli_frame_request_count": len(result.cli_frame_request_records),
                "cli_frame_request_head_motion_states": [
                    _head_motion_state(record)
                    for record in result.cli_frame_request_records
                ],
                "cli_frame_request_records": list(result.cli_frame_request_records),
                "cli_frame_request_log_parse_error_count": (
                    result.cli_frame_request_log_parse_error_count
                ),
                "cli_frame_request_log_parse_errors": list(
                    result.cli_frame_request_log_parse_errors
                ),
                "cli_frame_request_log_line_count": (
                    result.cli_frame_request_log_line_count
                ),
                "stdout_json": result.head_stdout_json,
                "stdout_parse_error": result.head_stdout_parse_error,
                "failure_reasons": result.failure_reasons,
            }
        )
    return reports


def _head_state_unknown_ratio(
    segment_results: list[SegmentSmokeResult],
) -> float | None:
    unknown_count = 0
    sample_count = 0
    for result in segment_results:
        if result.segment.state == "unknown" or result.head_stdout_json is None:
            continue
        samples = _published_count(result)
        sample_count += samples
        if (
            result.head_stdout_json.get("mapped_state") == "unknown"
            or result.head_stdout_json.get("mapped_valid") is False
        ):
            unknown_count += samples
    if sample_count == 0:
        return None
    return unknown_count / sample_count


def _head_motion_state(record: dict[str, Any]) -> Any:
    head_motion = record.get("head_motion")
    if not isinstance(head_motion, dict):
        return None
    return head_motion.get("state")


def _head_state_stale_count(segment_results: list[SegmentSmokeResult]) -> int | None:
    total = 0
    saw_stale_count = False
    for result in segment_results:
        summary = result.head_stdout_json
        if summary is None:
            continue
        stale_count = summary.get("stale_count")
        if isinstance(stale_count, bool) or not isinstance(stale_count, int):
            continue
        saw_stale_count = True
        total += stale_count
    return total if saw_stale_count else None


def _published_count(result: SegmentSmokeResult) -> int:
    summary = result.head_stdout_json
    if summary is None:
        return result.segment.frame_count
    published = summary.get("published")
    if isinstance(published, bool) or not isinstance(published, int):
        return result.segment.frame_count
    return published


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
            "manifest_schema_version": None,
            "manifest_authoritative": False,
            "manifest_validation_errors": [],
            "oracle_schema_present": False,
            "oracle_schema_valid": False,
            "oracle_summary": None,
            "scene_count": None,
            "frame_count": None,
            "effective_manifest": None,
        }
    manifest_report = _manifest_report_with_contract(
        manifest_report,
        contract_required=bool(getattr(args, "require_authoritative_manifest", False)),
    )
    failure_reasons = [failure_reason]
    validation_errors = manifest_report.get("manifest_validation_errors")
    if (
        isinstance(validation_errors, list)
        and MANIFEST_UNREADABLE_OR_INVALID in validation_errors
        and MANIFEST_UNREADABLE_OR_INVALID not in failure_reasons
    ):
        failure_reasons.append(MANIFEST_UNREADABLE_OR_INVALID)

    report = _base_report(
        manifest_report=manifest_report,
        status="preflight_failed",
        failure_reasons=failure_reasons,
    )
    runtime_report = _runtime_provenance_report_for_failure(args)
    report.update(
        {
            "slice_pass": False,
            "out": os.fspath(out),
            **_runtime_provenance_flat_aliases(
                runtime_report,
                server_exit_code=None,
                cli_exit_code=None,
            ),
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
        "overall_scope": _report_scope_for_status(status),
        "current_pc_core_gate_pass": False,
        "ga_gate_pass": False,
        "ga_gate_status": GA_GATE_STATUS,
        "post_ga_validation_status": "out_of_scope",
        "post_ga_not_covered": list(POST_GA_NOT_COVERED),
        "report_scope": _report_scope_for_status(status),
        "pc_local_e2e_status": status,
        "failure_reasons": failure_reasons,
        "not_covered": list(NOT_COVERED),
        "non_blocking_gaps": [],
        "botified_event_oracle": _unevaluated_botified_event_oracle(),
        "gaze_attention_oracle": _unevaluated_gaze_attention_oracle(),
        "gaze_attention_oracle_evaluated": False,
        "gaze_attention_oracle_pass": False,
        "gates": _default_gates(status),
    }
    for key in MANIFEST_REPORT_KEYS:
        report[key] = manifest_report.get(key)
    return report


def _report_scope_for_status(status: str) -> str:
    if status == "preflight_failed":
        return "preflight"
    if status.startswith("full_scene_matrix_"):
        return CURRENT_PC_CORE_GATE_SCOPE
    return "partial_smoke"


def _default_gates(status: str) -> dict[str, Any]:
    return {
        "current_pc_core": {
            "scope": CURRENT_PC_CORE_GATE_SCOPE,
            "pass": False,
            "status": "not_evaluated" if status != "preflight_failed" else "preflight_failed",
        },
        "ga": {
            "pass": False,
            "scope": "pc_simulated_ga",
            "status": GA_GATE_STATUS,
        },
    }


def _summarize_gaze_jsonl(
    stdout: str,
    logical_camera_name: str,
    expected_count: int,
) -> dict[str, Any]:
    summary, _ = _summarize_gaze_jsonl_with_samples(
        stdout,
        logical_camera_name,
        expected_count,
    )
    return summary


def _summarize_gaze_jsonl_with_samples(
    stdout: str,
    logical_camera_name: str,
    expected_count: int,
) -> tuple[dict[str, Any], tuple[float, ...]]:
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

    latency_samples, invalid_latency_sample_count = _gaze_latency_samples(
        accepted_samples
    )
    fresh_samples = [
        sample for sample in accepted_samples if sample.get("state") != "stale"
    ]
    summary = {
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
        "capture_to_gaze_publish_ms": _capture_to_gaze_publish_summary(
            latency_samples,
            invalid_sample_count=invalid_latency_sample_count,
        ),
        "gaze_publish_hz": _gaze_publish_hz_summary(accepted_samples),
        "fresh_gaze_publish_hz": _gaze_publish_hz_summary(fresh_samples),
        "accepted_samples": accepted_samples,
    }
    return summary, tuple(latency_samples)


def _gaze_latency_samples(
    accepted_samples: list[dict[str, Any]],
) -> tuple[list[float], int]:
    samples: list[float] = []
    invalid_sample_count = 0
    for sample in accepted_samples:
        frame_timestamp_ms = _finite_float(sample.get("frame_timestamp_ms"))
        publish_timestamp_ms = _finite_float(sample.get("publish_timestamp_ms"))
        if (
            frame_timestamp_ms is None
            or publish_timestamp_ms is None
            or publish_timestamp_ms < frame_timestamp_ms
        ):
            invalid_sample_count += 1
            continue
        latency_ms = publish_timestamp_ms - frame_timestamp_ms
        if latency_ms > MAX_CAPTURE_TO_GAZE_LATENCY_MS:
            invalid_sample_count += 1
            continue
        samples.append(latency_ms)
    return samples, invalid_sample_count


def _capture_to_gaze_publish_summary(
    samples: list[float] | tuple[float, ...],
    *,
    invalid_sample_count: int,
) -> dict[str, Any]:
    if not samples:
        return {
            "available": False,
            "sample_count": 0,
            "invalid_sample_count": invalid_sample_count,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "available": True,
        "sample_count": len(samples),
        "invalid_sample_count": invalid_sample_count,
        "p50": _percentile_nearest_rank(samples, 50),
        "p95": _percentile_nearest_rank(samples, 95),
        "p99": _percentile_nearest_rank(samples, 99),
    }


def _gaze_publish_hz_summary(
    accepted_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    publish_timestamps_ms = [
        timestamp
        for timestamp in (
            _finite_float(sample.get("publish_timestamp_ms"))
            for sample in accepted_samples
        )
        if timestamp is not None
    ]
    sample_count = len(publish_timestamps_ms)
    first_timestamp_ms = publish_timestamps_ms[0] if publish_timestamps_ms else None
    last_timestamp_ms = publish_timestamps_ms[-1] if publish_timestamps_ms else None
    timestamps_strictly_increasing = all(
        previous_timestamp_ms < current_timestamp_ms
        for previous_timestamp_ms, current_timestamp_ms in zip(
            publish_timestamps_ms, publish_timestamps_ms[1:]
        )
    )
    if (
        sample_count < 2
        or first_timestamp_ms is None
        or last_timestamp_ms is None
        or last_timestamp_ms <= first_timestamp_ms
        or not timestamps_strictly_increasing
    ):
        return {
            "available": False,
            "sample_count": sample_count,
            "first_publish_timestamp_ms": first_timestamp_ms,
            "last_publish_timestamp_ms": last_timestamp_ms,
            "hz": None,
            "min_hz": MIN_GAZE_PUBLISH_HZ,
            "max_hz": MAX_GAZE_PUBLISH_HZ,
            "pass": None,
        }

    elapsed_s = (last_timestamp_ms - first_timestamp_ms) / 1000.0
    hz = (sample_count - 1) / elapsed_s
    return {
        "available": True,
        "sample_count": sample_count,
        "first_publish_timestamp_ms": first_timestamp_ms,
        "last_publish_timestamp_ms": last_timestamp_ms,
        "hz": hz,
        "min_hz": MIN_GAZE_PUBLISH_HZ,
        "max_hz": MAX_GAZE_PUBLISH_HZ,
        "pass": MIN_GAZE_PUBLISH_HZ <= hz <= MAX_GAZE_PUBLISH_HZ,
    }


def _percentile_nearest_rank(
    samples: list[float] | tuple[float, ...],
    percentile: int,
) -> float:
    ordered = sorted(samples)
    rank = math.ceil((percentile / 100.0) * len(ordered))
    index = min(max(rank - 1, 0), len(ordered) - 1)
    return float(ordered[index])


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    sample = float(value)
    if not math.isfinite(sample):
        return None
    return sample


def _summarize_botified_stdout_from_process(process: ProcessLike | None) -> dict[str, Any]:
    lines = _collected_stream_lines(process, "stdout")
    line_timestamps_ms = _collected_stdout_line_timestamps_ms(process)
    line_count = _collected_stream_line_count(process, "stdout", fallback=len(lines))
    collection_truncated = _collection_truncated(process, "stdout", collected_count=len(lines))
    collection_incomplete = bool(getattr(process, "collection_incomplete", False))
    return _summarize_botified_stdout(
        lines,
        line_timestamps_ms=line_timestamps_ms,
        line_count=line_count,
        collection_truncated=collection_truncated,
        collection_incomplete=collection_incomplete,
    )


def _summarize_botified_stdout(
    lines: list[str],
    *,
    line_timestamps_ms: list[int] | None = None,
    line_count: int,
    collection_truncated: bool,
    collection_incomplete: bool,
) -> dict[str, Any]:
    frame_count = 0
    allowed_frame_count = 0
    pollution_count = 0
    parse_error_count = 0
    forbidden_event_count = 0
    event_counts: dict[str, int] = {}
    event_sequence: list[dict[str, Any]] = []
    first_frame: dict[str, Any] | None = None
    last_frame: dict[str, Any] | None = None
    contract_violations: list[str] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        if BOTIFIED_OPEN not in line and BOTIFIED_CLOSE not in line:
            pollution_count += 1
            contract_violations.append(f"line {line_number}: stdout pollution")
            continue

        if not _is_wrapped_botified_frame(line):
            parse_error_count += 1
            contract_violations.append(f"line {line_number}: malformed Botified frame")
            continue

        payload = _parse_botified_payload(line)
        if payload is None:
            parse_error_count += 1
            contract_violations.append(f"line {line_number}: invalid Botified JSON payload")
            continue

        event, contract_violation = _botified_contract_event(payload)
        if contract_violation is not None:
            parse_error_count += 1
            contract_violations.append(f"line {line_number}: {contract_violation}")
            continue
        if event is None:
            parse_error_count += 1
            contract_violations.append(f"line {line_number}: invalid Botified contract")
            continue

        frame_count += 1
        frame = {
            "line": line_number,
            "event": event,
            "payload": payload,
        }
        if first_frame is None:
            first_frame = frame
        last_frame = frame

        if event not in BOTIFIED_ALLOWED_EVENTS:
            forbidden_event_count += 1
            contract_violations.append(f"line {line_number}: forbidden Botified event {event}")
            continue

        allowed_frame_count += 1
        event_counts[event] = event_counts.get(event, 0) + 1
        sequence_item: dict[str, Any] = {"line": line_number, "event": event}
        event_id = _botified_event_id(payload)
        if event_id is not None:
            sequence_item["event_id"] = event_id
        track_id = _botified_track_id(payload)
        if track_id is not None:
            sequence_item["track_id"] = track_id
        received_monotonic_ms = _botified_line_timestamp_ms(
            line_timestamps_ms,
            line_number,
        )
        if received_monotonic_ms is not None:
            sequence_item["received_monotonic_ms"] = received_monotonic_ms
        event_sequence.append(sequence_item)

    output_limits = _botified_stdout_output_limits(
        event_sequence,
        timestamps_available=line_timestamps_ms is not None,
    )

    return {
        "source": "cli_stdout",
        "required_frame_count": None,
        "line_count": int(line_count),
        "frame_count": frame_count,
        "allowed_frame_count": allowed_frame_count,
        "pollution_count": pollution_count,
        "parse_error_count": parse_error_count,
        "forbidden_event_count": forbidden_event_count,
        "event_counts": dict(sorted(event_counts.items())),
        "event_sequence": event_sequence,
        "output_limits": output_limits,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "collection_truncated": collection_truncated,
        "collection_incomplete": collection_incomplete,
        "contract_violations": contract_violations,
    }


def _botified_line_timestamp_ms(
    line_timestamps_ms: list[int] | None,
    line_number: int,
) -> int | None:
    if line_timestamps_ms is None:
        return None
    index = line_number - 1
    if index < 0 or index >= len(line_timestamps_ms):
        return None
    timestamp = line_timestamps_ms[index]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        return None
    return timestamp


def _botified_stdout_output_limits(
    event_sequence: list[dict[str, Any]],
    *,
    timestamps_available: bool,
) -> dict[str, Any]:
    timed_events = _botified_timed_events(event_sequence)
    available = timestamps_available and len(timed_events) == len(event_sequence)
    return {
        "per_track_event_60s": _botified_output_limit_summary(
            timed_events,
            threshold=BOTIFIED_OUTPUT_LIMIT_PER_TRACK_EVENT_60S_THRESHOLD,
            window_ms=BOTIFIED_OUTPUT_LIMIT_60S_WINDOW_MS,
            available=available,
            group_fields=("track_id", "event"),
            require_track_id=True,
        ),
        "global_60s": _botified_output_limit_summary(
            timed_events,
            threshold=BOTIFIED_OUTPUT_LIMIT_GLOBAL_60S_THRESHOLD,
            window_ms=BOTIFIED_OUTPUT_LIMIT_60S_WINDOW_MS,
            available=available,
        ),
        "burst_1s": _botified_output_limit_summary(
            timed_events,
            threshold=BOTIFIED_OUTPUT_LIMIT_BURST_1S_THRESHOLD,
            window_ms=BOTIFIED_OUTPUT_LIMIT_1S_WINDOW_MS,
            available=available,
        ),
    }


def _botified_timed_events(
    event_sequence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    timed_events: list[dict[str, Any]] = []
    for item in event_sequence:
        received_monotonic_ms = item.get("received_monotonic_ms")
        if isinstance(received_monotonic_ms, bool) or not isinstance(
            received_monotonic_ms,
            int,
        ):
            continue
        line = item.get("line")
        if isinstance(line, bool) or not isinstance(line, int):
            continue
        event = item.get("event")
        if not isinstance(event, str):
            continue
        timed_event: dict[str, Any] = {
            "received_monotonic_ms": received_monotonic_ms,
            "line": line,
            "event": event,
        }
        event_id = item.get("event_id")
        if isinstance(event_id, str):
            timed_event["event_id"] = event_id
        track_id = item.get("track_id")
        if isinstance(track_id, int) and not isinstance(track_id, bool):
            timed_event["track_id"] = track_id
        timed_events.append(timed_event)
    return timed_events


def _botified_output_limit_summary(
    timed_events: list[dict[str, Any]],
    *,
    threshold: int,
    window_ms: int,
    available: bool,
    group_fields: tuple[str, ...] = (),
    require_track_id: bool = False,
) -> dict[str, Any]:
    if not available:
        return {
            "available": False,
            "passed": None,
            "max_count": 0,
            "threshold": threshold,
            "window_ms": window_ms,
            "violations": [],
        }

    grouped_events: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for event in timed_events:
        if require_track_id and "track_id" not in event:
            continue
        group_key = tuple(event.get(field) for field in group_fields)
        grouped_events.setdefault(group_key, []).append(event)
    if not group_fields:
        grouped_events = {(): list(timed_events)}

    max_count = 0
    violations: list[dict[str, Any]] = []
    for group_key in sorted(
        grouped_events,
        key=lambda key: tuple(str(part) for part in key),
    ):
        group_events = grouped_events[group_key]
        window = _botified_max_window(group_events, window_ms=window_ms)
        max_count = max(max_count, len(window))
        if len(window) <= threshold:
            continue
        violation: dict[str, Any] = {
            "count": len(window),
            "threshold": threshold,
            "window_ms": window_ms,
            "event_ids": [
                event["event_id"]
                for event in window
                if isinstance(event.get("event_id"), str)
            ],
            "lines": [event["line"] for event in window],
        }
        for field, value in zip(group_fields, group_key, strict=True):
            violation[field] = value
        violations.append(violation)

    return {
        "available": True,
        "passed": not violations,
        "max_count": max_count,
        "threshold": threshold,
        "window_ms": window_ms,
        "violations": violations,
    }


def _botified_max_window(
    timed_events: list[dict[str, Any]],
    *,
    window_ms: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        timed_events,
        key=lambda event: (
            int(event["received_monotonic_ms"]),
            int(event["line"]),
        ),
    )
    best: list[dict[str, Any]] = []
    start = 0
    for end, event in enumerate(ordered):
        end_ms = int(event["received_monotonic_ms"])
        while start <= end:
            start_ms = int(ordered[start]["received_monotonic_ms"])
            if end_ms - start_ms < window_ms:
                break
            start += 1
        candidate = ordered[start : end + 1]
        if len(candidate) > len(best):
            best = candidate
    return best


def _is_wrapped_botified_frame(line: str) -> bool:
    return (
        line.startswith(BOTIFIED_OPEN)
        and line.endswith(BOTIFIED_CLOSE)
        and line.count(BOTIFIED_OPEN) == 1
        and line.count(BOTIFIED_CLOSE) == 1
    )


def _parse_botified_payload(line: str) -> dict[str, Any] | None:
    inner = line[len(BOTIFIED_OPEN) : -len(BOTIFIED_CLOSE)]
    try:
        payload = json.loads(inner)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _botified_contract_event(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    frame_id = payload.get("id")
    if not isinstance(frame_id, str) or not frame_id.startswith("visual:"):
        return None, "invalid Botified id"
    if payload.get("urgency") != "normal":
        return None, "invalid Botified urgency"
    timeout_secs = payload.get("timeout_secs")
    if not isinstance(timeout_secs, int) or isinstance(timeout_secs, bool):
        return None, "invalid Botified timeout_secs"
    if payload.get("expect") != "ack":
        return None, "invalid Botified expect"
    request = payload.get("request")
    if not isinstance(request, str):
        return None, "invalid Botified request"
    match = BOTIFIED_EVENT_RE.search(request)
    if match is None:
        return None, "missing Botified request event"
    event = match.group(1)
    if event == "attention_target_changed":
        return event, None
    if event not in BOTIFIED_ALLOWED_EVENTS:
        return None, f"unsupported Botified event {event}"
    return event, None


def _botified_event_id(payload: dict[str, Any]) -> str | None:
    frame_id = payload.get("id")
    if not isinstance(frame_id, str) or not frame_id.startswith("visual:"):
        return None
    event_id = frame_id[len("visual:") :]
    return event_id or None


def _botified_track_id(payload: dict[str, Any]) -> int | None:
    request = payload.get("request")
    if not isinstance(request, str):
        return None
    match = BOTIFIED_TRACK_ID_RE.search(request)
    if match is None:
        return None
    return int(match.group(1))


def _botified_stdout_failure_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if summary["collection_truncated"]:
        reasons.append("botified_stdout_collection_truncated")
    if summary["collection_incomplete"]:
        reasons.append("botified_stdout_collection_incomplete")
    if summary["pollution_count"] > 0:
        reasons.append("botified_stdout_pollution")
    if summary["parse_error_count"] > 0:
        reasons.append("botified_stdout_parse_errors")
    if summary["forbidden_event_count"] > 0:
        reasons.append("botified_stdout_forbidden_event")
    output_limits = summary.get("output_limits")
    if isinstance(output_limits, dict):
        if _botified_output_limit_failed(output_limits, "per_track_event_60s"):
            reasons.append("botified_stdout_output_limit_per_track_event_60s")
        if _botified_output_limit_failed(output_limits, "global_60s"):
            reasons.append("botified_stdout_output_limit_global_60s")
        if _botified_output_limit_failed(output_limits, "burst_1s"):
            reasons.append("botified_stdout_output_limit_burst_1s")
    return reasons


def _botified_output_limit_failed(
    output_limits: dict[str, Any],
    key: str,
) -> bool:
    summary = output_limits.get(key)
    return isinstance(summary, dict) and summary.get("passed") is False


def _slice_pass(
    *,
    failure_reasons: list[str],
    health_result: HealthCheckResult | None,
    segment_results: list[SegmentSmokeResult],
    expected_segment_count: int,
    expected_gaze_count: int,
) -> bool:
    return (
        not failure_reasons
        and health_result is not None
        and health_result.passed
        and len(segment_results) == expected_segment_count
        and all(
            result.head_result is not None
            and result.head_result.returncode == 0
            and result.head_stdout_parse_error is None
            and result.image_result is not None
            and result.image_result.returncode == 0
            and result.gaze_result is not None
            and result.gaze_result.returncode == 0
            and result.gaze_summary["accepted_count"] >= expected_gaze_count
            for result in segment_results
        )
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


def _all_scene_names_from_effective_manifest(
    *,
    data_dir: Path,
    effective_manifest: Any,
) -> list[str]:
    if not isinstance(effective_manifest, dict):
        raise PreflightError("effective manifest is not an object")
    scenes = effective_manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise PreflightError("effective manifest has no scenes")

    names: list[str] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise PreflightError(f"effective manifest scene {index} is invalid")
        name = scene.get("scene_name")
        if not isinstance(name, str):
            name = scene.get("name")
        if not isinstance(name, str) or name == "":
            raise PreflightError(f"effective manifest scene {index} has no scene name")
        scene_path = Path(name)
        if scene_path.is_absolute() or len(scene_path.parts) != 1:
            raise PreflightError(f"effective manifest scene name is invalid: {name}")
        if name in names:
            raise PreflightError(f"effective manifest duplicate scene: {name}")
        full_scene_path = data_dir / name
        if not full_scene_path.exists():
            raise PreflightError(f"selected scene directory not found: {full_scene_path}")
        if not full_scene_path.is_dir():
            raise PreflightError(f"scene is not a directory: {full_scene_path}")
        if not cli_local_e2e_manifest.jpeg_files(full_scene_path):
            raise PreflightError(f"scene has no JPEG frames: {name}")
        names.append(name)
    return names


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


def _preflight_runtime_provenance(
    *,
    server_bin: Path,
    cli_bin: Path,
    server_config: Path | None,
) -> dict[str, Any]:
    try:
        report = _collect_runtime_provenance(
            server_bin=server_bin,
            cli_bin=cli_bin,
            server_config=server_config,
        )
    except runtime_provenance.RuntimeProvenanceError as exc:
        raise PreflightError(str(exc)) from exc
    if report["failure_reasons"]:
        reasons = ",".join(str(reason) for reason in report["failure_reasons"])
        raise PreflightError(f"runtime_provenance_failed:{reasons}")
    return report


def _runtime_provenance_report_for_failure(args: argparse.Namespace) -> dict[str, Any]:
    return runtime_provenance.runtime_provenance_report_for_failure(
        repo_root=REPO_ROOT,
        server_bin=args.server_bin,
        cli_bin=args.cli_bin,
        server_config=args.server_config,
    )


def _collect_runtime_provenance(
    *,
    server_bin: Path,
    cli_bin: Path,
    server_config: Path | None,
) -> dict[str, Any]:
    return runtime_provenance.collect_runtime_provenance(
        repo_root=REPO_ROOT,
        server_bin=server_bin,
        cli_bin=cli_bin,
        server_config=server_config,
    )


def _runtime_provenance_report_with_exit_codes(
    runtime_report: dict[str, Any],
    *,
    server_exit_code: int | None,
    cli_exit_code: int | None,
) -> dict[str, Any]:
    return runtime_provenance.runtime_provenance_report_with_exit_codes(
        runtime_report,
        server_exit_code=server_exit_code,
        cli_exit_code=cli_exit_code,
    )


def _runtime_provenance_flat_aliases(
    runtime_report: dict[str, Any],
    *,
    server_exit_code: int | None,
    cli_exit_code: int | None,
) -> dict[str, Any]:
    return runtime_provenance.runtime_provenance_flat_aliases(
        runtime_report,
        server_exit_code=server_exit_code,
        cli_exit_code=cli_exit_code,
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


def _process_exit_code(process: ProcessLike | None) -> int | None:
    return None if process is None else process.poll()


def _runtime_exit_code_failure_reasons(
    *,
    server_exit_code: int | None,
    cli_exit_code: int | None,
) -> list[str]:
    reasons: list[str] = []
    if (
        server_exit_code is not None
        and server_exit_code not in EXPECTED_RUNTIME_EXIT_CODES
    ):
        reasons.append("server_exit_code_unexpected")
    if cli_exit_code is not None and cli_exit_code not in EXPECTED_RUNTIME_EXIT_CODES:
        reasons.append("cli_exit_code_unexpected")
    return reasons


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


def _collected_stream_lines(process: ProcessLike | None, stream_name: str) -> list[str]:
    if process is None:
        return []
    lines = getattr(process, f"{stream_name}_lines", None)
    if lines is None:
        return []
    return [str(line) for line in lines]


def _collected_stream_line_count(
    process: ProcessLike | None,
    stream_name: str,
    *,
    fallback: int,
) -> int:
    if process is None:
        return 0
    line_count = getattr(process, f"{stream_name}_line_count", None)
    if isinstance(line_count, int) and not isinstance(line_count, bool):
        return line_count
    return fallback


def _collected_stdout_line_timestamps_ms(
    process: ProcessLike | None,
) -> list[int] | None:
    if process is None:
        return None
    timestamps = getattr(process, "stdout_line_timestamps_ms", None)
    if timestamps is None:
        return None
    result: list[int] = []
    for timestamp in timestamps:
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            return None
        result.append(timestamp)
    return result


def _collection_truncated(
    process: ProcessLike | None,
    stream_name: str,
    *,
    collected_count: int,
) -> bool:
    if process is None:
        return False
    explicit = bool(getattr(process, f"{stream_name}_truncated", False))
    observed_count = _collected_stream_line_count(
        process,
        stream_name,
        fallback=collected_count,
    )
    return explicit or observed_count > collected_count


def _capture_finished_process(process: ManagedProcess) -> None:
    process.join_readers(PROCESS_READER_JOIN_TIMEOUT_S)


def _tail(value: str, *, limit: int = 4096) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _lines_to_text(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


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
