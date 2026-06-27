from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    from tools import runtime_provenance
except ModuleNotFoundError:
    import runtime_provenance  # type: ignore[no-redef]


DEFAULT_OUT_DIR = Path("artifacts/runtime-smoke")
DEFAULT_CONFIG = Path("runtime/config/s2.toml")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_HEALTH_TIMEOUT_S = 30.0
DEFAULT_HEALTH_INTERVAL_S = 0.2
SERVER_STOP_TIMEOUT_S = 5.0


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


@dataclass(frozen=True)
class SmokeConfig:
    repo_root: Path
    out_dir: Path
    config_path: Path
    host: str
    port: int
    health_timeout_s: float
    health_interval_s: float

    @property
    def healthz_url(self) -> str:
        return f"http://{self.host}:{self.port}/healthz"


class PreflightError(Exception):
    pass


class RuntimeSmokeRunner:
    def run_sync(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> CommandResult:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
        description="Smoke test release/runtime visual-events-server startup."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--health-timeout-s",
        type=float,
        default=DEFAULT_HEALTH_TIMEOUT_S,
    )
    parser.add_argument(
        "--health-interval-s",
        type=float,
        default=DEFAULT_HEALTH_INTERVAL_S,
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    runner: RuntimeSmokeRunner | None = None,
) -> int:
    args = parse_args(argv)
    try:
        config = _build_config(args)
        _preflight(config)
    except PreflightError as exc:
        print(f"runtime smoke preflight failed: {exc}", file=sys.stderr)
        return 1

    active_runner = runner or RuntimeSmokeRunner()
    return _run_smoke(config, active_runner)


def _run_smoke(config: SmokeConfig, runner: RuntimeSmokeRunner) -> int:
    started_at = _utc_now()
    started_s = time.perf_counter()
    sync_command = _sync_command()
    server_command = _server_command(config)
    cli_check_command = _cli_import_check_command(config)
    sync_env = _runtime_env(config.repo_root)
    runtime_env = runtime_provenance.runtime_execution_env()
    failure_reasons: list[str] = []
    server_process: ProcessLike | None = None
    sync_result: CommandResult | None = None
    cli_check_result: CommandResult | None = None
    health_result: HealthCheckResult | None = None
    runtime_report: dict[str, Any] | None = None

    try:
        sync_result = runner.run_sync(sync_command, cwd=config.repo_root, env=sync_env)
        if sync_result.returncode != 0:
            failure_reasons.append("sync_failed")
            runtime_report = _runtime_provenance_not_run(config, reason="sync_failed")
            return _finish(
                config,
                passed=False,
                started_at=started_at,
                started_s=started_s,
                sync_command=sync_command,
                server_command=server_command,
                cli_check_command=cli_check_command,
                sync_result=sync_result,
                cli_check_result=cli_check_result,
                runtime_report=runtime_report,
                server_process=server_process,
                health_result=health_result,
                failure_reasons=failure_reasons,
            )

        runtime_report = _collect_runtime_provenance(config)
        if runtime_report["failure_reasons"]:
            reasons = ",".join(str(reason) for reason in runtime_report["failure_reasons"])
            failure_reasons.append(f"runtime_provenance_failed:{reasons}")
            return _finish(
                config,
                passed=False,
                started_at=started_at,
                started_s=started_s,
                sync_command=sync_command,
                server_command=server_command,
                cli_check_command=cli_check_command,
                sync_result=sync_result,
                cli_check_result=cli_check_result,
                runtime_report=runtime_report,
                server_process=server_process,
                health_result=health_result,
                failure_reasons=failure_reasons,
            )

        cli_check_result = runner.run_sync(
            cli_check_command,
            cwd=config.repo_root,
            env=runtime_env,
        )
        if cli_check_result.returncode != 0:
            failure_reasons.append("cli_import_check_failed")
            return _finish(
                config,
                passed=False,
                started_at=started_at,
                started_s=started_s,
                sync_command=sync_command,
                server_command=server_command,
                cli_check_command=cli_check_command,
                sync_result=sync_result,
                cli_check_result=cli_check_result,
                runtime_report=runtime_report,
                server_process=server_process,
                health_result=health_result,
                failure_reasons=failure_reasons,
            )

        server_process = runner.start_server(
            server_command,
            cwd=config.repo_root,
            env=runtime_env,
        )
        health_result = runner.wait_healthz(
            config.healthz_url,
            server_process,
            timeout_s=config.health_timeout_s,
            interval_s=config.health_interval_s,
        )
        if not health_result.passed:
            failure_reasons.append(health_result.failure_reason or "healthz_failed")
    except Exception as exc:
        failure_reasons.append(f"runtime_smoke_exception:{type(exc).__name__}")
    finally:
        if server_process is not None:
            runner.stop_server(server_process)

    return _finish(
        config,
        passed=not failure_reasons,
        started_at=started_at,
        started_s=started_s,
        sync_command=sync_command,
        server_command=server_command,
        cli_check_command=cli_check_command,
        sync_result=sync_result,
        cli_check_result=cli_check_result,
        runtime_report=runtime_report,
        server_process=server_process,
        health_result=health_result,
        failure_reasons=failure_reasons,
    )


def _build_config(args: argparse.Namespace) -> SmokeConfig:
    repo_root = args.repo_root.resolve()
    out_dir = _resolve_against(repo_root, args.out_dir)
    config_path = _resolve_against(repo_root, args.config)
    return SmokeConfig(
        repo_root=repo_root,
        out_dir=out_dir,
        config_path=config_path,
        host=str(args.host),
        port=int(args.port),
        health_timeout_s=float(args.health_timeout_s),
        health_interval_s=float(args.health_interval_s),
    )


def _preflight(config: SmokeConfig) -> None:
    if config.port <= 0:
        raise PreflightError("port must be positive")
    if config.health_timeout_s <= 0.0:
        raise PreflightError("health-timeout-s must be positive")
    if config.health_interval_s <= 0.0:
        raise PreflightError("health-interval-s must be positive")

    val_data = (config.repo_root / "val-data").resolve()
    out_dir = config.out_dir.resolve()
    if out_dir == val_data or out_dir.is_relative_to(val_data):
        raise PreflightError("out-dir must not be inside val-data")


def _sync_command() -> list[str]:
    return [
        "uv",
        "sync",
        "--frozen",
        "--no-dev",
        "--no-editable",
        "--extra",
        "inference",
        "--reinstall-package",
        "visual-events-server",
    ]


def _server_command(config: SmokeConfig) -> list[str]:
    return [
        str(_runtime_server_bin(config)),
        "--config",
        str(config.config_path),
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]


def _cli_import_check_command(config: SmokeConfig) -> list[str]:
    return [
        str(config.repo_root / "runtime" / "venv" / "bin" / "python"),
        "-c",
        "import visual_events_server.app; import visual_events_cli.main",
    ]


def _runtime_server_bin(config: SmokeConfig) -> Path:
    return config.repo_root / "runtime" / "venv" / "bin" / "visual-events-server"


def _runtime_cli_bin(config: SmokeConfig) -> Path:
    return config.repo_root / "runtime" / "venv" / "bin" / "visual-events-cli"


def _collect_runtime_provenance(config: SmokeConfig) -> dict[str, Any]:
    try:
        return runtime_provenance.collect_runtime_provenance(
            repo_root=config.repo_root,
            server_bin=_runtime_server_bin(config),
            cli_bin=_runtime_cli_bin(config),
            server_config=config.config_path,
        )
    except Exception:
        return runtime_provenance.runtime_provenance_report_for_failure(
            repo_root=config.repo_root,
            server_bin=_runtime_server_bin(config),
            cli_bin=_runtime_cli_bin(config),
            server_config=config.config_path,
        )


def _runtime_provenance_not_run(config: SmokeConfig, *, reason: str) -> dict[str, Any]:
    return runtime_provenance.runtime_provenance_not_run_report(
        repo_root=config.repo_root,
        server_bin=_runtime_server_bin(config),
        cli_bin=_runtime_cli_bin(config),
        server_config=config.config_path,
        reason=reason,
    )


def _runtime_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(repo_root / "runtime" / "cache" / "uv")
    env["UV_PROJECT_ENVIRONMENT"] = str(repo_root / "runtime" / "venv")
    return env


def _finish(
    config: SmokeConfig,
    *,
    passed: bool,
    started_at: str,
    started_s: float,
    sync_command: list[str],
    server_command: list[str],
    cli_check_command: list[str],
    sync_result: CommandResult | None,
    cli_check_result: CommandResult | None,
    runtime_report: dict[str, Any] | None,
    server_process: ProcessLike | None,
    health_result: HealthCheckResult | None,
    failure_reasons: list[str],
) -> int:
    report = _build_report(
        config,
        passed=passed,
        started_at=started_at,
        elapsed_s=max(0.0, time.perf_counter() - started_s),
        sync_command=sync_command,
        server_command=server_command,
        cli_check_command=cli_check_command,
        sync_result=sync_result,
        cli_check_result=cli_check_result,
        runtime_report=runtime_report,
        server_process=server_process,
        health_result=health_result,
        failure_reasons=failure_reasons,
    )
    _write_json(config.out_dir / "report.json", report)
    return 0 if passed else 1


def _build_report(
    config: SmokeConfig,
    *,
    passed: bool,
    started_at: str,
    elapsed_s: float,
    sync_command: list[str],
    server_command: list[str],
    cli_check_command: list[str],
    sync_result: CommandResult | None,
    cli_check_result: CommandResult | None,
    runtime_report: dict[str, Any] | None,
    server_process: ProcessLike | None,
    health_result: HealthCheckResult | None,
    failure_reasons: list[str],
) -> dict[str, object]:
    if runtime_report is None:
        runtime_report = _runtime_provenance_not_run(
            config,
            reason="runtime_report_missing",
        )
    server_returncode = None if server_process is None else server_process.poll()
    report = {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "repo_root": str(config.repo_root),
        "out_dir": str(config.out_dir),
        "config_path": str(config.config_path),
        "healthz_url": config.healthz_url,
        "sync_command": sync_command,
        "sync_env": {
            "UV_CACHE_DIR": str(config.repo_root / "runtime" / "cache" / "uv"),
            "UV_PROJECT_ENVIRONMENT": str(config.repo_root / "runtime" / "venv"),
        },
        "sync_returncode": None if sync_result is None else sync_result.returncode,
        "cli_check_command": cli_check_command,
        "cli_check_returncode": None
        if cli_check_result is None
        else cli_check_result.returncode,
        "server_command": server_command,
        "server_pid": None if server_process is None else server_process.pid,
        "server_returncode": server_returncode,
        "healthz_pid": None if health_result is None else health_result.healthz_pid,
        "healthz_identity_verified": False
        if health_result is None
        else health_result.healthz_identity_verified,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_s": float(elapsed_s),
    }
    report.update(
        runtime_provenance.runtime_provenance_flat_aliases(
            runtime_report,
            server_exit_code=server_returncode,
            cli_exit_code=None,
        )
    )
    return report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _request_healthz(url: str, *, timeout_s: float) -> HealthzResponse | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read()
    except (OSError, TimeoutError, urllib.error.URLError):
        return None

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return HealthzResponse(response.status, None, "healthz_invalid_json")
    if not isinstance(payload, dict):
        return HealthzResponse(response.status, None, "healthz_unhealthy")
    return HealthzResponse(response.status, payload)


def _healthz_pid(payload: dict[str, object] | None) -> int | None:
    if payload is None:
        return None
    pid = payload.get("pid")
    if isinstance(pid, bool):
        return None
    if isinstance(pid, int):
        return pid
    return None


def _resolve_against(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
