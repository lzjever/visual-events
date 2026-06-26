import json
from pathlib import Path

import pytest

import tools.run_runtime_smoke as runtime_smoke
from tools.run_runtime_smoke import (
    CommandResult,
    HealthCheckResult,
    HealthzResponse,
    RuntimeSmokeRunner,
    main,
)


class FakeProcess:
    def __init__(
        self,
        *,
        pid: int = 4242,
        returncode: int | None = None,
        returncodes: list[int | None] | None = None,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self.returncodes = list(returncodes or [])

    def poll(self) -> int | None:
        if self.returncodes:
            return self.returncodes.pop(0)
        return self.returncode


class FakeRunner:
    def __init__(
        self,
        *,
        sync_result: CommandResult | None = None,
        health_result: HealthCheckResult | None = None,
        process: FakeProcess | None = None,
    ) -> None:
        self.sync_result = sync_result or CommandResult(returncode=0)
        self.process = process or FakeProcess()
        self.health_result = health_result or HealthCheckResult(
            passed=True,
            healthz_pid=self.process.pid,
            healthz_identity_verified=True,
        )
        self.sync_calls: list[dict[str, object]] = []
        self.start_calls: list[dict[str, object]] = []
        self.health_calls: list[dict[str, object]] = []
        self.stopped_processes: list[FakeProcess] = []

    def run_sync(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> CommandResult:
        self.sync_calls.append({"command": command, "cwd": cwd, "env": env})
        return self.sync_result

    def start_server(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> FakeProcess:
        self.start_calls.append({"command": command, "cwd": cwd, "env": env})
        return self.process

    def wait_healthz(
        self,
        url: str,
        process: FakeProcess,
        *,
        timeout_s: float,
        interval_s: float,
    ) -> HealthCheckResult:
        self.health_calls.append(
            {
                "url": url,
                "process": process,
                "timeout_s": timeout_s,
                "interval_s": interval_s,
            }
        )
        return self.health_result

    def stop_server(self, process: FakeProcess) -> None:
        self.stopped_processes.append(process)


def read_report(out_dir: Path) -> dict:
    return json.loads((out_dir / "report.json").read_text(encoding="utf-8"))


def test_sync_command_and_env_use_repo_local_runtime_without_home_override(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", "/sentinel-home")
    runner = FakeRunner()
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 0
    assert len(runner.sync_calls) == 1
    sync_call = runner.sync_calls[0]
    assert sync_call["command"] == [
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
    assert sync_call["cwd"] == tmp_path
    sync_env = sync_call["env"]
    assert sync_env["UV_CACHE_DIR"] == str(tmp_path / "runtime" / "cache" / "uv")
    assert sync_env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "runtime" / "venv")
    assert sync_env["HOME"] == "/sentinel-home"


def test_success_writes_pass_report_and_stops_server(tmp_path):
    runner = FakeRunner()
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--config",
            str(tmp_path / "runtime" / "config.toml"),
            "--host",
            "127.0.0.1",
            "--port",
            "8766",
        ],
        runner=runner,
    )

    assert exit_code == 0
    assert runner.stopped_processes == [runner.process]
    assert runner.start_calls[0]["command"] == [
        str(tmp_path / "runtime" / "venv" / "bin" / "visual-events-server"),
        "--config",
        str(tmp_path / "runtime" / "config.toml"),
        "--host",
        "127.0.0.1",
        "--port",
        "8766",
    ]
    assert runner.health_calls[0]["url"] == "http://127.0.0.1:8766/healthz"

    report = read_report(out_dir)
    assert report["passed"] is True
    assert report["failure_reasons"] == []
    assert report["sync_command"] == runner.sync_calls[0]["command"]
    assert report["server_command"] == runner.start_calls[0]["command"]
    assert report["config_path"] == str(tmp_path / "runtime" / "config.toml")
    assert report["healthz_url"] == "http://127.0.0.1:8766/healthz"
    assert report["server_pid"] == 4242
    assert report["healthz_pid"] == 4242
    assert report["healthz_identity_verified"] is True
    assert isinstance(report["started_at"], str)
    assert isinstance(report["finished_at"], str)
    assert isinstance(report["elapsed_s"], float)


def test_wait_healthz_accepts_matching_process_pid(monkeypatch):
    monkeypatch.setattr(
        runtime_smoke,
        "_request_healthz",
        lambda url, *, timeout_s: HealthzResponse(200, {"ok": True, "pid": 4242}),
    )

    result = RuntimeSmokeRunner().wait_healthz(
        "http://127.0.0.1:8765/healthz",
        FakeProcess(pid=4242),
        timeout_s=0.01,
        interval_s=0.001,
    )

    assert result == HealthCheckResult(
        passed=True,
        healthz_pid=4242,
        healthz_identity_verified=True,
    )


def test_wait_healthz_rejects_healthz_from_existing_server(monkeypatch):
    monkeypatch.setattr(
        runtime_smoke,
        "_request_healthz",
        lambda url, *, timeout_s: HealthzResponse(200, {"ok": True, "pid": 9999}),
    )

    result = RuntimeSmokeRunner().wait_healthz(
        "http://127.0.0.1:8765/healthz",
        FakeProcess(pid=4242),
        timeout_s=0.01,
        interval_s=0.001,
    )

    assert result == HealthCheckResult(
        passed=False,
        failure_reason="healthz_identity_mismatch",
        healthz_pid=9999,
    )


def test_wait_healthz_rejects_legacy_healthz_without_pid(monkeypatch):
    monkeypatch.setattr(
        runtime_smoke,
        "_request_healthz",
        lambda url, *, timeout_s: HealthzResponse(200, {"ok": True}),
    )

    result = RuntimeSmokeRunner().wait_healthz(
        "http://127.0.0.1:8765/healthz",
        FakeProcess(pid=4242),
        timeout_s=0.01,
        interval_s=0.001,
    )

    assert result == HealthCheckResult(
        passed=False,
        failure_reason="healthz_identity_mismatch",
    )


def test_wait_healthz_rejects_unhealthy_healthz(monkeypatch):
    monkeypatch.setattr(
        runtime_smoke,
        "_request_healthz",
        lambda url, *, timeout_s: HealthzResponse(200, {"ok": False, "pid": 4242}),
    )

    result = RuntimeSmokeRunner().wait_healthz(
        "http://127.0.0.1:8765/healthz",
        FakeProcess(pid=4242),
        timeout_s=0.01,
        interval_s=0.001,
    )

    assert result == HealthCheckResult(
        passed=False,
        failure_reason="healthz_unhealthy",
        healthz_pid=4242,
    )


def test_wait_healthz_rejects_invalid_json_healthz(monkeypatch):
    monkeypatch.setattr(
        runtime_smoke,
        "_request_healthz",
        lambda url, *, timeout_s: HealthzResponse(200, None, "healthz_invalid_json"),
    )

    result = RuntimeSmokeRunner().wait_healthz(
        "http://127.0.0.1:8765/healthz",
        FakeProcess(pid=4242),
        timeout_s=0.01,
        interval_s=0.001,
    )

    assert result == HealthCheckResult(
        passed=False,
        failure_reason="healthz_unhealthy",
    )


def test_wait_healthz_rejects_process_exit_after_identity_response(monkeypatch):
    monkeypatch.setattr(
        runtime_smoke,
        "_request_healthz",
        lambda url, *, timeout_s: HealthzResponse(200, {"ok": True, "pid": 4242}),
    )

    result = RuntimeSmokeRunner().wait_healthz(
        "http://127.0.0.1:8765/healthz",
        FakeProcess(pid=4242, returncodes=[None, 1]),
        timeout_s=0.01,
        interval_s=0.001,
    )

    assert result == HealthCheckResult(
        passed=False,
        failure_reason="server_exited_early",
        healthz_pid=4242,
        healthz_identity_verified=True,
    )


def test_default_config_path_uses_runtime_config_s2_in_command_and_report(tmp_path):
    runner = FakeRunner()
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 0
    default_config = tmp_path / "runtime" / "config" / "s2.toml"
    assert runner.start_calls[0]["command"] == [
        str(tmp_path / "runtime" / "venv" / "bin" / "visual-events-server"),
        "--config",
        str(default_config),
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    ]

    report = read_report(out_dir)
    assert report["server_command"] == runner.start_calls[0]["command"]
    assert report["config_path"] == str(default_config)


@pytest.mark.parametrize(
    ("runner", "failure_reason", "expect_started"),
    [
        (
            FakeRunner(sync_result=CommandResult(returncode=9, stderr="sync failed")),
            "sync_failed",
            False,
        ),
        (
            FakeRunner(health_result=HealthCheckResult(False, "server_exited_early")),
            "server_exited_early",
            True,
        ),
        (
            FakeRunner(health_result=HealthCheckResult(False, "healthz_timeout")),
            "healthz_timeout",
            True,
        ),
    ],
)
def test_failures_return_nonzero_and_write_failure_report(
    tmp_path,
    runner,
    failure_reason,
    expect_started,
):
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 1
    assert bool(runner.start_calls) is expect_started
    if expect_started:
        assert runner.stopped_processes == [runner.process]

    report = read_report(out_dir)
    assert report["passed"] is False
    assert failure_reason in report["failure_reasons"]


def test_rejects_out_dir_under_val_data_without_running_or_writing_report(tmp_path):
    runner = FakeRunner()
    out_dir = tmp_path / "val-data" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 1
    assert runner.sync_calls == []
    assert runner.start_calls == []
    assert not (out_dir / "report.json").exists()
