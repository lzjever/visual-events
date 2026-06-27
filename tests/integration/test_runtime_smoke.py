import json
import base64
import hashlib
import os
import stat
import sys
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
        cli_check_result: CommandResult | None = None,
        health_result: HealthCheckResult | None = None,
        process: FakeProcess | None = None,
    ) -> None:
        self.sync_result = sync_result or CommandResult(returncode=0)
        self.cli_check_result = cli_check_result or CommandResult(returncode=0)
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
        self.events: list[str] = []

    def run_sync(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> CommandResult:
        self.sync_calls.append({"command": command, "cwd": cwd, "env": env})
        if command[:1] == ["uv"]:
            self.events.append("sync")
            return self.sync_result
        self.events.append("cli_import_check")
        return self.cli_check_result

    def start_server(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> FakeProcess:
        self.start_calls.append({"command": command, "cwd": cwd, "env": env})
        self.events.append("start_server")
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
        self.events.append("healthz")
        return self.health_result

    def stop_server(self, process: FakeProcess) -> None:
        self.stopped_processes.append(process)


def read_report(out_dir: Path) -> dict:
    return json.loads((out_dir / "report.json").read_text(encoding="utf-8"))


def write_project_contract(repo_root: Path) -> None:
    (repo_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "visual-events-server"',
                'version = "0.1.0"',
                "",
                "[project.scripts]",
                'visual-events-server = "visual_events_server.app:main"',
                'visual-events-cli = "visual_events_cli.main:main"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def record_digest(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"sha256={encoded}"


def record_line(site_packages: Path, path: Path) -> str:
    rel_path = os.path.relpath(path, site_packages).replace(os.sep, "/")
    return f"{rel_path},{record_digest(path)},{path.stat().st_size}\n"


def make_runtime_distribution(repo_root: Path) -> dict[str, Path]:
    write_project_contract(repo_root)
    config_path = repo_root / "runtime" / "config" / "s2.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("[server]\n", encoding="utf-8")
    venv = repo_root / "runtime" / "venv"
    bin_dir = venv / "bin"
    server_bin = make_executable(bin_dir / "visual-events-server")
    cli_bin = make_executable(bin_dir / "visual-events-cli")
    make_executable(bin_dir / "python")
    site_packages = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    dist_info = site_packages / "visual_events_server-0.1.0.dist-info"
    metadata = dist_info / "METADATA"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("Name: visual-events-server\nVersion: 0.1.0\n", encoding="utf-8")
    entry_points = dist_info / "entry_points.txt"
    entry_points.write_text(
        "[console_scripts]\n"
        "visual-events-server = visual_events_server.app:main\n"
        "visual-events-cli = visual_events_cli.main:main\n",
        encoding="utf-8",
    )
    record = dist_info / "RECORD"
    record.write_text(
        "".join(
            [
                record_line(site_packages, metadata),
                record_line(site_packages, entry_points),
                record_line(site_packages, server_bin),
                record_line(site_packages, cli_bin),
                f"{os.path.relpath(record, site_packages).replace(os.sep, '/')},,\n",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "config_path": config_path,
        "server_bin": server_bin,
        "cli_bin": cli_bin,
        "dist_info": dist_info,
    }


def test_sync_command_and_env_use_repo_local_runtime_without_home_override(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", "/sentinel-home")
    make_runtime_distribution(tmp_path)
    runner = FakeRunner()
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 0
    assert len(runner.sync_calls) == 2
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


def test_runtime_server_and_import_check_do_not_inherit_ambient_python_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/sentinel-home")
    monkeypatch.setenv("PYTHONPATH", "/ambient/source")
    monkeypatch.setenv("PYTHONHOME", "/ambient/python")
    monkeypatch.setenv("PYTHONUSERBASE", "/ambient/userbase")
    monkeypatch.setenv("VIRTUAL_ENV", "/ambient/venv")
    monkeypatch.setenv("VIRTUAL_ENV_PROMPT", "(ambient)")
    monkeypatch.setenv("__PYVENV_LAUNCHER__", "/ambient/python")
    make_runtime_distribution(tmp_path)
    runner = FakeRunner()
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 0
    assert len(runner.sync_calls) == 2
    runtime_envs = [runner.sync_calls[1]["env"], runner.start_calls[0]["env"]]
    for env in runtime_envs:
        assert env["HOME"] == "/sentinel-home"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["PYTHONSAFEPATH"] == "1"
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
            "VIRTUAL_ENV_PROMPT",
            "__PYVENV_LAUNCHER__",
        ):
            assert key not in env
    sync_env = runner.sync_calls[0]["env"]
    assert sync_env["UV_CACHE_DIR"] == str(tmp_path / "runtime" / "cache" / "uv")
    assert sync_env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "runtime" / "venv")


def test_success_writes_pass_report_and_stops_server(tmp_path):
    make_runtime_distribution(tmp_path)
    custom_config = tmp_path / "runtime" / "config.toml"
    custom_config.write_text("[server]\n", encoding="utf-8")
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
    assert runner.events == ["sync", "cli_import_check", "start_server", "healthz"]
    assert runner.stopped_processes == [runner.process]
    assert runner.sync_calls[1]["command"] == [
        str(tmp_path / "runtime" / "venv" / "bin" / "python"),
        "-c",
        "import visual_events_server.app; import visual_events_cli.main",
    ]
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
    assert report["cli_check_command"] == runner.sync_calls[1]["command"]
    assert report["cli_check_returncode"] == 0
    assert report["server_command"] == runner.start_calls[0]["command"]
    assert report["config_path"] == str(tmp_path / "runtime" / "config.toml")
    assert report["healthz_url"] == "http://127.0.0.1:8766/healthz"
    assert report["server_pid"] == 4242
    assert report["healthz_pid"] == 4242
    assert report["healthz_identity_verified"] is True
    assert report["runtime_provenance"]["failure_reasons"] == []
    assert report["server_bin_is_runtime_venv"] is True
    assert report["cli_bin_is_runtime_venv"] is True
    assert report["wheel_name"] == "visual-events-server"
    assert report["wheel_version"] == "0.1.0"
    assert isinstance(report["runtime_hash"], str)
    assert isinstance(report["config_hash"], str)
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
    make_runtime_distribution(tmp_path)
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
    if failure_reason != "sync_failed":
        make_runtime_distribution(tmp_path)
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 1
    assert bool(runner.start_calls) is expect_started
    if expect_started:
        assert runner.stopped_processes == [runner.process]

    report = read_report(out_dir)
    assert report["passed"] is False
    assert failure_reason in report["failure_reasons"]


def test_sync_failure_reports_provenance_not_run_without_sampling_stale_runtime(
    tmp_path: Path,
) -> None:
    make_runtime_distribution(tmp_path)
    runner = FakeRunner(sync_result=CommandResult(returncode=9, stderr="sync failed"))
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 1
    assert runner.events == ["sync"]
    assert runner.start_calls == []
    report = read_report(out_dir)
    assert report["passed"] is False
    assert report["failure_reasons"] == ["sync_failed"]
    provenance = report["runtime_provenance"]
    assert provenance["failure_reasons"] == ["runtime_provenance_not_run:sync_failed"]
    assert provenance["runtime_hash"] is None
    assert report["runtime_hash"] is None
    assert provenance["wheel_name"] is None
    assert provenance["metadata_sha256"] is None


def test_sync_success_with_provenance_failure_does_not_start_server(tmp_path):
    write_project_contract(tmp_path)
    runner = FakeRunner()
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 1
    assert runner.start_calls == []
    assert len(runner.sync_calls) == 1
    report = read_report(out_dir)
    assert report["passed"] is False
    assert any(
        reason.startswith("runtime_provenance_failed:")
        for reason in report["failure_reasons"]
    )
    assert "runtime_provenance" in report
    assert report["runtime_provenance"]["runtime_hash"] is None


def test_cli_import_check_failure_does_not_start_server(tmp_path):
    make_runtime_distribution(tmp_path)
    runner = FakeRunner(cli_check_result=CommandResult(returncode=7, stderr="boom"))
    out_dir = tmp_path / "artifacts" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 1
    assert runner.start_calls == []
    assert len(runner.sync_calls) == 2
    report = read_report(out_dir)
    assert report["passed"] is False
    assert "cli_import_check_failed" in report["failure_reasons"]
    assert report["cli_check_command"] == runner.sync_calls[1]["command"]
    assert report["cli_check_returncode"] == 7


def test_rejects_out_dir_under_val_data_without_running_or_writing_report(tmp_path):
    runner = FakeRunner()
    out_dir = tmp_path / "val-data" / "runtime-smoke"

    exit_code = main(["--repo-root", str(tmp_path), "--out-dir", str(out_dir)], runner=runner)

    assert exit_code == 1
    assert runner.sync_calls == []
    assert runner.start_calls == []
    assert not (out_dir / "report.json").exists()
