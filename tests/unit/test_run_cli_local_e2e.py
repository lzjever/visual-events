from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import json
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "run_cli_local_e2e.py"


@dataclass
class FakeResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    collection_incomplete: bool = False


@dataclass
class FakeHealth:
    passed: bool
    failure_reason: str | None = None
    healthz_pid: int | None = None
    healthz_identity_verified: bool = False


class FakeProcess:
    def __init__(
        self,
        name: str,
        pid: int,
        *,
        stdout: str = "",
        stderr: str = "",
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        collection_incomplete: bool = False,
    ) -> None:
        self.name = name
        self.pid = pid
        self.returncode: int | None = None
        self.stopped = False
        self.stdout_lines = stdout.splitlines()
        self.stderr_lines = stderr.splitlines()
        self.stdout_tail = stdout if stdout else f"{name} stdout tail"
        self.stderr_tail = stderr if stderr else f"{name} stderr tail"
        self.stdout_truncated = stdout_truncated
        self.stderr_truncated = stderr_truncated
        self.collection_incomplete = collection_incomplete

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.stopped = True
        if self.returncode is None:
            self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.stopped = True
        self.returncode = -9


class FakeRunner:
    def __init__(
        self,
        *,
        health: FakeHealth | None = None,
        sync_results: dict[str, FakeResult | list[FakeResult]] | None = None,
        process_results: dict[str, FakeResult | list[FakeResult]] | None = None,
        initial_process_returncodes: dict[str, int] | None = None,
    ) -> None:
        self.health = health or FakeHealth(
            passed=True,
            healthz_pid=1001,
            healthz_identity_verified=True,
        )
        self.sync_results = sync_results or {}
        self.process_results = process_results or {}
        self.initial_process_returncodes = initial_process_returncodes or {}
        self.events: list[tuple[str, str]] = []
        self.started: dict[str, FakeProcess] = {}
        self.commands: dict[str, list[str]] = {}
        self.command_history: list[tuple[str, list[str]]] = []
        self.envs: dict[str, dict[str, str]] = {}
        self.health_urls: list[str] = []
        self._next_pid = 1001

    def _lookup_result(
        self,
        results: dict[str, FakeResult | list[FakeResult]],
        name: str,
    ) -> FakeResult | None:
        if name in results:
            return self._consume_result(results, name)
        base_name = name.split(":", 1)[0]
        if base_name in results:
            return self._consume_result(results, base_name)
        return None

    def _consume_result(
        self,
        results: dict[str, FakeResult | list[FakeResult]],
        name: str,
    ) -> FakeResult | None:
        result = results[name]
        if isinstance(result, list):
            if not result:
                return None
            return result.pop(0)
        return result

    def _default_head_result(self, name: str) -> FakeResult | None:
        if not name.startswith("head_publisher:"):
            return None
        command = self.commands.get(name, [])
        state = arg_value(command, "--state") or name.split(":", 1)[1]
        count = int(arg_value(command, "--count") or "5")
        return FakeResult(0, stdout=head_publisher_stdout(state, count=count))

    def start_process(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        name: str,
    ) -> FakeProcess:
        del cwd
        result = self._lookup_result(self.process_results, name) or FakeResult(0)
        process = FakeProcess(
            name,
            self._next_pid,
            stdout=result.stdout,
            stderr=result.stderr,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            collection_incomplete=result.collection_incomplete,
        )
        if name in self.initial_process_returncodes:
            process.returncode = self.initial_process_returncodes[name]
        self._next_pid += 1
        self.started[name] = process
        self.commands[name] = command
        self.command_history.append((name, command))
        self.envs[name] = env
        self.events.append(("start", name))
        return process

    def run_sync(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        name: str,
        timeout_s: float | None = None,
    ) -> FakeResult:
        del cwd, timeout_s
        self.commands[name] = command
        self.command_history.append((name, command))
        self.envs[name] = env
        self.events.append(("run_sync", name))
        return self._lookup_result(self.sync_results, name) or FakeResult(0)

    def wait_healthz(
        self,
        url: str,
        process: FakeProcess,
        *,
        timeout_s: float,
        interval_s: float,
    ) -> FakeHealth:
        del process, timeout_s, interval_s
        self.health_urls.append(url)
        self.events.append(("healthz", "server"))
        return self.health

    def wait_process(
        self,
        process: FakeProcess,
        *,
        name: str,
        timeout_s: float | None = None,
    ) -> FakeResult:
        del timeout_s
        self.events.append(("wait", name))
        result = (
            self._lookup_result(self.sync_results, name)
            or self._lookup_result(self.process_results, name)
            or self._default_head_result(name)
            or FakeResult(0)
        )
        process.returncode = result.returncode
        return result

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", str(seconds)))

    def stop_process(self, process: FakeProcess) -> None:
        self.events.append(("stop", process.name))
        process.terminate()


def import_runner_module() -> Any:
    try:
        return importlib.import_module("tools.run_cli_local_e2e")
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected tools.run_cli_local_e2e module: {exc}")


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


@pytest.fixture(autouse=True)
def repo_local_runtime_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_runner_module()
    write_project_contract(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)


def write_frame(path: Path, payload: bytes = b"jpeg") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def record_digest(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"sha256={encoded}"


def record_line(site_packages: Path, path: Path, *, digest: str | None = None) -> str:
    rel_path = os.path.relpath(path, site_packages).replace(os.sep, "/")
    digest_value = record_digest(path) if digest is None else digest
    return f"{rel_path},{digest_value},{path.stat().st_size}\n"


def rewrite_runtime_record(
    paths: dict[str, Path],
    *,
    include_server: bool = True,
    include_cli: bool = True,
    server_digest: str | None = None,
    cli_digest: str | None = None,
) -> None:
    site_packages = paths["site_packages"]
    record = paths["record"]
    lines = [
        record_line(site_packages, paths["metadata"]),
        record_line(site_packages, paths["entry_points"]),
    ]
    if include_server:
        lines.append(record_line(site_packages, paths["server_bin"], digest=server_digest))
    if include_cli:
        lines.append(record_line(site_packages, paths["cli_bin"], digest=cli_digest))
    lines.append(f"{os.path.relpath(record, site_packages).replace(os.sep, '/')},,\n")
    record.write_text("".join(lines), encoding="utf-8")


def make_runtime_distribution(
    tmp_path: Path,
    *,
    name: str = "visual-events-server",
    version: str = "0.1.0",
    root_name: str = "runtime",
) -> dict[str, Path]:
    runtime_root = tmp_path / root_name
    venv = runtime_root / "venv"
    bin_dir = venv / "bin"
    server_bin = make_executable(bin_dir / "visual-events-server", "#!/bin/sh\nexit 0\n")
    cli_bin = make_executable(bin_dir / "visual-events-cli", "#!/bin/sh\nexit 0\n")
    site_packages = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    dist_info = site_packages / "visual_events_server-0.1.0.dist-info"
    metadata = dist_info / "METADATA"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    entry_points = dist_info / "entry_points.txt"
    entry_points.write_text(
        "\n".join(
            [
                "[console_scripts]",
                "visual-events-server = visual_events_server.app:main",
                "visual-events-cli = visual_events_cli.main:main",
                "",
            ]
        ),
        encoding="utf-8",
    )
    record = dist_info / "RECORD"
    result = {
        "runtime_root": runtime_root,
        "runtime_venv": venv,
        "runtime_bin_dir": bin_dir,
        "site_packages": site_packages,
        "server_bin": server_bin,
        "cli_bin": cli_bin,
        "dist_info": dist_info,
        "metadata": metadata,
        "entry_points": entry_points,
        "record": record,
        "direct_url": dist_info / "direct_url.json",
    }
    rewrite_runtime_record(result)
    return result


def make_case(tmp_path: Path) -> dict[str, Path]:
    data_dir = tmp_path / "val-data"
    write_frame(data_dir / "scene-a" / "001.jpg")
    write_frame(data_dir / "scene-b" / "001.jpeg")
    out = tmp_path / "artifacts" / "cli-local-e2e.json"
    runtime = make_runtime_distribution(tmp_path)
    build_dir = tmp_path / "build"
    dds_bridge = make_executable(build_dir / "visual_events_dds_bridge")
    return {
        "data_dir": data_dir,
        "out": out,
        "server_bin": runtime["server_bin"],
        "cli_bin": runtime["cli_bin"],
        "runtime_root": runtime["runtime_root"],
        "runtime_venv": runtime["runtime_venv"],
        "runtime_bin_dir": runtime["runtime_bin_dir"],
        "site_packages": runtime["site_packages"],
        "dist_info": runtime["dist_info"],
        "metadata": runtime["metadata"],
        "entry_points": runtime["entry_points"],
        "record": runtime["record"],
        "direct_url": runtime["direct_url"],
        "build_dir": build_dir,
        "dds_bridge": dds_bridge,
    }


def authoritative_manifest_for_data(manifest_module: Any, data_dir: Path) -> dict[str, Any]:
    inventory = manifest_module.generate_effective_manifest(data_dir)
    return {
        "schema_version": 1,
        "fps": 10.0,
        "scene_count": inventory["scene_count"],
        "frame_count": inventory["frame_count"],
        "scenes": [
            {
                "scene_name": scene["scene_name"],
                "frame_count": scene["frame_count"],
                "scene_sha256": scene["scene_sha256"],
            }
            for scene in inventory["scenes"]
        ],
        "oracle": {
            "expected_event_timeline": {
                "source": "oracle/events.json",
                "version": "events-v1",
            },
            "expected_attention_target_timeline": {
                "source": "oracle/attention.json",
                "rule": "largest_stable_person_v1",
            },
        },
    }


def write_authoritative_manifest(module: Any, paths: dict[str, Path]) -> Path:
    manifest_path = paths["data_dir"] / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            authoritative_manifest_for_data(
                module.cli_local_e2e_manifest,
                paths["data_dir"],
            ),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def base_argv(paths: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--data-dir",
        os.fspath(paths["data_dir"]),
        "--out",
        os.fspath(paths["out"]),
        "--server-bin",
        os.fspath(paths["server_bin"]),
        "--cli-bin",
        os.fspath(paths["cli_bin"]),
        "--build-dir",
        os.fspath(paths["build_dir"]),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        *extra,
    ]


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arg_value(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    try:
        return command[index + 1]
    except IndexError:
        return None


def head_publisher_stdout(
    state: str,
    *,
    count: int = 5,
    mapped_state: str | None = None,
    mapped_valid: bool | None = None,
    dds_valid: bool | None = None,
) -> str:
    if mapped_state is None:
        mapped_state = state
    if mapped_valid is None:
        mapped_valid = state != "unknown"
    if dds_valid is None:
        dds_valid = state != "unknown"
    payload = {
        "protocol_version": 1,
        "type": "status",
        "code": "publish_test_head_state_ok",
        "published": count,
        "state": state,
        "head_state_topic": "/robot/head_state",
        "dds_valid": dds_valid,
        "mapped_valid": mapped_valid,
        "mapped_state": mapped_state,
    }
    return json.dumps(payload, separators=(",", ":")) + "\n"


def successful_runner(
    *,
    cli_stdout: str = "",
    cli_stderr: str = "",
) -> FakeRunner:
    return FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        },
        process_results={
            "cli": FakeResult(0, stdout=cli_stdout, stderr=cli_stderr),
        },
    )


def botified_frame(event: str = "person_waving", event_id: str = "front:evt_000456") -> str:
    payload = {
        "id": f"visual:{event_id}",
        "urgency": "normal",
        "timeout_secs": 8,
        "request": f"event={event} camera=front track_id=7 confidence=0.86",
        "expect": "ack",
    }
    inner = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"<botified>{inner}</botified>\n"


def gaze_jsonl(*payloads: dict[str, Any]) -> str:
    return "".join(
        json.dumps(payload, separators=(",", ":")) + "\n"
        for payload in payloads
    )


def assert_explicit_partial_smoke_fields(report: dict[str, Any], paths: dict[str, Path]) -> None:
    assert report["server_bin"] == os.fspath(paths["server_bin"])
    assert report["cli_bin"] == os.fspath(paths["cli_bin"])
    assert report["build_dir"] == os.fspath(paths["build_dir"])
    assert report["dds_bridge_bin"] == os.fspath(paths["dds_bridge"])
    assert report["head_state"]["required"] is True
    assert report["head_state"]["publisher_mode"] == "required"


def assert_runtime_provenance_success(
    report: dict[str, Any],
    paths: dict[str, Path],
    *,
    config_hash: str | None,
    wheel_name: str = "visual-events-server",
) -> None:
    assert report["server_bin_is_runtime_venv"] is True
    assert report["cli_bin_is_runtime_venv"] is True
    assert report["wheel_name"] == wheel_name
    assert report["wheel_version"] == "0.1.0"
    assert re.fullmatch(r"[0-9a-f]{64}", report["runtime_hash"])
    assert report["config_hash"] == config_hash
    provenance = report["runtime_provenance"]
    assert provenance["server_bin"] == os.fspath(paths["server_bin"])
    assert provenance["cli_bin"] == os.fspath(paths["cli_bin"])
    assert provenance["runtime_venv"] == os.fspath(paths["runtime_venv"])
    assert provenance["runtime_bin_dir"] == os.fspath(paths["runtime_bin_dir"])
    assert provenance["dist_info_dir"] == os.fspath(paths["dist_info"])
    assert provenance["metadata_path"] == os.fspath(paths["metadata"])
    assert provenance["metadata_sha256"] == sha256_file(paths["metadata"])
    assert provenance["entry_points_path"] == os.fspath(paths["entry_points"])
    assert provenance["entry_points_sha256"] == sha256_file(paths["entry_points"])
    assert provenance["record_path"] == os.fspath(paths["record"])
    assert provenance["record_sha256"] == sha256_file(paths["record"])
    assert provenance["server_entry_point"] == "visual_events_server.app:main"
    assert provenance["cli_entry_point"] == "visual_events_cli.main:main"
    assert provenance["direct_url_path"] is None
    assert provenance["direct_url_editable"] is False
    assert provenance["server_script_sha256"] == sha256_file(paths["server_bin"])
    assert provenance["cli_script_sha256"] == sha256_file(paths["cli_bin"])
    assert provenance["server_record_sha256"] == record_digest(paths["server_bin"])
    assert provenance["cli_record_sha256"] == record_digest(paths["cli_bin"])
    assert provenance["server_bin_is_runtime_venv"] is True
    assert provenance["cli_bin_is_runtime_venv"] is True
    assert provenance["same_runtime_venv"] is True
    assert provenance["wheel_name"] == wheel_name
    assert provenance["wheel_version"] == "0.1.0"
    assert provenance["runtime_hash"] == report["runtime_hash"]
    assert provenance["config_hash"] == config_hash
    assert provenance["failure_reasons"] == []


def test_importable_and_source_audit_for_forbidden_imports() -> None:
    module = import_runner_module()
    assert callable(module.main)

    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    denied = {
        "visual_events_cli",
        "visual_events_server",
        "unitree",
        "cyclonedds",
        "fastdds",
        "rti",
        "rclpy",
        "torch",
        "ultralytics",
    }
    assert imported_roots.isdisjoint(denied)


def test_argparse_rejects_missing_required_domain_or_network(capsys: pytest.CaptureFixture[str]) -> None:
    module = import_runner_module()

    with pytest.raises(SystemExit) as exc:
        module.parse_args(
            [
                "--data-dir",
                "/tmp/val-data",
                "--out",
                "/tmp/out.json",
                "--server-bin",
                "/tmp/server",
                "--cli-bin",
                "/tmp/cli",
            ]
        )

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "--dds-domain" in captured.err
    assert "--dds-network" in captured.err


def test_preflight_rejects_non_loopback_without_allow_and_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)

    rc = module.main(base_argv(paths, "--dds-network", "eth0"), runner=FakeRunner())

    captured = capsys.readouterr()
    assert rc != 0
    assert captured.out == ""
    assert paths["out"].exists()
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any("--allow-non-loopback-dds" in reason for reason in report["failure_reasons"])
    assert_runtime_provenance_success(report, paths, config_hash=None)
    assert report["server_exit_code"] is None
    assert report["cli_exit_code"] is None


def test_preflight_rejects_out_under_data_dir_without_writing(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["out"] = paths["data_dir"] / "reports" / "bad.json"

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    assert not paths["out"].exists()


def test_command_construction_uses_required_args_and_wrapper_paths(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout=(
                    '{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n'
                ),
            )
        }
    )

    rc = module.main(
        base_argv(
            paths,
            "--server",
            "ws://127.0.0.1:8877/v1/stream",
            "--scene",
            "scene-b",
            "--frame-count",
            "7",
            "--image-hz",
            "12.5",
            "--head-state",
            "moving",
            "--head-state-hz",
            "9.5",
            "--gaze-count",
            "1",
            "--gaze-timeout-ms",
            "1234",
        ),
        runner=runner,
    )

    assert rc == 0
    tools_dir = REPO_ROOT / "tools"
    assert runner.commands["server"] == [
        os.fspath(paths["server_bin"]),
        "--host",
        "127.0.0.1",
        "--port",
        "8877",
    ]
    assert runner.commands["cli"] == [
        os.fspath(paths["cli_bin"]),
        "--dds-runtime",
        "bridge",
        "--dds-bridge-bin",
        os.fspath(paths["dds_bridge"]),
        "--server",
        "ws://127.0.0.1:8877/v1/stream",
        "--camera",
        "front",
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        "--image-topic",
        "/camera/image/jpeg",
        "--head-state-topic",
        "/robot/head_state",
        "--gaze-topic",
        "/visual_events/gaze_target",
    ]
    assert runner.commands["gaze_subscriber"] == [
        os.fspath(module.sys.executable),
        os.fspath(tools_dir / "subscribe_test_gaze_targets.py"),
        "--build-dir",
        os.fspath(paths["build_dir"]),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        "--count",
        "1",
        "--timeout-ms",
        "1234",
        "--gaze-topic",
        "/visual_events/gaze_target",
    ]
    assert runner.commands["image_publisher:moving"] == [
        os.fspath(module.sys.executable),
        os.fspath(tools_dir / "publish_test_dds_images.py"),
        "--build-dir",
        os.fspath(paths["build_dir"]),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        "--input",
        os.fspath(paths["data_dir"] / "scene-b"),
        "--count",
        "7",
        "--hz",
        "12.5",
        "--camera-name",
        "image",
        "--camera-topic",
        "/camera/image/jpeg",
    ]
    assert runner.commands["head_publisher:moving"] == [
        os.fspath(module.sys.executable),
        os.fspath(tools_dir / "publish_test_head_state.py"),
        "--build-dir",
        os.fspath(paths["build_dir"]),
        "--dds-domain",
        "57",
        "--dds-network",
        "lo",
        "--state",
        "moving",
        "--count",
        "7",
        "--hz",
        "9.5",
        "--head-state-topic",
        "/robot/head_state",
    ]


def test_default_replay_mode_remains_partial_smoke_counts(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    write_frame(paths["data_dir"] / "scene-a" / "002.jpg")
    write_frame(paths["data_dir"] / "scene-a" / "003.jpeg")
    runner = successful_runner()

    rc = module.main(base_argv(paths, "--head-state", "stationary"), runner=runner)

    assert rc == 0
    assert arg_value(runner.commands["image_publisher:stationary"], "--count") == "5"
    assert arg_value(runner.commands["head_publisher:stationary"], "--count") == "5"
    assert arg_value(runner.commands["gaze_subscriber"], "--count") == "1"
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "partial_smoke_pass"
    assert report["scene_replay_mode"] == "partial"
    assert report["selected_scene_frame_count"] == 3
    assert report["published_frames"] == 5
    assert report["gaze"]["expected_count"] == 1
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False


def test_full_scene_derives_frame_and_default_gaze_count_from_selected_scene(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    write_frame(paths["data_dir"] / "scene-a" / "002.jpg")
    write_frame(paths["data_dir"] / "scene-a" / "003.jpeg")
    write_frame(paths["data_dir"] / "scene-b" / "002.jpg")
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout=gaze_jsonl(
                    {
                        "type": "gaze_target",
                        "camera": "front",
                        "state": "tracking",
                        "valid": True,
                    },
                    {
                        "type": "gaze_target",
                        "camera": "front",
                        "state": "lost",
                        "valid": False,
                    },
                    {
                        "type": "gaze_target",
                        "camera": "front",
                        "state": "tracking",
                        "valid": True,
                    },
                ),
            )
        }
    )

    rc = module.main(
        base_argv(
            paths,
            "--full-scene",
            "--scene",
            "scene-a",
            "--head-state",
            "stationary",
        ),
        runner=runner,
    )

    assert rc == 0
    assert arg_value(runner.commands["image_publisher:stationary"], "--count") == "3"
    assert arg_value(runner.commands["head_publisher:stationary"], "--count") == "3"
    assert arg_value(runner.commands["gaze_subscriber"], "--count") == "3"
    report = load_report(paths["out"])
    assert report["scene_replay_mode"] == "full_scene"
    assert report["selected_scene_frame_count"] == 3
    assert report["published_frames"] == 3
    assert report["gaze"]["expected_count"] == 3
    assert report["pc_local_e2e_status"] == "partial_smoke_pass"
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False


def test_full_scene_rejects_explicit_frame_count(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(
        base_argv(paths, "--full-scene", "--frame-count", "5"),
        runner=runner,
    )

    assert rc == 2
    assert runner.events == []
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any(
        "--full-scene cannot be combined with --frame-count" in reason
        for reason in report["failure_reasons"]
    )


def test_full_scene_default_gaze_count_shortfall_fails_slice(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    write_frame(paths["data_dir"] / "scene-a" / "002.jpg")
    write_frame(paths["data_dir"] / "scene-a" / "003.jpeg")
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout=gaze_jsonl(
                    {
                        "type": "gaze_target",
                        "camera": "front",
                        "state": "tracking",
                        "valid": True,
                    },
                    {
                        "type": "gaze_target",
                        "camera": "front",
                        "state": "lost",
                        "valid": False,
                    },
                ),
            )
        }
    )

    rc = module.main(
        base_argv(
            paths,
            "--full-scene",
            "--scene",
            "scene-a",
            "--head-state",
            "stationary",
        ),
        runner=runner,
    )

    assert rc != 0
    report = load_report(paths["out"])
    assert report["scene_replay_mode"] == "full_scene"
    assert report["selected_scene_frame_count"] == 3
    assert report["published_frames"] == 3
    assert report["gaze"]["expected_count"] == 3
    assert report["gaze"]["accepted_count"] == 2
    assert "gaze_target_count_shortfall:segment=stationary" in report["failure_reasons"]
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert report["slice_pass"] is False


def test_all_scenes_full_scene_runs_each_manifest_scene_with_scene_frame_counts(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    write_frame(paths["data_dir"] / "scene-a" / "002.jpg")
    write_frame(paths["data_dir"] / "scene-a" / "003.jpeg")
    write_frame(paths["data_dir"] / "scene-b" / "002.jpg")
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": [
                FakeResult(
                    0,
                    stdout=gaze_jsonl(
                        {
                            "type": "gaze_target",
                            "camera": "front",
                            "state": "tracking",
                            "valid": True,
                        },
                        {
                            "type": "gaze_target",
                            "camera": "front",
                            "state": "lost",
                            "valid": False,
                        },
                        {
                            "type": "gaze_target",
                            "camera": "front",
                            "state": "tracking",
                            "valid": True,
                        },
                    ),
                ),
                FakeResult(
                    0,
                    stdout=gaze_jsonl(
                        {
                            "type": "gaze_target",
                            "camera": "front",
                            "state": "tracking",
                            "valid": True,
                        },
                        {
                            "type": "gaze_target",
                            "camera": "front",
                            "state": "lost",
                            "valid": False,
                        },
                    ),
                ),
            ]
        }
    )

    rc = module.main(
        base_argv(paths, "--full-scene", "--all-scenes", "--head-state", "stationary"),
        runner=runner,
    )

    assert rc == 0
    image_commands = [
        command
        for name, command in runner.command_history
        if name == "image_publisher:stationary"
    ]
    head_commands = [
        command
        for name, command in runner.command_history
        if name == "head_publisher:stationary"
    ]
    gaze_commands = [
        command for name, command in runner.command_history if name == "gaze_subscriber"
    ]
    assert [arg_value(command, "--input") for command in image_commands] == [
        os.fspath(paths["data_dir"] / "scene-a"),
        os.fspath(paths["data_dir"] / "scene-b"),
    ]
    assert [arg_value(command, "--count") for command in image_commands] == ["3", "2"]
    assert [arg_value(command, "--count") for command in head_commands] == ["3", "2"]
    assert [arg_value(command, "--count") for command in gaze_commands] == ["3", "2"]

    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "full_scene_matrix_pass"
    assert report["scene_replay_mode"] == "full_scene_matrix"
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["oracle_evaluated"] is False
    assert report["oracle_evaluation_passed"] is None
    assert "full_scene_matrix" not in report["not_covered"]
    for gap in ["oracle", "latency_p95_p99", "fault_matrix", "soak", "release_report"]:
        assert gap in report["not_covered"]
    assert [
        (
            result["scene"],
            result["selected_scene_frame_count"],
            result["published_frames"],
            result["slice_pass"],
            result["failure_reasons"],
            result["gaze"]["expected_count"],
            result["head_state"]["count"],
            result["botified_stdout"]["source"],
        )
        for result in report["scene_results"]
    ] == [
        ("scene-a", 3, 3, True, [], 3, 3, "cli_stdout"),
        ("scene-b", 2, 2, True, [], 2, 2, "cli_stdout"),
    ]


def test_all_scenes_requires_full_scene(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(base_argv(paths, "--all-scenes"), runner=runner)

    assert rc == 2
    assert runner.events == []
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any("--all-scenes requires --full-scene" in reason for reason in report["failure_reasons"])


def test_all_scenes_rejects_explicit_scene(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(
        base_argv(paths, "--full-scene", "--all-scenes", "--scene", "scene-a"),
        runner=runner,
    )

    assert rc == 2
    assert runner.events == []
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any("--all-scenes cannot be combined with --scene" in reason for reason in report["failure_reasons"])


def test_all_scenes_matrix_failure_identifies_scene_and_keeps_other_results(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    write_frame(paths["data_dir"] / "scene-a" / "002.jpg")
    write_frame(paths["data_dir"] / "scene-a" / "003.jpeg")
    write_frame(paths["data_dir"] / "scene-b" / "002.jpg")
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": [
                FakeResult(
                    0,
                    stdout=gaze_jsonl(
                        {
                            "type": "gaze_target",
                            "camera": "front",
                            "state": "tracking",
                            "valid": True,
                        }
                    ),
                ),
                FakeResult(
                    0,
                    stdout=gaze_jsonl(
                        {
                            "type": "gaze_target",
                            "camera": "front",
                            "state": "tracking",
                            "valid": True,
                        },
                        {
                            "type": "gaze_target",
                            "camera": "front",
                            "state": "lost",
                            "valid": False,
                        },
                    ),
                ),
            ]
        }
    )

    rc = module.main(
        base_argv(paths, "--full-scene", "--all-scenes", "--head-state", "stationary"),
        runner=runner,
    )

    assert rc == 1
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "full_scene_matrix_failed"
    assert report["scene_replay_mode"] == "full_scene_matrix"
    assert report["slice_pass"] is False
    assert "scene_failed:scene-a" in report["failure_reasons"]
    assert [result["scene"] for result in report["scene_results"]] == [
        "scene-a",
        "scene-b",
    ]
    scene_a, scene_b = report["scene_results"]
    assert scene_a["slice_pass"] is False
    assert any("gaze_target_count_shortfall" in reason for reason in scene_a["failure_reasons"])
    assert scene_b["slice_pass"] is True
    assert scene_b["failure_reasons"] == []


def test_runtime_server_and_cli_do_not_inherit_ambient_python_env(
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
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(base_argv(paths, "--head-state", "stationary"), runner=runner)

    assert rc == 0
    for name in ("server", "cli"):
        env = runner.envs[name]
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
    assert runner.envs["image_publisher:stationary"]["PYTHONPATH"] == "/ambient/source"


def test_default_head_state_segments_run_all_states_and_report_ga_fields(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(base_argv(paths), runner=runner)

    assert rc == 0
    expected_states = ["stationary", "moving", "unknown"]
    for state in expected_states:
        command = runner.commands[f"head_publisher:{state}"]
        assert arg_value(command, "--state") == state
        assert arg_value(command, "--count") == "5"
        assert arg_value(command, "--hz") == "10"

    report = load_report(paths["out"])
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["head_state"]["required"] is True
    assert report["head_state"]["publisher_mode"] == "required"
    assert report["head_state"]["hz"] == 10.0
    assert report["head_state"]["count"] == 5
    assert [
        segment["requested_state"] for segment in report["head_state"]["segments"]
    ] == expected_states
    assert [segment["returncode"] for segment in report["head_state"]["segments"]] == [0, 0, 0]
    assert report["head_state"]["evidence_source"] == "synthetic_publisher_stdout"
    assert report["head_state_publisher_mode"] == "required"
    assert report["head_state_hz"] == 10.0
    assert report["head_state_stale_count"] is None
    assert report["head_state_unknown_ratio"] == 0.0
    assert report["head_state_segments"] == expected_states


def test_default_missing_manifest_partial_smoke_reports_contract_gap(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(
        base_argv(paths, "--head-state", "stationary"),
        runner=runner,
    )

    assert rc == 0
    assert runner.events
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "partial_smoke_pass"
    assert report["slice_pass"] is True
    assert report["manifest_source"] == "generated"
    assert report["manifest_authoritative"] is False
    assert report["manifest_validation_errors"] == []
    assert report["manifest_contract_required"] is False
    assert report["manifest_contract_satisfied"] is False
    assert report["manifest_contract_failure_reasons"] == [
        "manifest_source_not_file",
        "manifest_not_authoritative",
        "oracle_schema_missing",
        "oracle_schema_invalid",
    ]
    assert report["oracle_evaluated"] is False
    assert report["oracle_evaluation_passed"] is None


def test_require_authoritative_manifest_without_manifest_fails_before_runner(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(
        base_argv(paths, "--require-authoritative-manifest"),
        runner=runner,
    )

    assert rc == 2
    assert runner.events == []
    assert runner.started == {}
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert report["manifest_contract_required"] is True
    assert report["manifest_contract_satisfied"] is False
    assert report["manifest_contract_failure_reasons"] == [
        "manifest_source_not_file",
        "manifest_not_authoritative",
        "oracle_schema_missing",
        "oracle_schema_invalid",
    ]
    failure_reason = report["failure_reasons"][0]
    assert "authoritative manifest contract required but not satisfied" in failure_reason
    for reason in report["manifest_contract_failure_reasons"]:
        assert reason in failure_reason


def test_require_authoritative_manifest_with_invalid_json_manifest_reports_file_evidence(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    manifest_path = paths["data_dir"] / "manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")
    runner = successful_runner()

    rc = module.main(
        base_argv(
            paths,
            "--manifest",
            os.fspath(manifest_path),
            "--require-authoritative-manifest",
        ),
        runner=runner,
    )

    marker = module.MANIFEST_UNREADABLE_OR_INVALID
    assert rc == 2
    assert runner.events == []
    assert runner.started == {}
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert report["manifest_source"] == "file"
    assert report["manifest_path"] == os.fspath(manifest_path.resolve())
    assert report["manifest_sha256"] is None
    assert report["manifest_schema_version"] is None
    assert report["manifest_authoritative"] is False
    assert report["manifest_validation_errors"] == [marker]
    assert report["oracle_schema_present"] is False
    assert report["oracle_schema_valid"] is False
    assert report["oracle_summary"] is None
    assert report["scene_count"] is None
    assert report["frame_count"] is None
    assert report["effective_manifest"] is None
    assert report["manifest_contract_required"] is True
    assert report["manifest_contract_satisfied"] is False
    assert report["manifest_contract_failure_reasons"] == [
        "manifest_not_authoritative",
        f"manifest_validation_errors:{marker}",
        "oracle_schema_missing",
        "oracle_schema_invalid",
    ]
    assert report["oracle_evaluated"] is False
    assert report["oracle_evaluation_passed"] is None
    assert marker in report["failure_reasons"]
    assert "manifest JSON is invalid" in report["failure_reasons"][0]


def test_require_authoritative_manifest_with_legacy_manifest_fails_before_runner(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    manifest_path = paths["data_dir"] / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scene_count": 2,
                "frame_count": 2,
                "scenes": [{"name": "scene-a"}, {"name": "scene-b"}],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    runner = successful_runner()

    rc = module.main(
        base_argv(
            paths,
            "--manifest",
            os.fspath(manifest_path),
            "--require-authoritative-manifest",
        ),
        runner=runner,
    )

    assert rc == 2
    assert runner.events == []
    assert runner.started == {}
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert report["manifest_source"] == "file"
    assert report["manifest_authoritative"] is False
    assert report["manifest_contract_required"] is True
    assert report["manifest_contract_satisfied"] is False
    assert report["manifest_contract_failure_reasons"] == [
        "manifest_not_authoritative",
        "oracle_schema_missing",
        "oracle_schema_invalid",
    ]


def test_explicit_head_state_segments_override_single_head_state_shortcut(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(
        base_argv(
            paths,
            "--head-state",
            "moving",
            "--head-state-segments",
            "stationary,moving,unknown",
        ),
        runner=runner,
    )

    assert rc == 0
    assert {
        name
        for event, name in runner.events
        if event == "start" and name.startswith("head_publisher:")
    } == {
        "head_publisher:stationary",
        "head_publisher:moving",
        "head_publisher:unknown",
    }
    report = load_report(paths["out"])
    assert report["head_state_segments"] == ["stationary", "moving", "unknown"]


@pytest.mark.parametrize(
    "segments",
    [
        "",
        "stationary,,moving",
        "stationary,stationary",
        "stationary,bad",
    ],
)
def test_preflight_rejects_invalid_head_state_segments_and_writes_report(
    tmp_path: Path,
    segments: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)

    rc = module.main(
        base_argv(paths, "--head-state-segments", segments),
        runner=FakeRunner(),
    )

    assert rc != 0
    assert paths["out"].exists()
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert report["slice_pass"] is False
    assert any("--head-state-segments" in reason for reason in report["failure_reasons"])


def test_head_publisher_overlaps_image_replay_for_each_segment(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(base_argv(paths), runner=runner)

    assert rc == 0
    for state in ("stationary", "moving", "unknown"):
        segment_events = [
            event
            for event in runner.events
            if event[1] in {f"head_publisher:{state}", f"image_publisher:{state}"}
        ]
        assert segment_events == [
            ("start", f"head_publisher:{state}"),
            ("run_sync", f"image_publisher:{state}"),
            ("wait", f"head_publisher:{state}"),
        ]
    assert not any(event == ("run_sync", "head_publisher") for event in runner.events)


@pytest.mark.parametrize(
    ("process_result", "expected_reason"),
    [
        (
            FakeResult(7, stdout=head_publisher_stdout("moving"), stderr="head failed"),
            "head_publisher_failed",
        ),
        (
            FakeResult(0, stdout="{not-json}\n"),
            "head_publisher_malformed_json",
        ),
    ],
)
def test_segment_head_publisher_failure_or_malformed_json_fails_partial_slice(
    tmp_path: Path,
    process_result: FakeResult,
    expected_reason: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        },
        process_results={"head_publisher:moving": process_result},
    )

    rc = module.main(
        base_argv(paths, "--head-state-segments", "stationary,moving,unknown"),
        runner=runner,
    )

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert any(
        expected_reason in reason and "moving" in reason
        for reason in report["failure_reasons"]
    )
    moving_segment = next(
        segment
        for segment in report["head_state"]["segments"]
        if segment["requested_state"] == "moving"
    )
    assert moving_segment["returncode"] == process_result.returncode


@pytest.mark.parametrize(
    ("segments", "process_results", "expected_reason"),
    [
        (
            "stationary",
            {
                "head_publisher:stationary": FakeResult(
                    0,
                    stdout=head_publisher_stdout(
                        "stationary",
                        mapped_state="unknown",
                        mapped_valid=False,
                    ),
                )
            },
            "head_state_unknown_ratio_above_max",
        ),
        (
            "stationary",
            {
                "head_publisher:stationary": FakeResult(
                    0,
                    stdout=head_publisher_stdout(
                        "stationary",
                        mapped_state="moving",
                    ),
                )
            },
            "head_state_segment_state_mismatch:segment=stationary",
        ),
        (
            "moving",
            {
                "head_publisher:moving": FakeResult(
                    0,
                    stdout=head_publisher_stdout(
                        "moving",
                        mapped_state="stationary",
                    ),
                )
            },
            "head_state_segment_state_mismatch:segment=moving",
        ),
        (
            "unknown",
            {
                "head_publisher:unknown": FakeResult(
                    0,
                    stdout=head_publisher_stdout(
                        "unknown",
                        mapped_state="moving",
                        mapped_valid=True,
                        dds_valid=True,
                    ),
                )
            },
            "head_state_unknown_segment_missing_unknown_evidence",
        ),
    ],
)
def test_head_state_native_json_evidence_enforces_segment_mapping(
    tmp_path: Path,
    segments: str,
    process_results: dict[str, FakeResult],
    expected_reason: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        },
        process_results=process_results,
    )

    rc = module.main(base_argv(paths, "--head-state-segments", segments), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert any(expected_reason in reason for reason in report["failure_reasons"])


def test_legacy_scalar_fields_use_first_failing_segment(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        },
        process_results={
            "head_publisher:moving": FakeResult(
                7,
                stdout=head_publisher_stdout("moving"),
                stderr="moving head failed",
            ),
        },
    )

    rc = module.main(
        base_argv(paths, "--head-state-segments", "stationary,moving,unknown"),
        runner=runner,
    )

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["returncodes"]["head_publisher"] == 7
    assert report["returncodes"]["head_publishers"] == {
        "stationary": 0,
        "moving": 7,
        "unknown": 0,
    }
    assert '"state":"moving"' in report["stdout_stderr_tails"]["head_publisher"][
        "stdout_tail"
    ]
    assert "moving head failed" in report["stdout_stderr_tails"]["head_publisher"][
        "stderr_tail"
    ]
    assert '"state":"unknown"' in report["stdout_stderr_tails"]["head_publishers"][
        "unknown"
    ]["stdout_tail"]
    assert [segment["returncode"] for segment in report["head_state"]["segments"]] == [
        0,
        7,
        0,
    ]


def test_image_failure_reason_is_segment_specific(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            ),
            "image_publisher:moving": FakeResult(4, stderr="moving image failed"),
        }
    )

    rc = module.main(
        base_argv(paths, "--head-state-segments", "stationary,moving,unknown"),
        runner=runner,
    )

    assert rc != 0
    report = load_report(paths["out"])
    assert "image_publisher_failed:segment=moving" in report["failure_reasons"]
    assert "image_publisher_failed" not in report["failure_reasons"]
    assert report["returncodes"]["image_publishers"] == {
        "stationary": 0,
        "moving": 4,
        "unknown": 0,
    }


def test_head_state_hz_below_min_fails_partial_slice(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner()

    rc = module.main(base_argv(paths, "--head-state-hz", "8.5"), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert report["slice_pass"] is False
    assert "head_state_hz_below_min" in report["failure_reasons"]


def test_fake_runner_success_writes_partial_pass_and_cleans_up(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout=(
                    '{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n'
                    '{"type":"gaze_target","camera":"front","state":"lost","valid":false}\n'
                    '{"type":"other","camera":"front","state":"tracking"}\n'
                ),
            )
        }
    )

    rc = module.main(
        base_argv(paths, "--head-state", "stationary", "--gaze-count", "2"),
        runner=runner,
    )

    assert rc == 0
    assert runner.events == [
        ("start", "server"),
        ("healthz", "server"),
        ("start", "cli"),
        ("sleep", "0.5"),
        ("start", "gaze_subscriber"),
        ("start", "head_publisher:stationary"),
        ("run_sync", "image_publisher:stationary"),
        ("wait", "head_publisher:stationary"),
        ("wait", "gaze_subscriber"),
        ("stop", "cli"),
        ("stop", "server"),
    ]
    assert runner.started["cli"].stopped is True
    assert runner.started["server"].stopped is True
    report = load_report(paths["out"])
    assert report["report_type"] == "cli_local_e2e_smoke_v1"
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["pc_local_e2e_status"] == "partial_smoke_pass"
    assert report["failure_reasons"] == []
    assert report["manifest_source"] == "generated"
    assert report["scene_count"] == 2
    assert report["selected_scene"] == "scene-a"
    assert_explicit_partial_smoke_fields(report, paths)
    assert report["gaze"]["expected_count"] == 2
    assert report["gaze"]["received_lines"] == 3
    assert report["gaze"]["accepted_count"] == 2
    assert report["gaze"]["rejected_count"] == 1
    assert report["gaze"]["valid_count"] == 1
    assert report["gaze"]["invalid_count"] == 1
    assert report["gaze"]["state_counts"] == {"lost": 1, "tracking": 1}
    assert report["gaze"]["first_sample"]["state"] == "tracking"
    assert report["gaze"]["last_sample"]["state"] == "lost"
    assert report["botified_stdout"] == {
        "source": "cli_stdout",
        "required_frame_count": None,
        "line_count": 0,
        "frame_count": 0,
        "allowed_frame_count": 0,
        "pollution_count": 0,
        "parse_error_count": 0,
        "forbidden_event_count": 0,
        "event_counts": {},
        "first_frame": None,
        "last_frame": None,
        "collection_truncated": False,
        "collection_incomplete": False,
        "contract_violations": [],
    }
    assert "full_scene_matrix" in report["not_covered"]
    assert_runtime_provenance_success(report, paths, config_hash=None)
    assert report["server_exit_code"] == -15
    assert report["cli_exit_code"] == -15


def test_gaze_timestamps_report_latency_and_publish_hz_per_segment(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    gaze_stdout = gaze_jsonl(
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 1000,
            "publish_timestamp_ms": 1010,
        },
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 1100,
            "publish_timestamp_ms": 1125,
        },
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "lost",
            "valid": False,
            "frame_timestamp_ms": 1200,
            "publish_timestamp_ms": 1230,
        },
    )
    runner = FakeRunner(
        sync_results={"gaze_subscriber": FakeResult(0, stdout=gaze_stdout)}
    )

    rc = module.main(
        base_argv(
            paths,
            "--head-state-segments",
            "stationary,moving",
            "--gaze-count",
            "3",
        ),
        runner=runner,
    )

    assert rc == 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False

    expected_latency = {
        "available": True,
        "sample_count": 3,
        "invalid_sample_count": 0,
        "p50": 25.0,
        "p95": 30.0,
        "p99": 30.0,
    }
    expected_hz = 2 / ((1230 - 1010) / 1000)
    assert [segment["requested_head_state"] for segment in report["gaze_segments"]] == [
        "stationary",
        "moving",
    ]
    for segment in report["gaze_segments"]:
        summary = segment["summary"]
        assert summary["capture_to_gaze_publish_ms"] == expected_latency
        publish_hz = summary["gaze_publish_hz"]
        assert publish_hz["available"] is True
        assert publish_hz["sample_count"] == 3
        assert publish_hz["first_publish_timestamp_ms"] == 1010.0
        assert publish_hz["last_publish_timestamp_ms"] == 1230.0
        assert publish_hz["hz"] == pytest.approx(expected_hz)
        assert publish_hz["min_hz"] == 9.0
        assert publish_hz["pass"] is True

    evidence = report["evidence_summary"]
    assert evidence["gaze"]["total_expected_count"] == 6
    assert evidence["gaze"]["total_accepted_count"] == 6
    assert evidence["gaze"]["total_valid_count"] == 4
    assert evidence["gaze"]["total_invalid_count"] == 2
    assert evidence["gaze"]["total_rejected_count"] == 0
    assert evidence["gaze"]["per_segment_accepted_counts"] == {
        "moving": 3,
        "stationary": 3,
    }
    assert evidence["gaze"]["min_available_gaze_hz"] == pytest.approx(expected_hz)
    assert evidence["gaze"]["max_available_gaze_hz"] == pytest.approx(expected_hz)
    assert evidence["gaze"]["rate_pass"] is True
    expected_global_latency = {
        "available": True,
        "sample_count": 6,
        "invalid_sample_count": 0,
        "p50": 25.0,
        "p95": 30.0,
        "p99": 30.0,
    }
    expected_botified_latency = {
        "available": False,
        "sample_count": 0,
        "p50": None,
        "p95": None,
        "p99": None,
    }
    assert report["latency"] == {
        "capture_to_gaze_publish_ms": expected_global_latency,
        "capture_to_botified_stdout_ms": expected_botified_latency,
    }
    assert evidence["latency"] == report["latency"]


def test_missing_or_invalid_gaze_timestamps_are_unavailable_without_failure(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    gaze_stdout = gaze_jsonl(
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
        },
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "lost",
            "valid": False,
            "frame_timestamp_ms": 1000,
            "publish_timestamp_ms": 900,
        },
    )
    runner = FakeRunner(
        sync_results={"gaze_subscriber": FakeResult(0, stdout=gaze_stdout)}
    )

    rc = module.main(
        base_argv(paths, "--head-state", "stationary", "--gaze-count", "2"),
        runner=runner,
    )

    assert rc == 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is True
    assert report["failure_reasons"] == []
    latency = report["gaze"]["capture_to_gaze_publish_ms"]
    assert latency == {
        "available": False,
        "sample_count": 0,
        "invalid_sample_count": 2,
        "p50": None,
        "p95": None,
        "p99": None,
    }
    publish_hz = report["gaze"]["gaze_publish_hz"]
    assert publish_hz["available"] is False
    assert publish_hz["sample_count"] == 1
    assert publish_hz["hz"] is None
    assert publish_hz["pass"] is None
    assert report["evidence_summary"]["gaze"]["rate_pass"] is None
    assert report["latency"]["capture_to_gaze_publish_ms"] == latency
    assert report["latency"]["capture_to_botified_stdout_ms"] == {
        "available": False,
        "sample_count": 0,
        "p50": None,
        "p95": None,
        "p99": None,
    }
    assert report["evidence_summary"]["latency"] == report["latency"]


def test_huge_gaze_timestamp_delta_is_unavailable_without_partial_failure(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    gaze_stdout = gaze_jsonl(
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 1000,
            "publish_timestamp_ms": 1_700_000_000_000,
        },
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 1100,
            "publish_timestamp_ms": 1_700_000_000_100,
        },
    )
    runner = FakeRunner(
        sync_results={"gaze_subscriber": FakeResult(0, stdout=gaze_stdout)}
    )

    rc = module.main(
        base_argv(paths, "--head-state", "stationary", "--gaze-count", "2"),
        runner=runner,
    )

    assert rc == 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "partial_smoke_pass"
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert "latency_p95_p99" in report["not_covered"]
    assert report["failure_reasons"] == []

    latency = report["gaze"]["capture_to_gaze_publish_ms"]
    assert latency == {
        "available": False,
        "sample_count": 0,
        "invalid_sample_count": 2,
        "p50": None,
        "p95": None,
        "p99": None,
    }
    publish_hz = report["gaze"]["gaze_publish_hz"]
    assert publish_hz["available"] is True
    assert publish_hz["sample_count"] == 2
    assert publish_hz["first_publish_timestamp_ms"] == 1_700_000_000_000.0
    assert publish_hz["last_publish_timestamp_ms"] == 1_700_000_000_100.0
    assert publish_hz["hz"] == pytest.approx(10.0)
    assert publish_hz["pass"] is True
    assert report["evidence_summary"]["gaze"]["rate_pass"] is True
    assert report["latency"]["capture_to_gaze_publish_ms"] == latency
    assert report["evidence_summary"]["latency"] == report["latency"]


def test_non_monotonic_gaze_publish_timestamps_make_hz_unavailable_without_partial_failure(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    gaze_stdout = gaze_jsonl(
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 990,
            "publish_timestamp_ms": 1000,
        },
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 1990,
            "publish_timestamp_ms": 2000,
        },
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 1490,
            "publish_timestamp_ms": 1500,
        },
    )
    runner = FakeRunner(
        sync_results={"gaze_subscriber": FakeResult(0, stdout=gaze_stdout)}
    )

    rc = module.main(
        base_argv(paths, "--head-state", "stationary", "--gaze-count", "3"),
        runner=runner,
    )

    assert rc == 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "partial_smoke_pass"
    assert report["slice_pass"] is True
    assert report["failure_reasons"] == []

    latency = report["gaze"]["capture_to_gaze_publish_ms"]
    assert latency == {
        "available": True,
        "sample_count": 3,
        "invalid_sample_count": 0,
        "p50": 10.0,
        "p95": 10.0,
        "p99": 10.0,
    }
    publish_hz = report["gaze"]["gaze_publish_hz"]
    assert publish_hz == {
        "available": False,
        "sample_count": 3,
        "first_publish_timestamp_ms": 1000.0,
        "last_publish_timestamp_ms": 1500.0,
        "hz": None,
        "min_hz": 9.0,
        "pass": None,
    }
    assert report["evidence_summary"]["gaze"]["rate_pass"] is None
    assert report["latency"]["capture_to_gaze_publish_ms"] == latency
    assert report["evidence_summary"]["latency"] == report["latency"]


def test_low_available_gaze_publish_hz_fails_with_segment_reason(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    gaze_stdout = gaze_jsonl(
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 900,
            "publish_timestamp_ms": 1000,
        },
        {
            "type": "gaze_target",
            "camera": "front",
            "state": "tracking",
            "valid": True,
            "frame_timestamp_ms": 1900,
            "publish_timestamp_ms": 2000,
        },
    )
    runner = FakeRunner(
        sync_results={"gaze_subscriber": FakeResult(0, stdout=gaze_stdout)}
    )

    rc = module.main(
        base_argv(paths, "--head-state", "moving", "--gaze-count", "2"),
        runner=runner,
    )

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert report["slice_pass"] is False
    assert "gaze_publish_hz_below_min:segment=moving" in report["failure_reasons"]
    publish_hz = report["gaze"]["gaze_publish_hz"]
    assert publish_hz["available"] is True
    assert publish_hz["hz"] == pytest.approx(1.0)
    assert publish_hz["pass"] is False
    assert report["evidence_summary"]["gaze"]["rate_pass"] is False


def test_evidence_summary_aggregates_existing_report_fields(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    manifest_path = write_authoritative_manifest(module, paths)
    runner = successful_runner(cli_stdout=botified_frame())

    rc = module.main(
        base_argv(
            paths,
            "--manifest",
            os.fspath(manifest_path),
            "--head-state",
            "stationary",
        ),
        runner=runner,
    )

    assert rc == 0
    report = load_report(paths["out"])
    evidence = report["evidence_summary"]
    assert evidence["manifest"] == {
        "manifest_source": report["manifest_source"],
        "manifest_sha256": report["manifest_sha256"],
        "manifest_authoritative": report["manifest_authoritative"],
        "oracle_schema_present": report["oracle_schema_present"],
        "oracle_schema_valid": report["oracle_schema_valid"],
    }
    assert evidence["runtime"] == {
        "server_bin_is_runtime_venv": report["server_bin_is_runtime_venv"],
        "cli_bin_is_runtime_venv": report["cli_bin_is_runtime_venv"],
        "runtime_hash": report["runtime_hash"],
        "config_hash": report["config_hash"],
    }
    assert evidence["head_state"] == {
        "required": report["head_state"]["required"],
        "mode": report["head_state"]["publisher_mode"],
        "hz": report["head_state"]["hz"],
        "stale_count": report["head_state"]["stale_count"],
        "unknown_ratio": report["head_state"]["unknown_ratio"],
        "segments": report["head_state"]["segments"],
    }
    assert evidence["botified_stdout"] == {
        "allowed_frame_count": 1,
        "pollution_count": 0,
        "forbidden_event_count": 0,
        "parse_error_count": 0,
    }
    assert evidence["gaze"]["total_expected_count"] == report["gaze"]["expected_count"]
    assert evidence["gaze"]["total_accepted_count"] == report["gaze"]["accepted_count"]
    assert evidence["gaze"]["rate_pass"] is None
    assert evidence["latency"] == report["latency"]
    assert evidence["latency"]["capture_to_botified_stdout_ms"] == {
        "available": False,
        "sample_count": 0,
        "p50": None,
        "p95": None,
        "p99": None,
    }


def test_success_report_hashes_runtime_provenance_and_server_config(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    server_config = tmp_path / "runtime" / "config" / "s2.toml"
    server_config.parent.mkdir(parents=True, exist_ok=True)
    server_config.write_text("[server]\nmode = 'smoke'\n", encoding="utf-8")
    runner = successful_runner()

    rc = module.main(
        base_argv(
            paths,
            "--server-config",
            os.fspath(server_config),
            "--head-state",
            "stationary",
        ),
        runner=runner,
    )

    assert rc == 0
    report = load_report(paths["out"])
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert_runtime_provenance_success(
        report,
        paths,
        config_hash=sha256_file(server_config),
    )
    assert report["runtime_provenance"]["config_path"] == os.fspath(server_config)
    assert report["server_exit_code"] == -15
    assert report["cli_exit_code"] == -15


def test_partial_smoke_report_projects_authoritative_manifest_oracle_summary(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    manifest_path = write_authoritative_manifest(module, paths)
    runner = successful_runner()

    rc = module.main(
        base_argv(
            paths,
            "--manifest",
            os.fspath(manifest_path),
            "--require-authoritative-manifest",
            "--head-state",
            "stationary",
        ),
        runner=runner,
    )

    assert rc == 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["not_covered"] == module.NOT_COVERED
    assert report["manifest_source"] == "file"
    assert report["manifest_schema_version"] == 1
    assert report["manifest_authoritative"] is True
    assert report["manifest_validation_errors"] == []
    assert report["oracle_schema_present"] is True
    assert report["oracle_schema_valid"] is True
    assert report["manifest_contract_required"] is True
    assert report["manifest_contract_satisfied"] is True
    assert report["manifest_contract_failure_reasons"] == []
    assert report["oracle_evaluated"] is False
    assert report["oracle_evaluation_passed"] is None
    assert report["oracle_summary"] == {
        "expected_event_timeline": {
            "source": "oracle/events.json",
            "version": "events-v1",
        },
        "expected_attention_target_timeline": {
            "source": "oracle/attention.json",
            "rule": "largest_stable_person_v1",
        },
    }


def test_invalid_authoritative_manifest_fails_preflight_without_starting_runner(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    manifest_payload = authoritative_manifest_for_data(
        module.cli_local_e2e_manifest,
        paths["data_dir"],
    )
    manifest_payload["scenes"][0]["scene_sha256"] = "0" * 64
    manifest_path = paths["data_dir"] / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    runner = successful_runner()

    rc = module.main(
        base_argv(paths, "--manifest", os.fspath(manifest_path)),
        runner=runner,
    )

    assert rc == 2
    assert runner.events == []
    assert runner.started == {}
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert report["slice_pass"] is False
    assert report["manifest_authoritative"] is True
    assert report["manifest_validation_errors"] == [
        "scene_sha256_mismatch:scene-a"
    ]
    assert report["oracle_schema_present"] is True
    assert report["oracle_schema_valid"] is True
    assert report["manifest_contract_required"] is False
    assert report["manifest_contract_satisfied"] is False
    assert report["manifest_contract_failure_reasons"] == [
        "manifest_validation_errors:scene_sha256_mismatch:scene-a"
    ]
    assert report["oracle_evaluated"] is False
    assert report["oracle_evaluation_passed"] is None
    assert any(
        "manifest_validation_failed" in reason for reason in report["failure_reasons"]
    )


def test_require_authoritative_manifest_with_invalid_v1_manifest_fails_preflight(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    manifest_payload = authoritative_manifest_for_data(
        module.cli_local_e2e_manifest,
        paths["data_dir"],
    )
    manifest_payload["scene_count"] = 999
    manifest_path = paths["data_dir"] / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    runner = successful_runner()

    rc = module.main(
        base_argv(
            paths,
            "--manifest",
            os.fspath(manifest_path),
            "--require-authoritative-manifest",
        ),
        runner=runner,
    )

    assert rc == 2
    assert runner.events == []
    assert runner.started == {}
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert report["manifest_source"] == "file"
    assert report["manifest_authoritative"] is True
    assert report["manifest_validation_errors"] == ["scene_count_mismatch"]
    assert report["oracle_schema_present"] is True
    assert report["oracle_schema_valid"] is True
    assert report["manifest_contract_required"] is True
    assert report["manifest_contract_satisfied"] is False
    assert report["manifest_contract_failure_reasons"] == [
        "manifest_validation_errors:scene_count_mismatch"
    ]
    failure_reason = report["failure_reasons"][0]
    assert "authoritative manifest contract required but not satisfied" in failure_reason
    assert "manifest_validation_errors:scene_count_mismatch" in failure_reason


@pytest.mark.parametrize(
    ("process_name", "returncode", "expected_reason", "report_field"),
    [
        ("cli", 7, "cli_exit_code_unexpected", "cli_exit_code"),
        ("server", 42, "server_exit_code_unexpected", "server_exit_code"),
    ],
)
def test_unexpected_server_or_cli_exit_code_fails_partial_slice(
    tmp_path: Path,
    process_name: str,
    returncode: int,
    expected_reason: str,
    report_field: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        },
        initial_process_returncodes={process_name: returncode},
    )

    rc = module.main(base_argv(paths, "--head-state", "stationary"), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert report[report_field] == returncode
    assert report["runtime_provenance"][report_field] == returncode
    assert expected_reason in report["failure_reasons"]


@pytest.mark.parametrize(
    ("server_bin_factory", "expected_reason"),
    [
        (
            lambda tmp_path: make_executable(tmp_path / "bin" / "server"),
            "server_bin_not_runtime_venv",
        ),
        (
            lambda tmp_path: make_executable(
                tmp_path / ".venv" / "bin" / "visual-events-server"
            ),
            "server_bin_not_runtime_venv",
        ),
        (
            lambda tmp_path: make_runtime_distribution(tmp_path / "other")["server_bin"],
            "server_bin_not_runtime_venv",
        ),
    ],
)
def test_preflight_rejects_server_bin_not_from_runtime_venv(
    tmp_path: Path,
    server_bin_factory: Any,
    expected_reason: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["server_bin"] = server_bin_factory(tmp_path)

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any(expected_reason in reason for reason in report["failure_reasons"])
    assert report["server_bin_is_runtime_venv"] is False
    assert report["cli_bin_is_runtime_venv"] is True
    assert report["server_exit_code"] is None
    assert report["cli_exit_code"] is None


@pytest.mark.parametrize(
    ("cli_bin_factory", "expected_reason"),
    [
        (
            lambda tmp_path, paths: make_executable(
                tmp_path / "runtime" / "venv" / "bin" / "cli"
            ),
            "cli_bin_not_runtime_venv",
        ),
        (
            lambda tmp_path, paths: make_runtime_distribution(
                tmp_path / "other",
            )["cli_bin"],
            "cli_bin_not_runtime_venv",
        ),
    ],
)
def test_preflight_rejects_cli_bin_not_same_runtime_venv(
    tmp_path: Path,
    cli_bin_factory: Any,
    expected_reason: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["cli_bin"] = cli_bin_factory(tmp_path, paths)

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any(expected_reason in reason for reason in report["failure_reasons"])
    assert report["server_bin_is_runtime_venv"] is True
    assert report["server_exit_code"] is None
    assert report["cli_exit_code"] is None


def test_preflight_rejects_missing_runtime_dist_info(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["metadata"].unlink()
    paths["entry_points"].unlink()
    paths["record"].unlink()
    paths["dist_info"].rmdir()

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any("runtime_dist_info_missing" in reason for reason in report["failure_reasons"])
    assert report["server_bin_is_runtime_venv"] is True
    assert report["cli_bin_is_runtime_venv"] is True
    assert report["wheel_name"] is None
    assert report["wheel_version"] is None
    assert report["runtime_hash"] is None


def test_preflight_accepts_normalized_runtime_metadata_name(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["metadata"].write_text("Name: visual_events_server\nVersion: 0.1.0\n", encoding="utf-8")

    rc = module.main(
        base_argv(paths, "--head-state", "stationary"),
        runner=successful_runner(),
    )

    assert rc == 0
    report = load_report(paths["out"])
    assert_runtime_provenance_success(
        report,
        paths,
        config_hash=None,
        wheel_name="visual_events_server",
    )
    assert report["runtime_provenance"]["wheel_name"] == "visual_events_server"


@pytest.mark.parametrize(
    "metadata_text",
    [
        "Name: not-visual-events-server\nVersion: 0.1.0\n",
        "Name: visual-events-server\nVersion: 9.9.9\n",
    ],
)
def test_preflight_rejects_runtime_metadata_name_or_version_mismatch(
    tmp_path: Path,
    metadata_text: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["metadata"].write_text(metadata_text, encoding="utf-8")

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any("runtime_metadata_mismatch" in reason for reason in report["failure_reasons"])
    assert report["server_bin_is_runtime_venv"] is True
    assert report["cli_bin_is_runtime_venv"] is True
    assert report["runtime_hash"] is None


def test_preflight_rejects_invalid_utf8_runtime_metadata(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["metadata"].write_bytes(b"Name: visual-events-server\nVersion: \xff\n")

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any("runtime_metadata_invalid" in reason for reason in report["failure_reasons"])
    assert "runtime_provenance" in report
    assert "runtime_metadata_invalid" in report["runtime_provenance"]["failure_reasons"]
    assert report["runtime_hash"] is None
    assert report["runtime_provenance"]["runtime_hash"] is None


def test_preflight_rejects_dist_info_relative_record_script_path(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    server_record_path = os.path.relpath(
        paths["server_bin"],
        paths["dist_info"],
    ).replace(os.sep, "/")
    record_self_path = os.path.relpath(
        paths["record"],
        paths["site_packages"],
    ).replace(os.sep, "/")
    paths["record"].write_text(
        "".join(
            [
                record_line(paths["site_packages"], paths["metadata"]),
                record_line(paths["site_packages"], paths["entry_points"]),
                (
                    f"{server_record_path},{record_digest(paths['server_bin'])},"
                    f"{paths['server_bin'].stat().st_size}\n"
                ),
                record_line(paths["site_packages"], paths["cli_bin"]),
                f"{record_self_path},,\n",
            ]
        ),
        encoding="utf-8",
    )

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any("server_script_record_missing" in reason for reason in report["failure_reasons"])
    assert "server_script_record_missing" in report["runtime_provenance"]["failure_reasons"]
    assert report["runtime_provenance"]["server_record_path"] is None
    assert report["runtime_hash"] is None


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda paths: paths["entry_points"].unlink(),
            "runtime_entry_points_missing",
        ),
        (
            lambda paths: paths["entry_points"].write_text(
                "\n".join(
                    [
                        "[console_scripts]",
                        "visual-events-server = wrong.module:main",
                        "visual-events-cli = visual_events_cli.main:main",
                        "",
                    ]
                ),
                encoding="utf-8",
            ),
            "runtime_entry_point_mismatch:visual-events-server",
        ),
        (
            lambda paths: rewrite_runtime_record(paths, include_cli=False),
            "cli_script_record_missing",
        ),
        (
            lambda paths: rewrite_runtime_record(
                paths,
                server_digest="sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            ),
            "server_script_record_sha256_mismatch",
        ),
        (
            lambda paths: paths["direct_url"].write_text(
                json.dumps({"url": "file:///repo", "dir_info": {"editable": True}}),
                encoding="utf-8",
            ),
            "runtime_direct_url_editable",
        ),
    ],
)
def test_preflight_rejects_invalid_runtime_distribution_provenance(
    tmp_path: Path,
    mutate: Any,
    expected_reason: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    mutate(paths)

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert report["pc_local_e2e_status"] == "preflight_failed"
    assert any(expected_reason in reason for reason in report["failure_reasons"])
    assert expected_reason in report["runtime_provenance"]["failure_reasons"]
    assert report["runtime_hash"] is None


def test_preflight_rejects_multiple_matching_runtime_dist_info(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    second_dist_info = paths["site_packages"] / "visual_events_server-0.1.1.dist-info"
    second_dist_info.mkdir()
    (second_dist_info / "METADATA").write_text(
        "Name: visual-events-server\nVersion: 0.1.0\n",
        encoding="utf-8",
    )

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert any("runtime_dist_info_ambiguous" in reason for reason in report["failure_reasons"])
    assert "runtime_dist_info_ambiguous" in report["runtime_provenance"]["failure_reasons"]
    assert report["runtime_hash"] is None


def test_preflight_missing_cli_executable_populates_nested_provenance_failure(
    tmp_path: Path,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    paths["cli_bin"].unlink()

    rc = module.main(base_argv(paths), runner=FakeRunner())

    assert rc != 0
    report = load_report(paths["out"])
    assert any("cli-bin not found" in reason for reason in report["failure_reasons"])
    provenance = report["runtime_provenance"]
    assert provenance["cli_script_sha256"] is None
    assert "cli_executable_missing" in provenance["failure_reasons"]
    assert report["runtime_hash"] is None
    assert provenance["runtime_hash"] is None


def test_botified_stdout_valid_frame_is_reported_without_required_count(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner(cli_stdout=botified_frame())

    rc = module.main(base_argv(paths), runner=runner)

    assert rc == 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is True
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    botified = report["botified_stdout"]
    assert botified["source"] == "cli_stdout"
    assert botified["required_frame_count"] is None
    assert botified["line_count"] == 1
    assert botified["frame_count"] == 1
    assert botified["allowed_frame_count"] == 1
    assert botified["event_counts"] == {"person_waving": 1}
    assert botified["pollution_count"] == 0
    assert botified["parse_error_count"] == 0
    assert botified["forbidden_event_count"] == 0
    assert botified["contract_violations"] == []
    assert botified["first_frame"] == {
        "line": 1,
        "event": "person_waving",
        "payload": {
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "request": "event=person_waving camera=front track_id=7 confidence=0.86",
            "expect": "ack",
        },
    }
    assert botified["last_frame"] == botified["first_frame"]


@pytest.mark.parametrize(
    "cli_stdout",
    [
        '{"type":"gaze_target","camera":"front","state":"tracking"}\n',
        "status: connected to visual events service\n",
        "visual_state frame=1 gaze_target=front:track-7\n",
    ],
)
def test_botified_stdout_pollution_fails_partial_slice(
    tmp_path: Path,
    cli_stdout: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner(cli_stdout=cli_stdout)

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert "botified_stdout_pollution" in report["failure_reasons"]
    assert report["botified_stdout"]["pollution_count"] == 1
    assert report["botified_stdout"]["frame_count"] == 0


def test_botified_stdout_malformed_frame_fails_partial_slice(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner(cli_stdout="<botified>{not-json}</botified>\n")

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert "botified_stdout_parse_errors" in report["failure_reasons"]
    assert report["botified_stdout"]["parse_error_count"] == 1
    assert report["botified_stdout"]["pollution_count"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "event": "person_waving",
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "expect": "ack",
        },
        {
            "id": "front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "request": "event=person_waving camera=front",
            "expect": "ack",
        },
        {
            "id": "visual:front:evt_000456",
            "urgency": "urgent",
            "timeout_secs": 8,
            "request": "event=person_waving camera=front",
            "expect": "ack",
        },
        {
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": "8",
            "request": "event=person_waving camera=front",
            "expect": "ack",
        },
        {
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "request": "event=unknown_visual_event camera=front",
            "expect": "ack",
        },
        {
            "id": "visual:front:evt_000456",
            "urgency": "normal",
            "timeout_secs": 8,
            "request": "event=person_waving camera=front",
            "expect": "reply",
        },
    ],
)
def test_botified_stdout_contract_violations_fail_as_parse_errors(
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    inner = json.dumps(payload, separators=(",", ":"))
    runner = successful_runner(cli_stdout=f"<botified>{inner}</botified>\n")

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert "botified_stdout_parse_errors" in report["failure_reasons"]
    botified = report["botified_stdout"]
    assert botified["parse_error_count"] == 1
    assert botified["frame_count"] == 0
    assert botified["allowed_frame_count"] == 0
    assert botified["forbidden_event_count"] == 0
    assert botified["contract_violations"]


def test_botified_stdout_forbidden_event_fails_partial_slice(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = successful_runner(cli_stdout=botified_frame("attention_target_changed"))

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert "botified_stdout_forbidden_event" in report["failure_reasons"]
    botified = report["botified_stdout"]
    assert botified["frame_count"] == 1
    assert botified["allowed_frame_count"] == 0
    assert botified["forbidden_event_count"] == 1
    assert botified["event_counts"] == {}


def test_cli_stderr_tail_does_not_pollute_botified_stdout(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    cli_stderr = '{"type":"visual_state","status":"debug"}\n'
    runner = successful_runner(cli_stderr=cli_stderr)

    rc = module.main(base_argv(paths), runner=runner)

    assert rc == 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is True
    assert report["botified_stdout"]["pollution_count"] == 0
    assert report["botified_stdout"]["contract_violations"] == []
    assert report["stdout_stderr_tails"]["cli"]["stderr_tail"] == cli_stderr


@pytest.mark.parametrize(
    ("process_result", "expected_reason", "expected_field"),
    [
        (
            FakeResult(0, stdout=botified_frame(), stdout_truncated=True),
            "botified_stdout_collection_truncated",
            "collection_truncated",
        ),
        (
            FakeResult(0, stdout=botified_frame(), collection_incomplete=True),
            "botified_stdout_collection_incomplete",
            "collection_incomplete",
        ),
    ],
)
def test_botified_stdout_untrustworthy_collection_fails_partial_slice(
    tmp_path: Path,
    process_result: FakeResult,
    expected_reason: str,
    expected_field: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        },
        process_results={"cli": process_result},
    )

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert expected_reason in report["failure_reasons"]
    assert report["botified_stdout"][expected_field] is True


def test_botified_stdout_summary_flags_untrustworthy_collection_as_failure_reasons() -> None:
    module = import_runner_module()

    summary = module._summarize_botified_stdout(
        [botified_frame().strip()],
        line_count=3,
        collection_truncated=True,
        collection_incomplete=True,
    )

    assert summary["collection_truncated"] is True
    assert summary["collection_incomplete"] is True
    assert module._botified_stdout_failure_reasons(summary) == [
        "botified_stdout_collection_truncated",
        "botified_stdout_collection_incomplete",
    ]


def test_local_process_runner_stop_process_drains_live_stdout_and_stderr(tmp_path: Path) -> None:
    module = import_runner_module()
    script = tmp_path / "writer.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                "import time",
                "print('stdout-one', flush=True)",
                "print('stderr-one', file=sys.stderr, flush=True)",
                "print('stdout-two', flush=True)",
                "print('stderr-two', file=sys.stderr, flush=True)",
                "time.sleep(30)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = module.LocalProcessRunner()
    process = runner.start_process(
        [sys.executable, os.fspath(script)],
        cwd=tmp_path,
        env=os.environ.copy(),
        name="writer",
    )

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if process.stdout_line_count >= 2 and process.stderr_line_count >= 2:
            break
        time.sleep(0.01)

    runner.stop_process(process)

    assert process.poll() is not None
    assert process.stdout_lines == ["stdout-one", "stdout-two"]
    assert process.stderr_lines == ["stderr-one", "stderr-two"]
    assert process.stdout_tail == "stdout-one\nstdout-two\n"
    assert process.stderr_tail == "stderr-one\nstderr-two\n"
    assert process.stdout_truncated is False
    assert process.stderr_truncated is False
    assert process.collection_incomplete is False


@pytest.mark.parametrize(
    ("health", "sync_results", "expected_reason"),
    [
        (
            FakeHealth(False, failure_reason="healthz_timeout"),
            {},
            "healthz_timeout",
        ),
        (
            None,
            {"image_publisher": FakeResult(9, stderr="image failed")},
            "image_publisher_failed:segment=stationary",
        ),
        (
            None,
            {"gaze_subscriber": FakeResult(0, stdout="{not-json}\n")},
            "gaze_target_count_shortfall:segment=stationary",
        ),
    ],
)
def test_fake_runner_failures_write_nonzero_report_and_cleanup(
    tmp_path: Path,
    health: FakeHealth | None,
    sync_results: dict[str, FakeResult],
    expected_reason: str,
) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(health=health, sync_results=sync_results)

    rc = module.main(base_argv(paths), runner=runner)

    assert rc != 0
    report = load_report(paths["out"])
    assert report["slice_pass"] is False
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert report["pc_local_e2e_status"] == "partial_smoke_failed"
    assert_explicit_partial_smoke_fields(report, paths)
    assert expected_reason in report["failure_reasons"]
    if "cli" in runner.started:
        assert runner.started["cli"].stopped is True
    if "server" in runner.started:
        assert runner.started["server"].stopped is True
    if "gaze_subscriber" in runner.started and runner.started["gaze_subscriber"].poll() is None:
        assert runner.started["gaze_subscriber"].stopped is True


def test_report_never_claims_full_ga_pass(tmp_path: Path) -> None:
    module = import_runner_module()
    paths = make_case(tmp_path)
    runner = FakeRunner(
        sync_results={
            "gaze_subscriber": FakeResult(
                0,
                stdout='{"type":"gaze_target","camera":"front","state":"tracking","valid":true}\n',
            )
        }
    )

    rc = module.main(base_argv(paths), runner=runner)

    assert rc == 0
    report = load_report(paths["out"])
    serialized = json.dumps(report, sort_keys=True).lower()
    assert report["overall_pass"] is False
    assert report["ga_gate_pass"] is False
    assert '"passed": true' not in serialized
    assert '"full_pass": true' not in serialized
    assert '"ga_gate_pass": true' not in serialized
    assert '"overall_pass": true' not in serialized
