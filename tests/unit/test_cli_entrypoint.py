from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def import_main():
    try:
        return importlib.import_module("visual_events_cli.main").main
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.main module: {exc}")


def import_main_module():
    try:
        return importlib.import_module("visual_events_cli.main")
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.main module: {exc}")


def import_fresh_main_module_blocking_bridge_factories(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "visual_events_cli.runtime_factories":
            raise AssertionError("runtime_factories must be imported only for bridge")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "visual_events_cli.main", raising=False)
    monkeypatch.delitem(sys.modules, "visual_events_cli.runtime_factories", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return import_main_module()


def test_check_config_returns_zero_writes_no_output_and_skips_runtime(capsys):
    main = import_main()

    def fake_runtime_runner(_config):
        raise AssertionError("--check-config must not start runtime")

    result = main(["--check-config"], runtime_runner=fake_runtime_runner)

    captured = capsys.readouterr()
    assert isinstance(result, int)
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


def test_check_config_accepts_server_and_camera_overrides(capsys):
    main = import_main()

    result = main(
        [
            "--check-config",
            "--server",
            "ws://10.0.0.1:8765/v1/stream",
            "--camera",
            "front",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


def test_invalid_config_path_writes_stderr_and_returns_two(tmp_path, capsys):
    main = import_main()
    missing_path = tmp_path / "missing.toml"

    result = main(["--check-config", "--config", str(missing_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "config" in captured.err.lower()


def test_invalid_config_writes_stderr_and_returns_two(tmp_path, capsys):
    main = import_main()
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[service]
response_timeout_ms = 0
""".strip(),
        encoding="utf-8",
    )

    result = main(["--check-config", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "response_timeout_ms" in captured.err


def test_runtime_without_check_config_calls_runtime_runner_and_returns_exit_code(capsys):
    main = import_main()
    received_configs = []

    def fake_runtime_runner(config):
        received_configs.append(config)
        return 37

    result = main([], runtime_runner=fake_runtime_runner)

    captured = capsys.readouterr()
    assert result == 37
    assert len(received_configs) == 1
    assert captured.out == ""
    assert captured.err == ""


def test_default_runtime_runner_writes_dds_not_implemented_and_returns_two(capsys):
    main = import_main()

    result = main([])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "Step 4 DDS adapters not implemented" in captured.err


def test_default_runtime_runner_does_not_use_bridge_when_env_exists(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    module = import_fresh_main_module_blocking_bridge_factories(monkeypatch)
    monkeypatch.setenv("VISUAL_EVENTS_DDS_BRIDGE_BIN", "/tmp/visual-events-dds-bridge")

    result = module.main([])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "Step 4 DDS adapters not implemented" in captured.err


def test_check_config_does_not_load_bridge_factories(monkeypatch: pytest.MonkeyPatch, capsys):
    module = import_fresh_main_module_blocking_bridge_factories(monkeypatch)
    monkeypatch.delenv("VISUAL_EVENTS_DDS_BRIDGE_BIN", raising=False)

    result = module.main(["--check-config", "--dds-runtime", "bridge"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


def test_explicit_bridge_runtime_without_bridge_bin_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    main = import_main()
    monkeypatch.delenv("VISUAL_EVENTS_DDS_BRIDGE_BIN", raising=False)

    result = main(["--dds-runtime", "bridge"])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "VISUAL_EVENTS_DDS_BRIDGE_BIN" in captured.err


def test_check_config_does_not_require_explicit_bridge_bin(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    main = import_main()
    monkeypatch.delenv("VISUAL_EVENTS_DDS_BRIDGE_BIN", raising=False)

    result = main(["--check-config", "--dds-runtime", "bridge"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


def test_default_runtime_runner_uses_bridge_factories_for_explicit_bridge(
    monkeypatch: pytest.MonkeyPatch,
):
    module = import_main_module()
    sentinels = []

    def fake_bridge_runtime_factories(*, bridge_bin):
        sentinels.append(("factory", bridge_bin))
        return "bridge-factories"

    def fake_run_runtime(config, *, factories=None):
        sentinels.append(("run", config.dds.runtime, config.dds.bridge_bin, factories))
        return 0

    monkeypatch.setitem(
        sys.modules,
        "visual_events_cli.runtime_factories",
        SimpleNamespace(bridge_runtime_factories=fake_bridge_runtime_factories),
    )
    monkeypatch.setattr(module, "run_runtime", fake_run_runtime)

    result = module.main(
        [
            "--dds-runtime",
            "bridge",
            "--dds-bridge-bin",
            "/tmp/visual-events-dds-bridge",
        ]
    )

    assert result == 0
    assert sentinels == [
        ("factory", "/tmp/visual-events-dds-bridge"),
        (
            "run",
            "bridge",
            Path("/tmp/visual-events-dds-bridge"),
            "bridge-factories",
        ),
    ]


def test_default_runtime_runner_uses_bridge_factories_from_config_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = import_main_module()
    config_path = tmp_path / "cli.toml"
    bridge_bin = tmp_path / "visual-events-dds-bridge"
    config_path.write_text(
        f"""
[dds]
runtime = "bridge"
bridge_bin = "{bridge_bin}"
""".strip(),
        encoding="utf-8",
    )
    sentinels = []

    def fake_bridge_runtime_factories(*, bridge_bin):
        sentinels.append(("factory", bridge_bin))
        return "bridge-factories"

    def fake_run_runtime(config, *, factories=None):
        sentinels.append(("run", config.dds.runtime, config.dds.bridge_bin, factories))
        return 0

    monkeypatch.setitem(
        sys.modules,
        "visual_events_cli.runtime_factories",
        SimpleNamespace(bridge_runtime_factories=fake_bridge_runtime_factories),
    )
    monkeypatch.setattr(module, "run_runtime", fake_run_runtime)

    result = module.main(["--config", str(config_path)])

    assert result == 0
    assert sentinels == [
        ("factory", str(bridge_bin)),
        ("run", "bridge", bridge_bin, "bridge-factories"),
    ]


def test_runtime_runner_receives_config_after_cli_overrides(capsys):
    main = import_main()
    received_configs = []

    def fake_runtime_runner(config):
        received_configs.append(config)
        return 0

    result = main(
        [
            "--server",
            "ws://10.0.0.1:8765/v1/stream",
            "--camera",
            "rear",
            "--dds-domain",
            "57",
            "--dds-network",
            "lo",
            "--dds-runtime",
            "bridge",
            "--dds-bridge-bin",
            "/tmp/visual-events-dds-bridge",
        ],
        runtime_runner=fake_runtime_runner,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""
    assert len(received_configs) == 1
    config = received_configs[0]
    assert config.service.url == "ws://10.0.0.1:8765/v1/stream"
    assert config.camera.name == "rear"
    assert config.dds.domain == 57
    assert config.dds.network == "lo"
    assert config.dds.runtime == "bridge"
    assert config.dds.bridge_bin == Path("/tmp/visual-events-dds-bridge")
