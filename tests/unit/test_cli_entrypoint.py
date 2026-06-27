from __future__ import annotations

import importlib

import pytest


def import_main():
    try:
        return importlib.import_module("visual_events_cli.main").main
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.main module: {exc}")


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
