from __future__ import annotations

import importlib

import pytest


def import_main():
    try:
        return importlib.import_module("visual_events_cli.main").main
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.main module: {exc}")


def test_check_config_returns_zero_and_writes_no_stdout(capsys):
    main = import_main()

    result = main(["--check-config"])

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


def test_runtime_without_check_config_is_explicit_until_loop_exists(capsys):
    main = import_main()

    result = main([])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "--check-config" in captured.err or "runtime" in captured.err.lower()
