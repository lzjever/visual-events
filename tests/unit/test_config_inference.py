import os
import sys
import tomllib
import types
from pathlib import Path

import pytest

from visual_events_server.config import InferenceConfig, load_config
from visual_events_server.inference.factory import create_infer_backend
from visual_events_server.inference.mock import MockInferBackend
from visual_events_server.inference.ultralytics_pose import InferenceConfigError


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_config_parses_inference_section_and_default_model_path(tmp_path):
    config_path = tmp_path / "config.toml"
    runtime_dir = tmp_path / "runtime"
    config_path.write_text(
        f"""
[server]
host = "0.0.0.0"
port = 9000
runtime_dir = "{runtime_dir}"

[inference]
backend = "ultralytics"
device = "0"
imgsz = 320
conf = 0.4

[tracking]
high_conf = 0.55
low_conf = 0.12
new_track_conf = 0.65
match_iou = 0.25
lost_ttl_ms = 1800
history_ms = 3200
velocity_window_ms = 900

[attention]
stable_min_hits = 3
stable_min_age_ms = 350
switch_area_ratio = 1.4
switch_confirm_ms = 700
lost_hold_ms = 800

[events]
history_ms = 2500
cooldown_ms = 4500
stable_min_hits = 3
stable_min_age_ms = 400
left_lost_ms = 1700
stop_max_speed_px_s = 28.5
passing_min_dx_ratio = 0.5
wave_window_ms = 1600
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.host == "0.0.0.0"
    assert config.port == 9000
    assert config.runtime_dir == runtime_dir
    assert config.inference.backend == "ultralytics"
    assert config.inference.model_path == runtime_dir / "models" / "yolov8n-pose.pt"
    assert config.inference.device == "0"
    assert config.inference.imgsz == 320
    assert config.inference.conf == 0.4
    assert config.tracking.high_conf == 0.55
    assert config.tracking.low_conf == 0.12
    assert config.tracking.new_track_conf == 0.65
    assert config.tracking.match_iou == 0.25
    assert config.tracking.lost_ttl_ms == 1800
    assert config.tracking.history_ms == 3200
    assert config.tracking.velocity_window_ms == 900
    assert config.attention.stable_min_hits == 3
    assert config.attention.stable_min_age_ms == 350
    assert config.attention.switch_area_ratio == 1.4
    assert config.attention.switch_confirm_ms == 700
    assert config.attention.lost_hold_ms == 800
    assert config.events.history_ms == 2500
    assert config.events.cooldown_ms == 4500
    assert config.events.stable_min_hits == 3
    assert config.events.stable_min_age_ms == 400
    assert config.events.left_lost_ms == 1700
    assert config.events.stop_max_speed_px_s == 28.5
    assert config.events.passing_min_dx_ratio == 0.5
    assert config.events.wave_window_ms == 1600


def test_metrics_config_is_disabled_by_default():
    config = load_config()

    assert config.metrics.jsonl_path is None


def test_pc_ga_server_config_sets_750ms_attention_switch_dwell():
    config_path = REPO_ROOT / "configs" / "pc-ga-server.toml"

    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)
    config = load_config(config_path)

    assert raw_config["attention"]["switch_confirm_ms"] >= 750
    assert config.attention.switch_confirm_ms == raw_config["attention"][
        "switch_confirm_ms"
    ]


def test_load_config_parses_metrics_jsonl_path(tmp_path):
    metrics_path = tmp_path / "metrics" / "server.jsonl"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[metrics]
jsonl_path = "{metrics_path}"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.metrics.jsonl_path == metrics_path


def test_server_cli_metrics_jsonl_override_wires_config(tmp_path, monkeypatch):
    from visual_events_server import app as app_module

    captured: dict[str, object] = {}
    metrics_path = tmp_path / "cli-metrics.jsonl"

    def fake_create_processor_from_config(config):
        captured["processor_config"] = config
        return object()

    def fake_uvicorn_run(app, *, host, port):
        captured["app_config"] = app.state.config
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(
        app_module,
        "create_processor_from_config",
        fake_create_processor_from_config,
    )
    monkeypatch.setattr(app_module.uvicorn, "run", fake_uvicorn_run)

    app_module.main(["--metrics-jsonl", str(metrics_path), "--port", "9911"])

    assert captured["processor_config"].metrics.jsonl_path == metrics_path
    assert captured["app_config"].metrics.jsonl_path == metrics_path
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9911


def test_load_config_rejects_invalid_attention_section(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[attention]
stable_min_hits = 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stable_min_hits"):
        load_config(config_path)


def test_load_config_rejects_invalid_events_section_value(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[events]
cooldown_ms = -1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cooldown_ms"):
        load_config(config_path)


def test_load_config_rejects_non_object_events_section(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"events": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[events\] section"):
        load_config(config_path)


def test_factory_mock_backend_does_not_touch_runtime_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    backend = create_infer_backend(InferenceConfig(), runtime_dir=tmp_path / "runtime")

    assert isinstance(backend, MockInferBackend)
    assert "YOLO_CONFIG_DIR" not in os.environ


def test_factory_ultralytics_sets_cache_env_and_fails_clearly_for_missing_model(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "runtime"
    original_home = "/tmp/visual-events-home"
    monkeypatch.setenv("HOME", original_home)
    for key in ("YOLO_CONFIG_DIR", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(InferenceConfigError, match="model file not found"):
        create_infer_backend(
            InferenceConfig(
                backend="ultralytics",
                model_path=runtime_dir / "models" / "missing.pt",
            ),
            runtime_dir=runtime_dir,
        )

    assert Path(os.environ["YOLO_CONFIG_DIR"]) == runtime_dir / "cache" / "yolo"
    assert Path(os.environ["TORCH_HOME"]) == runtime_dir / "cache" / "torch"
    assert Path(os.environ["XDG_CACHE_HOME"]) == runtime_dir / "cache" / "xdg"
    assert Path(os.environ["MPLCONFIGDIR"]) == runtime_dir / "cache" / "matplotlib"
    assert os.environ["HOME"] == original_home


def test_ultralytics_backend_loads_explicit_model_path_with_cache_env_ready(
    tmp_path,
    monkeypatch,
):
    runtime_dir = tmp_path / "runtime"
    model_path = runtime_dir / "models" / "yolov8n-pose.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"fake model")
    for key in ("YOLO_CONFIG_DIR", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        monkeypatch.delenv(key, raising=False)

    constructor_calls: list[dict[str, object]] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            constructor_calls.append(
                {
                    "path": path,
                    "env": {
                        "YOLO_CONFIG_DIR": os.environ.get("YOLO_CONFIG_DIR"),
                        "TORCH_HOME": os.environ.get("TORCH_HOME"),
                        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
                        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
                    },
                }
            )

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    backend = create_infer_backend(
        InferenceConfig(
            backend="ultralytics",
            model_path=model_path,
        ),
        runtime_dir=runtime_dir,
    )

    assert backend._load_model() is backend._load_model()
    assert constructor_calls == [
        {
            "path": str(model_path),
            "env": {
                "YOLO_CONFIG_DIR": str(runtime_dir / "cache" / "yolo"),
                "TORCH_HOME": str(runtime_dir / "cache" / "torch"),
                "XDG_CACHE_HOME": str(runtime_dir / "cache" / "xdg"),
                "MPLCONFIGDIR": str(runtime_dir / "cache" / "matplotlib"),
            },
        }
    ]
    assert constructor_calls[0]["path"] != "yolov8n-pose.pt"
