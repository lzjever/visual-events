import os
from pathlib import Path

import pytest

from visual_events_server.config import InferenceConfig, load_config
from visual_events_server.inference.factory import create_infer_backend
from visual_events_server.inference.mock import MockInferBackend
from visual_events_server.inference.ultralytics_pose import InferenceConfigError


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
