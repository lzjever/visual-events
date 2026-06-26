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
