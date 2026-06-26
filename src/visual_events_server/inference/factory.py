from __future__ import annotations

import os
from pathlib import Path

from visual_events_server.config import InferenceConfig

from .base import InferBackend
from .mock import MockInferBackend
from .ultralytics_pose import UltralyticsPoseBackend


def create_infer_backend(
    inference_config: InferenceConfig,
    *,
    runtime_dir: Path,
) -> InferBackend:
    if inference_config.backend == "mock":
        return MockInferBackend()
    if inference_config.backend == "ultralytics":
        configure_inference_cache(runtime_dir)
        return UltralyticsPoseBackend(
            model_path=inference_config.model_path,
            device=inference_config.device,
            imgsz=inference_config.imgsz,
            conf=inference_config.conf,
        )
    raise ValueError(f"unsupported inference backend: {inference_config.backend}")


def configure_inference_cache(runtime_dir: Path) -> None:
    cache_dir = Path(runtime_dir) / "cache"
    paths = {
        "YOLO_CONFIG_DIR": cache_dir / "yolo",
        "TORCH_HOME": cache_dir / "torch",
        "XDG_CACHE_HOME": cache_dir / "xdg",
        "MPLCONFIGDIR": cache_dir / "matplotlib",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
