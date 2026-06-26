from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceConfig:
    backend: str = "mock"
    model_path: Path = Path("runtime/models/yolov8n-pose.pt")
    device: str | None = None
    imgsz: int = 640
    conf: float = 0.25


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    runtime_dir: Path = Path("runtime")
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def load_config(path: str | Path | None = None) -> ServerConfig:
    if path is None:
        return ServerConfig()

    config_path = Path(path)
    data = _load_mapping(config_path)
    server_data = data.get("server", data)
    if not isinstance(server_data, dict):
        raise ValueError("config root or [server] section must be an object")

    runtime_dir = Path(server_data.get("runtime_dir", ServerConfig.runtime_dir))
    inference_data = data.get("inference", {})
    if not isinstance(inference_data, dict):
        raise ValueError("[inference] section must be an object")

    return ServerConfig(
        host=str(server_data.get("host", ServerConfig.host)),
        port=int(server_data.get("port", ServerConfig.port)),
        runtime_dir=runtime_dir,
        inference=_parse_inference_config(inference_data, runtime_dir=runtime_dir),
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        if path.suffix == ".json":
            data = json.load(file)
        else:
            data = tomllib.load(file)
    if not isinstance(data, dict):
        raise ValueError("config file must contain an object")
    return data


def _parse_inference_config(
    data: dict[str, Any],
    *,
    runtime_dir: Path,
) -> InferenceConfig:
    backend = str(data.get("backend", InferenceConfig.backend))
    if backend not in {"mock", "ultralytics"}:
        raise ValueError("[inference].backend must be 'mock' or 'ultralytics'")

    model_path = Path(
        data.get("model_path", runtime_dir / "models" / "yolov8n-pose.pt")
    )
    device_value = data.get("device", InferenceConfig.device)
    device = None if device_value is None else str(device_value)
    imgsz = int(data.get("imgsz", InferenceConfig.imgsz))
    conf = float(data.get("conf", InferenceConfig.conf))
    if imgsz <= 0:
        raise ValueError("[inference].imgsz must be positive")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("[inference].conf must be between 0 and 1")

    return InferenceConfig(
        backend=backend,
        model_path=model_path,
        device=device,
        imgsz=imgsz,
        conf=conf,
    )
