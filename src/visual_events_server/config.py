from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    runtime_dir: Path = Path("runtime")


def load_config(path: str | Path | None = None) -> ServerConfig:
    if path is None:
        return ServerConfig()

    config_path = Path(path)
    data = _load_mapping(config_path)
    server_data = data.get("server", data)
    if not isinstance(server_data, dict):
        raise ValueError("config root or [server] section must be an object")

    return ServerConfig(
        host=str(server_data.get("host", ServerConfig.host)),
        port=int(server_data.get("port", ServerConfig.port)),
        runtime_dir=Path(server_data.get("runtime_dir", ServerConfig.runtime_dir)),
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
