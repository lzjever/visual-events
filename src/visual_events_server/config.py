from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visual_events_server.attention import AttentionConfig
from visual_events_server.events import EventConfig
from visual_events_server.tracking import TrackingConfig


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
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    events: EventConfig = field(default_factory=EventConfig)


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
    tracking_data = data.get("tracking", {})
    if not isinstance(tracking_data, dict):
        raise ValueError("[tracking] section must be an object")
    attention_data = data.get("attention", {})
    if not isinstance(attention_data, dict):
        raise ValueError("[attention] section must be an object")
    events_data = data.get("events", {})
    if not isinstance(events_data, dict):
        raise ValueError("[events] section must be an object")

    return ServerConfig(
        host=str(server_data.get("host", ServerConfig.host)),
        port=int(server_data.get("port", ServerConfig.port)),
        runtime_dir=runtime_dir,
        inference=_parse_inference_config(inference_data, runtime_dir=runtime_dir),
        tracking=_parse_tracking_config(tracking_data),
        attention=_parse_attention_config(attention_data),
        events=_parse_event_config(events_data),
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


def _parse_tracking_config(data: dict[str, Any]) -> TrackingConfig:
    defaults = TrackingConfig()
    return TrackingConfig(
        high_conf=float(data.get("high_conf", defaults.high_conf)),
        low_conf=float(data.get("low_conf", defaults.low_conf)),
        new_track_conf=float(data.get("new_track_conf", defaults.new_track_conf)),
        match_iou=float(data.get("match_iou", defaults.match_iou)),
        lost_ttl_ms=int(data.get("lost_ttl_ms", defaults.lost_ttl_ms)),
        history_ms=int(data.get("history_ms", defaults.history_ms)),
        velocity_window_ms=int(
            data.get("velocity_window_ms", defaults.velocity_window_ms)
        ),
    )


def _parse_attention_config(data: dict[str, Any]) -> AttentionConfig:
    defaults = AttentionConfig()
    return AttentionConfig(
        stable_min_hits=int(data.get("stable_min_hits", defaults.stable_min_hits)),
        stable_min_age_ms=int(
            data.get("stable_min_age_ms", defaults.stable_min_age_ms)
        ),
        switch_area_ratio=float(
            data.get("switch_area_ratio", defaults.switch_area_ratio)
        ),
        switch_confirm_ms=int(
            data.get("switch_confirm_ms", defaults.switch_confirm_ms)
        ),
        lost_hold_ms=int(data.get("lost_hold_ms", defaults.lost_hold_ms)),
    )


def _parse_event_config(data: dict[str, Any]) -> EventConfig:
    defaults = EventConfig()
    return EventConfig(
        history_ms=int(data.get("history_ms", defaults.history_ms)),
        cooldown_ms=int(data.get("cooldown_ms", defaults.cooldown_ms)),
        stable_min_hits=int(data.get("stable_min_hits", defaults.stable_min_hits)),
        stable_min_age_ms=int(
            data.get("stable_min_age_ms", defaults.stable_min_age_ms)
        ),
        left_lost_ms=int(data.get("left_lost_ms", defaults.left_lost_ms)),
        stop_duration_ms=int(
            data.get("stop_duration_ms", defaults.stop_duration_ms)
        ),
        near_area_ratio=float(
            data.get("near_area_ratio", defaults.near_area_ratio)
        ),
        near_height_ratio=float(
            data.get("near_height_ratio", defaults.near_height_ratio)
        ),
        stop_max_speed_px_s=float(
            data.get("stop_max_speed_px_s", defaults.stop_max_speed_px_s)
        ),
        approach_duration_ms=int(
            data.get("approach_duration_ms", defaults.approach_duration_ms)
        ),
        approach_area_growth_ratio=float(
            data.get(
                "approach_area_growth_ratio",
                defaults.approach_area_growth_ratio,
            )
        ),
        approach_min_area_delta=float(
            data.get("approach_min_area_delta", defaults.approach_min_area_delta)
        ),
        approach_min_current_area=float(
            data.get(
                "approach_min_current_area",
                defaults.approach_min_current_area,
            )
        ),
        passing_duration_ms=int(
            data.get("passing_duration_ms", defaults.passing_duration_ms)
        ),
        passing_min_dx_ratio=float(
            data.get("passing_min_dx_ratio", defaults.passing_min_dx_ratio)
        ),
        passing_min_abs_vx_px_s=float(
            data.get(
                "passing_min_abs_vx_px_s",
                defaults.passing_min_abs_vx_px_s,
            )
        ),
        keypoint_min_conf=float(
            data.get("keypoint_min_conf", defaults.keypoint_min_conf)
        ),
        wave_window_ms=int(data.get("wave_window_ms", defaults.wave_window_ms)),
        wave_min_x_span_px=float(
            data.get("wave_min_x_span_px", defaults.wave_min_x_span_px)
        ),
        wave_min_x_span_bbox_ratio=float(
            data.get(
                "wave_min_x_span_bbox_ratio",
                defaults.wave_min_x_span_bbox_ratio,
            )
        ),
    )
