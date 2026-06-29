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
class MetricsConfig:
    jsonl_path: Path | None = None


@dataclass(frozen=True)
class MemoryEmbeddingConfig:
    backend: str = "disabled"
    person_model_path: Path | None = None
    scene_model_path: Path | None = None
    teach_queue_size: int = 2
    teach_queue_timeout_ms: int = 500


@dataclass(frozen=True)
class MemoryMatchingConfig:
    known_person_threshold: float = 0.82
    known_person_margin: float = 0.06
    anonymous_threshold: float = 0.78
    anonymous_margin: float = 0.04
    familiar_seen_count: int = 3
    familiar_threshold: float = 0.78
    scene_threshold: float = 0.78
    event_cooldown_ms: int = 60000


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = False
    db_path: Path = Path("runtime/memory/visual_memory.sqlite3")
    frame_cache_seconds: int = 5
    query_interval_ms: int = 1000
    queue_size: int = 2
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)
    matching: MemoryMatchingConfig = field(default_factory=MemoryMatchingConfig)


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    runtime_dir: Path = Path("runtime")
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    events: EventConfig = field(default_factory=EventConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


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
    metrics_data = data.get("metrics", {})
    if not isinstance(metrics_data, dict):
        raise ValueError("[metrics] section must be an object")
    memory_data = data.get("memory", {})
    if not isinstance(memory_data, dict):
        raise ValueError("[memory] section must be an object")

    return ServerConfig(
        host=str(server_data.get("host", ServerConfig.host)),
        port=int(server_data.get("port", ServerConfig.port)),
        runtime_dir=runtime_dir,
        inference=_parse_inference_config(inference_data, runtime_dir=runtime_dir),
        tracking=_parse_tracking_config(tracking_data),
        attention=_parse_attention_config(attention_data),
        events=_parse_event_config(events_data),
        metrics=_parse_metrics_config(metrics_data),
        memory=_parse_memory_config(memory_data, runtime_dir=runtime_dir),
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


def _parse_metrics_config(data: dict[str, Any]) -> MetricsConfig:
    jsonl_path = data.get("jsonl_path")
    if jsonl_path is None:
        return MetricsConfig()
    jsonl_path_text = str(jsonl_path)
    if not jsonl_path_text:
        raise ValueError("[metrics].jsonl_path must be non-empty")
    return MetricsConfig(jsonl_path=Path(jsonl_path_text))


def _parse_memory_config(
    data: dict[str, Any],
    *,
    runtime_dir: Path,
) -> MemoryConfig:
    defaults = MemoryConfig()
    embedding_data = data.get("embedding", {})
    if not isinstance(embedding_data, dict):
        raise ValueError("[memory.embedding] section must be an object")
    matching_data = data.get("matching", {})
    if not isinstance(matching_data, dict):
        raise ValueError("[memory.matching] section must be an object")

    enabled = _parse_bool(data.get("enabled", defaults.enabled), "[memory].enabled")
    db_path = Path(
        data.get("db_path", runtime_dir / "memory" / "visual_memory.sqlite3")
    )
    frame_cache_seconds = int(
        data.get("frame_cache_seconds", defaults.frame_cache_seconds)
    )
    query_interval_ms = int(data.get("query_interval_ms", defaults.query_interval_ms))
    queue_size = int(data.get("queue_size", defaults.queue_size))
    if frame_cache_seconds <= 0:
        raise ValueError("[memory].frame_cache_seconds must be positive")
    if query_interval_ms <= 0:
        raise ValueError("[memory].query_interval_ms must be positive")
    if queue_size <= 0:
        raise ValueError("[memory].queue_size must be positive")

    return MemoryConfig(
        enabled=enabled,
        db_path=db_path,
        frame_cache_seconds=frame_cache_seconds,
        query_interval_ms=query_interval_ms,
        queue_size=queue_size,
        embedding=_parse_memory_embedding_config(embedding_data),
        matching=_parse_memory_matching_config(matching_data),
    )


def _parse_memory_embedding_config(data: dict[str, Any]) -> MemoryEmbeddingConfig:
    defaults = MemoryEmbeddingConfig()
    backend = str(data.get("backend", defaults.backend))
    if backend not in {"disabled", "fake", "local"}:
        raise ValueError(
            "[memory.embedding].backend must be 'disabled', 'fake', or 'local'"
        )
    teach_queue_size = int(data.get("teach_queue_size", defaults.teach_queue_size))
    teach_queue_timeout_ms = int(
        data.get("teach_queue_timeout_ms", defaults.teach_queue_timeout_ms)
    )
    if teach_queue_size < 0:
        raise ValueError("[memory.embedding].teach_queue_size must be non-negative")
    if teach_queue_timeout_ms <= 0:
        raise ValueError("[memory.embedding].teach_queue_timeout_ms must be positive")
    return MemoryEmbeddingConfig(
        backend=backend,
        person_model_path=_optional_path(data.get("person_model_path")),
        scene_model_path=_optional_path(data.get("scene_model_path")),
        teach_queue_size=teach_queue_size,
        teach_queue_timeout_ms=teach_queue_timeout_ms,
    )


def _parse_memory_matching_config(data: dict[str, Any]) -> MemoryMatchingConfig:
    defaults = MemoryMatchingConfig()
    known_person_threshold = float(
        data.get("known_person_threshold", defaults.known_person_threshold)
    )
    known_person_margin = float(
        data.get("known_person_margin", defaults.known_person_margin)
    )
    anonymous_threshold = float(
        data.get("anonymous_threshold", defaults.anonymous_threshold)
    )
    anonymous_margin = float(data.get("anonymous_margin", defaults.anonymous_margin))
    familiar_seen_count = int(
        data.get("familiar_seen_count", defaults.familiar_seen_count)
    )
    familiar_threshold = float(
        data.get("familiar_threshold", defaults.familiar_threshold)
    )
    scene_threshold = float(data.get("scene_threshold", defaults.scene_threshold))
    event_cooldown_ms = int(
        data.get("event_cooldown_ms", defaults.event_cooldown_ms)
    )
    for name, value in (
        ("known_person_threshold", known_person_threshold),
        ("known_person_margin", known_person_margin),
        ("anonymous_threshold", anonymous_threshold),
        ("anonymous_margin", anonymous_margin),
        ("familiar_threshold", familiar_threshold),
        ("scene_threshold", scene_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"[memory.matching].{name} must be between 0 and 1")
    if familiar_seen_count <= 0:
        raise ValueError("[memory.matching].familiar_seen_count must be positive")
    if event_cooldown_ms <= 0:
        raise ValueError("[memory.matching].event_cooldown_ms must be positive")
    return MemoryMatchingConfig(
        known_person_threshold=known_person_threshold,
        known_person_margin=known_person_margin,
        anonymous_threshold=anonymous_threshold,
        anonymous_margin=anonymous_margin,
        familiar_seen_count=familiar_seen_count,
        familiar_threshold=familiar_threshold,
        scene_threshold=scene_threshold,
        event_cooldown_ms=event_cooldown_ms,
    )


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return Path(text)


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
        reacquire_alias_window_ms=int(
            data.get(
                "reacquire_alias_window_ms",
                defaults.reacquire_alias_window_ms,
            )
        ),
        reacquire_center_distance_ratio=float(
            data.get(
                "reacquire_center_distance_ratio",
                defaults.reacquire_center_distance_ratio,
            )
        ),
    )
