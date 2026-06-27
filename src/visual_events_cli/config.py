from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


ALLOWED_BOTIFIED_EVENTS = (
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
)


class ConfigError(Exception):
    """Raised when CLI configuration is missing, malformed, or invalid."""


@dataclass(frozen=True)
class DdsConfig:
    domain: int = 0
    network: str = "eth0"


@dataclass(frozen=True)
class CameraConfig:
    name: str = "front"
    image_topic: str = "/camera/image/jpeg"
    hz: int = 10


@dataclass(frozen=True)
class HeadStateConfig:
    enabled: bool = True
    required: bool = True
    topic: str = "/robot/head_state"
    stale_ms: int = 250


@dataclass(frozen=True)
class ServiceConfig:
    url: str = "ws://127.0.0.1:8765/v1/stream"
    response_timeout_ms: int = 1000
    reconnect_min_ms: int = 200
    reconnect_max_ms: int = 3000


@dataclass(frozen=True)
class GazeTargetConfig:
    topic: str = "/visual_events/gaze_target"
    stale_ms: int = 250


@dataclass(frozen=True)
class BotifiedConfig:
    allowed_events: tuple[str, ...] = ALLOWED_BOTIFIED_EVENTS


@dataclass(frozen=True)
class LoggingConfig:
    stderr_level: str = "info"
    jsonl_path: Path | None = None


@dataclass(frozen=True)
class CliConfig:
    dds: DdsConfig = field(default_factory=DdsConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    head_state: HeadStateConfig = field(default_factory=HeadStateConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    gaze_target: GazeTargetConfig = field(default_factory=GazeTargetConfig)
    botified: BotifiedConfig = field(default_factory=BotifiedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def default_config() -> CliConfig:
    return CliConfig()


def load_config(path: str | Path | None) -> CliConfig:
    if path is None:
        return default_config()

    config_path = Path(path)
    data = _load_mapping(config_path)
    return _config_from_mapping(data)


def apply_overrides(config: CliConfig, overrides: dict[str, Any]) -> CliConfig:
    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in _OVERRIDE_KEYS:
            raise ConfigError(f"unknown override: {key}")
        updates[key] = value

    result = config
    if "server" in updates:
        result = replace(
            result,
            service=replace(result.service, url=_as_str(updates["server"], "server")),
        )
    if "camera" in updates:
        result = replace(
            result,
            camera=replace(result.camera, name=_as_str(updates["camera"], "camera")),
        )
    if "dds_domain" in updates:
        result = replace(
            result,
            dds=replace(result.dds, domain=_as_int(updates["dds_domain"], "dds_domain")),
        )
    if "dds_network" in updates:
        result = replace(
            result,
            dds=replace(
                result.dds,
                network=_as_str(updates["dds_network"], "dds_network"),
            ),
        )
    if "image_topic" in updates:
        result = replace(
            result,
            camera=replace(
                result.camera,
                image_topic=_as_str(updates["image_topic"], "image_topic"),
            ),
        )
    if "head_state_topic" in updates:
        result = replace(
            result,
            head_state=replace(
                result.head_state,
                topic=_as_str(updates["head_state_topic"], "head_state_topic"),
            ),
        )
    if "gaze_topic" in updates:
        result = replace(
            result,
            gaze_target=replace(
                result.gaze_target,
                topic=_as_str(updates["gaze_topic"], "gaze_topic"),
            ),
        )
    if "log_path" in updates:
        result = replace(
            result,
            logging=replace(
                result.logging,
                jsonl_path=_as_optional_path(updates["log_path"], "log_path"),
            ),
        )

    _validate_config(result)
    return result


_SECTIONS = {
    "dds",
    "camera",
    "head_state",
    "service",
    "gaze_target",
    "botified",
    "logging",
}
_OVERRIDE_KEYS = {
    "server",
    "camera",
    "dds_domain",
    "dds_network",
    "image_topic",
    "head_state_topic",
    "gaze_topic",
    "log_path",
}
_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            if path.suffix.lower() == ".json":
                data = json.load(file)
            else:
                data = tomllib.load(file)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read config file {path}: {exc}") from exc
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"invalid config file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("config file must contain an object")
    return data


def _config_from_mapping(data: dict[str, Any]) -> CliConfig:
    unknown_sections = sorted(set(data) - _SECTIONS)
    if unknown_sections:
        joined = ", ".join(unknown_sections)
        raise ConfigError(f"unknown config section: {joined}")

    defaults = default_config()
    config = CliConfig(
        dds=_parse_dds(_section(data, "dds"), defaults.dds),
        camera=_parse_camera(_section(data, "camera"), defaults.camera),
        head_state=_parse_head_state(
            _section(data, "head_state"),
            defaults.head_state,
        ),
        service=_parse_service(_section(data, "service"), defaults.service),
        gaze_target=_parse_gaze_target(
            _section(data, "gaze_target"),
            defaults.gaze_target,
        ),
        botified=_parse_botified(_section(data, "botified"), defaults.botified),
        logging=_parse_logging(_section(data, "logging"), defaults.logging),
    )
    _validate_config(config)
    return config


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] section must be an object")
    return value


def _parse_dds(data: dict[str, Any], defaults: DdsConfig) -> DdsConfig:
    _reject_unknown_keys(data, "dds", {"domain", "network"})
    return DdsConfig(
        domain=_as_int(data.get("domain", defaults.domain), "dds.domain"),
        network=_as_str(data.get("network", defaults.network), "dds.network"),
    )


def _parse_camera(data: dict[str, Any], defaults: CameraConfig) -> CameraConfig:
    _reject_unknown_keys(data, "camera", {"name", "image_topic", "hz"})
    return CameraConfig(
        name=_as_str(data.get("name", defaults.name), "camera.name"),
        image_topic=_as_str(
            data.get("image_topic", defaults.image_topic),
            "camera.image_topic",
        ),
        hz=_as_int(data.get("hz", defaults.hz), "camera.hz"),
    )


def _parse_head_state(
    data: dict[str, Any],
    defaults: HeadStateConfig,
) -> HeadStateConfig:
    _reject_unknown_keys(data, "head_state", {"enabled", "required", "topic", "stale_ms"})
    return HeadStateConfig(
        enabled=_as_bool(data.get("enabled", defaults.enabled), "head_state.enabled"),
        required=_as_bool(data.get("required", defaults.required), "head_state.required"),
        topic=_as_str(data.get("topic", defaults.topic), "head_state.topic"),
        stale_ms=_as_int(data.get("stale_ms", defaults.stale_ms), "head_state.stale_ms"),
    )


def _parse_service(data: dict[str, Any], defaults: ServiceConfig) -> ServiceConfig:
    _reject_unknown_keys(
        data,
        "service",
        {
            "url",
            "response_timeout_ms",
            "reconnect_min_ms",
            "reconnect_max_ms",
        },
    )
    return ServiceConfig(
        url=_as_str(data.get("url", defaults.url), "service.url"),
        response_timeout_ms=_as_int(
            data.get("response_timeout_ms", defaults.response_timeout_ms),
            "service.response_timeout_ms",
        ),
        reconnect_min_ms=_as_int(
            data.get("reconnect_min_ms", defaults.reconnect_min_ms),
            "service.reconnect_min_ms",
        ),
        reconnect_max_ms=_as_int(
            data.get("reconnect_max_ms", defaults.reconnect_max_ms),
            "service.reconnect_max_ms",
        ),
    )


def _parse_gaze_target(
    data: dict[str, Any],
    defaults: GazeTargetConfig,
) -> GazeTargetConfig:
    _reject_unknown_keys(data, "gaze_target", {"topic", "stale_ms"})
    return GazeTargetConfig(
        topic=_as_str(data.get("topic", defaults.topic), "gaze_target.topic"),
        stale_ms=_as_int(data.get("stale_ms", defaults.stale_ms), "gaze_target.stale_ms"),
    )


def _parse_botified(data: dict[str, Any], defaults: BotifiedConfig) -> BotifiedConfig:
    _reject_unknown_keys(data, "botified", {"allowed_events"})
    allowed_events = data.get("allowed_events", defaults.allowed_events)
    if isinstance(allowed_events, str) or not isinstance(allowed_events, (list, tuple)):
        raise ConfigError("botified.allowed_events must be a list")
    return BotifiedConfig(
        allowed_events=tuple(
            _as_str(event, "botified.allowed_events") for event in allowed_events
        )
    )


def _parse_logging(data: dict[str, Any], defaults: LoggingConfig) -> LoggingConfig:
    _reject_unknown_keys(data, "logging", {"stderr_level", "jsonl_path"})
    return LoggingConfig(
        stderr_level=_as_str(
            data.get("stderr_level", defaults.stderr_level),
            "logging.stderr_level",
        ),
        jsonl_path=_as_optional_path(
            data.get("jsonl_path", defaults.jsonl_path),
            "logging.jsonl_path",
        ),
    )


def _reject_unknown_keys(
    data: dict[str, Any],
    section: str,
    allowed_keys: set[str],
) -> None:
    unknown_keys = sorted(set(data) - allowed_keys)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ConfigError(f"unknown [{section}] key: {joined}")


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{name} must be a boolean")


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _as_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string")
    if value == "":
        raise ConfigError(f"{name} must be non-empty")
    return value


def _as_optional_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a path string")
    if value == "":
        raise ConfigError(f"{name} must be non-empty")
    return Path(value)


def _validate_config(config: CliConfig) -> None:
    _validate_nonnegative(config.dds.domain, "dds.domain")
    _validate_positive(config.camera.hz, "camera.hz")
    _validate_topic(config.camera.image_topic, "camera.image_topic")
    _validate_topic(config.head_state.topic, "head_state.topic")
    _validate_positive(config.head_state.stale_ms, "head_state.stale_ms")
    _validate_positive(
        config.service.response_timeout_ms,
        "service.response_timeout_ms",
    )
    _validate_positive(config.service.reconnect_min_ms, "service.reconnect_min_ms")
    _validate_positive(config.service.reconnect_max_ms, "service.reconnect_max_ms")
    if config.service.reconnect_min_ms > config.service.reconnect_max_ms:
        raise ConfigError(
            "service.reconnect_min_ms must be less than or equal to "
            "service.reconnect_max_ms"
        )
    _validate_topic(config.gaze_target.topic, "gaze_target.topic")
    _validate_positive(config.gaze_target.stale_ms, "gaze_target.stale_ms")
    _validate_allowed_events(config.botified.allowed_events)

    level = config.logging.stderr_level.lower()
    if level not in _LOG_LEVELS:
        raise ConfigError("logging.stderr_level must be a standard log level")
    if level != config.logging.stderr_level:
        raise ConfigError("logging.stderr_level must be lowercase")


def _validate_topic(value: str, name: str) -> None:
    if not value.startswith("/"):
        raise ConfigError(f"{name} must start with '/'")


def _validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


def _validate_nonnegative(value: int, name: str) -> None:
    if value < 0:
        raise ConfigError(f"{name} must be non-negative")


def _validate_allowed_events(events: tuple[str, ...]) -> None:
    allowed = set(ALLOWED_BOTIFIED_EVENTS)
    unknown = sorted(set(events) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigError(f"botified.allowed_events contains unsupported events: {joined}")
