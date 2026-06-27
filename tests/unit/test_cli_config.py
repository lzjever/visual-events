from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


EXPECTED_BOTIFIED_EVENTS = {
    "person_appeared",
    "person_left",
    "person_passing_by",
    "person_approaching_robot",
    "person_stopped_near_robot",
    "person_waving",
}


def import_config_module():
    try:
        return importlib.import_module("visual_events_cli.config")
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.config module: {exc}")


def config_error(module) -> type[Exception]:
    error_type = getattr(module, "ConfigError", None)
    assert error_type is not None
    assert issubclass(error_type, Exception)
    return error_type


def assert_default_config(config: Any) -> None:
    assert config.dds.domain == 0
    assert config.dds.network == "eth0"
    assert config.camera.name == "front"
    assert config.camera.image_topic == "/camera/image/jpeg"
    assert config.camera.hz == 10
    assert config.head_state.enabled is True
    assert config.head_state.required is True
    assert config.head_state.topic == "/robot/head_state"
    assert config.head_state.stale_ms == 250
    assert config.head_state.stationary_yaw_vel_rad_s == 0.03
    assert config.head_state.stationary_pitch_vel_rad_s == 0.03
    assert config.head_state.report_segments == ("stationary", "moving", "unknown")
    assert config.service.url == "ws://127.0.0.1:8765/v1/stream"
    assert config.service.response_timeout_ms == 1000
    assert config.service.reconnect_min_ms == 200
    assert config.service.reconnect_max_ms == 3000
    assert config.gaze_target.enabled is True
    assert config.gaze_target.topic == "/visual_events/gaze_target"
    assert config.gaze_target.stale_ms == 250
    assert config.gaze_target.publish_invalid_on_loss is True
    assert config.botified.enabled is True
    assert config.botified.stdout is True
    assert config.botified.event_ttl_secs == 8
    assert set(config.botified.allowed_events) == EXPECTED_BOTIFIED_EVENTS
    assert len(config.botified.allowed_events) == len(EXPECTED_BOTIFIED_EVENTS)
    assert config.botified.stdout_queue_max == 32
    assert config.botified.stdout_drop_policy == "drop_oldest_duplicate_event"
    assert config.botified.broken_pipe == "publish_stale_then_exit_nonzero"
    assert config.logging.stderr_level == "info"


def test_default_config_matches_ga_cli_plan():
    module = import_config_module()

    assert_default_config(module.default_config())


def test_load_config_parses_toml_and_merges_with_defaults(tmp_path):
    module = import_config_module()
    config_path = tmp_path / "cli.toml"
    log_path = tmp_path / "logs" / "cli.jsonl"
    config_path.write_text(
        f"""
[dds]
domain = 57
network = "lo"

[camera]
name = "rear"
image_topic = "/camera/rear/jpeg"
hz = 15

[head_state]
enabled = false
required = false
topic = "/robot/custom_head_state"
stale_ms = 500
stationary_yaw_vel_rad_s = 0.04
stationary_pitch_vel_rad_s = 0.05
report_segments = ["moving", "unknown"]

[service]
url = "ws://10.0.0.1:8765/v1/stream"
response_timeout_ms = 1500
reconnect_min_ms = 100
reconnect_max_ms = 2000

[gaze_target]
enabled = false
topic = "/visual_events/custom_gaze"
stale_ms = 400
publish_invalid_on_loss = false

[botified]
enabled = false
stdout = false
event_ttl_secs = 9
allowed_events = ["person_appeared", "person_waving"]
stdout_queue_max = 7
stdout_drop_policy = "drop_oldest_duplicate_event"
broken_pipe = "publish_stale_then_exit_nonzero"

[logging]
stderr_level = "debug"
jsonl_path = "{log_path}"
""".strip(),
        encoding="utf-8",
    )

    config = module.load_config(config_path)

    assert config.dds.domain == 57
    assert config.dds.network == "lo"
    assert config.camera.name == "rear"
    assert config.camera.image_topic == "/camera/rear/jpeg"
    assert config.camera.hz == 15
    assert config.head_state.enabled is False
    assert config.head_state.required is False
    assert config.head_state.topic == "/robot/custom_head_state"
    assert config.head_state.stale_ms == 500
    assert config.head_state.stationary_yaw_vel_rad_s == 0.04
    assert config.head_state.stationary_pitch_vel_rad_s == 0.05
    assert config.head_state.report_segments == ("moving", "unknown")
    assert config.service.url == "ws://10.0.0.1:8765/v1/stream"
    assert config.service.response_timeout_ms == 1500
    assert config.service.reconnect_min_ms == 100
    assert config.service.reconnect_max_ms == 2000
    assert config.gaze_target.enabled is False
    assert config.gaze_target.topic == "/visual_events/custom_gaze"
    assert config.gaze_target.stale_ms == 400
    assert config.gaze_target.publish_invalid_on_loss is False
    assert config.botified.enabled is False
    assert config.botified.stdout is False
    assert config.botified.event_ttl_secs == 9
    assert config.botified.allowed_events == (
        "person_appeared",
        "person_waving",
    )
    assert config.botified.stdout_queue_max == 7
    assert config.botified.stdout_drop_policy == "drop_oldest_duplicate_event"
    assert config.botified.broken_pipe == "publish_stale_then_exit_nonzero"
    assert config.logging.stderr_level == "debug"
    assert config.logging.jsonl_path == log_path


def test_load_config_parses_json_and_merges_with_defaults(tmp_path):
    module = import_config_module()
    config_path = tmp_path / "cli.json"
    config_path.write_text(
        json.dumps(
            {
                "dds": {"domain": 3, "network": "wlan0"},
                "service": {
                    "url": "ws://192.168.1.20:8765/v1/stream",
                    "response_timeout_ms": 2000,
                },
                "gaze_target": {"stale_ms": 300},
            }
        ),
        encoding="utf-8",
    )

    config = module.load_config(config_path)

    assert config.dds.domain == 3
    assert config.dds.network == "wlan0"
    assert config.service.url == "ws://192.168.1.20:8765/v1/stream"
    assert config.service.response_timeout_ms == 2000
    assert config.gaze_target.stale_ms == 300
    assert config.camera.name == "front"
    assert config.head_state.required is True


def test_apply_overrides_covers_runtime_flags(tmp_path):
    module = import_config_module()
    log_path = tmp_path / "cli.log.jsonl"

    config = module.apply_overrides(
        module.default_config(),
        {
            "server": "ws://10.0.0.1:8765/v1/stream",
            "camera": "rear",
            "dds_domain": 57,
            "dds_network": "lo",
            "image_topic": "/camera/rear/jpeg",
            "head_state_topic": "/robot/head_state_test",
            "gaze_topic": "/visual_events/gaze_test",
            "log_path": str(log_path),
        },
    )

    assert config.service.url == "ws://10.0.0.1:8765/v1/stream"
    assert config.camera.name == "rear"
    assert config.dds.domain == 57
    assert config.dds.network == "lo"
    assert config.camera.image_topic == "/camera/rear/jpeg"
    assert config.head_state.topic == "/robot/head_state_test"
    assert config.gaze_target.topic == "/visual_events/gaze_test"
    assert config.logging.jsonl_path == log_path


@pytest.mark.parametrize(
    "body, expected",
    [
        ("[camera]\nimage_topic = \"camera/image/jpeg\"\n", "image_topic"),
        ("[head_state]\ntopic = \"robot/head_state\"\n", "head_state"),
        ("[gaze_target]\ntopic = \"visual_events/gaze_target\"\n", "gaze"),
    ],
)
def test_load_config_rejects_invalid_topics(tmp_path, body, expected):
    module = import_config_module()
    config_path = tmp_path / "cli.toml"
    config_path.write_text(body, encoding="utf-8")

    with pytest.raises(config_error(module), match=expected):
        module.load_config(config_path)


@pytest.mark.parametrize(
    "body, expected",
    [
        ("[service]\nresponse_timeout_ms = 0\n", "response_timeout_ms"),
        ("[head_state]\nstale_ms = -1\n", "stale_ms"),
        ("[gaze_target]\nstale_ms = 0\n", "stale_ms"),
    ],
)
def test_load_config_rejects_non_positive_timeouts(tmp_path, body, expected):
    module = import_config_module()
    config_path = tmp_path / "cli.toml"
    config_path.write_text(body, encoding="utf-8")

    with pytest.raises(config_error(module), match=expected):
        module.load_config(config_path)


def test_load_config_rejects_unknown_botified_event(tmp_path):
    module = import_config_module()
    config_path = tmp_path / "cli.toml"
    config_path.write_text(
        """
[botified]
allowed_events = ["person_appeared", "attention_target_changed"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(config_error(module), match="allowed_events"):
        module.load_config(config_path)


@pytest.mark.parametrize(
    "body, expected",
    [
        ("[head_state]\nstationary_yaw_vel_rad_s = -0.01\n", "stationary_yaw"),
        ("[head_state]\nstationary_pitch_vel_rad_s = -0.01\n", "stationary_pitch"),
        ("[head_state]\nreport_segments = []\n", "report_segments"),
        ("[head_state]\nreport_segments = [\"stationary\", \"bad\"]\n", "report_segments"),
        ("[botified]\nevent_ttl_secs = 0\n", "event_ttl_secs"),
        ("[botified]\nstdout_queue_max = 0\n", "stdout_queue_max"),
        ("[botified]\nstdout_drop_policy = \"drop_newest\"\n", "stdout_drop_policy"),
        ("[botified]\nbroken_pipe = \"fail_fast\"\n", "broken_pipe"),
    ],
)
def test_load_config_rejects_invalid_ga_config_fields(tmp_path, body, expected):
    module = import_config_module()
    config_path = tmp_path / "cli.toml"
    config_path.write_text(body, encoding="utf-8")

    with pytest.raises(config_error(module), match=expected):
        module.load_config(config_path)


def test_load_config_rejects_reconnect_min_greater_than_max(tmp_path):
    module = import_config_module()
    config_path = tmp_path / "cli.toml"
    config_path.write_text(
        """
[service]
reconnect_min_ms = 3001
reconnect_max_ms = 3000
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(config_error(module), match="reconnect_min_ms"):
        module.load_config(config_path)
