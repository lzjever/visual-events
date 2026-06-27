from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from visual_events_cli.botified_output import BotifiedStdoutWriter
from visual_events_cli.config import default_config
from visual_events_cli.runtime import RuntimeFactories, RuntimeUnavailable
from visual_events_cli.service_client import VisualEventsServiceClient


def import_runtime_factories() -> Any:
    try:
        import visual_events_cli.runtime_factories as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected visual_events_cli.runtime_factories module: {exc}")
    return module


class FakeBridgeProcess:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.start_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_bridge_runtime_factories_missing_bridge_bin_raises_runtime_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    module = import_runtime_factories()
    monkeypatch.delenv("VISUAL_EVENTS_DDS_BRIDGE_BIN", raising=False)

    with pytest.raises(RuntimeUnavailable, match="VISUAL_EVENTS_DDS_BRIDGE_BIN"):
        module.bridge_runtime_factories()


def test_bridge_runtime_factories_return_runtime_factories_and_share_one_process():
    module = import_runtime_factories()
    created: list[FakeBridgeProcess] = []

    def process_factory(config: Any) -> FakeBridgeProcess:
        process = FakeBridgeProcess(config)
        created.append(process)
        return process

    factories = module.bridge_runtime_factories(
        bridge_bin="/opt/visual-events/dds-bridge",
        process_factory=process_factory,
    )
    config = default_config()

    image = factories.image_subscriber(config)
    head = factories.head_state_subscriber(config)
    gaze = factories.gaze_publisher(config)

    assert isinstance(factories, RuntimeFactories)
    assert len(created) == 1
    assert image._process is created[0]
    assert head._process is created[0]
    assert gaze._process is created[0]


def test_bridge_runtime_factories_map_config_to_process_config_and_env():
    module = import_runtime_factories()
    created: list[FakeBridgeProcess] = []

    def process_factory(config: Any) -> FakeBridgeProcess:
        process = FakeBridgeProcess(config)
        created.append(process)
        return process

    base = default_config()
    config = replace(
        base,
        dds=replace(base.dds, domain=57, network="lo"),
        camera=replace(base.camera, name="logical-front", image_topic="/camera/front/jpeg"),
        head_state=replace(
            base.head_state,
            topic="/robot/head_state",
            stale_ms=333,
            stationary_yaw_vel_rad_s=0.04,
            stationary_pitch_vel_rad_s=0.05,
        ),
        gaze_target=replace(
            base.gaze_target,
            topic="/visual_events/gaze_target",
            stale_ms=444,
        ),
    )

    factories = module.bridge_runtime_factories(
        bridge_bin="/opt/visual-events/dds-bridge",
        process_factory=process_factory,
    )
    head = factories.head_state_subscriber(config)

    process_config = created[0].config
    assert process_config.bridge_bin == "/opt/visual-events/dds-bridge"
    assert process_config.dds_domain == 57
    assert process_config.dds_network == "lo"
    assert process_config.camera_topic == "/camera/front/jpeg"
    assert process_config.head_state_topic == "/robot/head_state"
    assert process_config.gaze_topic == "/visual_events/gaze_target"
    assert process_config.logical_camera_name == "logical-front"

    env = process_config.build_env({"LD_LIBRARY_PATH": "/opt/dds/lib", "KEEP": "1"})
    assert env["KEEP"] == "1"
    assert env["LD_LIBRARY_PATH"] == "/opt/dds/lib"
    assert env["VISUAL_EVENTS_DDS_DOMAIN"] == "57"
    assert env["VISUAL_EVENTS_DDS_NETWORK"] == "lo"
    assert env["VISUAL_EVENTS_CAMERA_TOPIC"] == "/camera/front/jpeg"
    assert env["VISUAL_EVENTS_HEAD_STATE_TOPIC"] == "/robot/head_state"
    assert env["VISUAL_EVENTS_GAZE_TOPIC"] == "/visual_events/gaze_target"
    assert env["VISUAL_EVENTS_LOGICAL_CAMERA_NAME"] == "logical-front"

    assert head._stale_ms == 333
    assert head._stationary_yaw_vel_rad_s == pytest.approx(0.04)
    assert head._stationary_pitch_vel_rad_s == pytest.approx(0.05)


def test_bridge_runtime_factories_use_service_and_botified_config():
    module = import_runtime_factories()
    base = default_config()
    config = replace(
        base,
        service=replace(
            base.service,
            url="ws://127.0.0.1:9999/v1/stream",
            response_timeout_ms=1234,
        ),
        botified=replace(base.botified, enabled=True, stdout=True, stdout_queue_max=7),
    )

    factories = module.bridge_runtime_factories(bridge_bin="/tmp/bridge")
    service = factories.service_client(config)
    botified = factories.botified_writer(config)

    assert isinstance(service, VisualEventsServiceClient)
    assert service._url == "ws://127.0.0.1:9999/v1/stream"
    assert service._response_timeout_s == pytest.approx(1.234)
    assert isinstance(botified, BotifiedStdoutWriter)
    assert botified._max_queue_size == 7

    disabled = replace(config, botified=replace(config.botified, enabled=False))
    stdout_disabled = replace(config, botified=replace(config.botified, stdout=False))
    assert factories.botified_writer(disabled) is None
    assert factories.botified_writer(stdout_disabled) is None


def test_bridge_runtime_factories_can_read_bridge_bin_from_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    module = import_runtime_factories()
    created: list[FakeBridgeProcess] = []

    monkeypatch.setenv("VISUAL_EVENTS_DDS_BRIDGE_BIN", "/env/dds-bridge")
    factories = module.bridge_runtime_factories(
        process_factory=lambda config: created.append(FakeBridgeProcess(config)) or created[-1]
    )

    factories.image_subscriber(default_config())

    assert created[0].config.bridge_bin == "/env/dds-bridge"


def test_default_runtime_factories_remain_fail_fast():
    from visual_events_cli import runtime

    factories = runtime.default_runtime_factories()
    config = default_config()

    for factory in (
        factories.image_subscriber,
        factories.head_state_subscriber,
        factories.gaze_publisher,
    ):
        with pytest.raises(RuntimeUnavailable, match="Step 4 DDS adapters not implemented"):
            factory(config)
