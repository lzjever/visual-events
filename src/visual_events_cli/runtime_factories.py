from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from visual_events_cli.botified_output import BotifiedStdoutWriter
from visual_events_cli.dds.bridge_adapters import (
    BridgeDdsGazeTargetPublisher,
    BridgeDdsHeadStateSubscriber,
    BridgeDdsImageSubscriber,
)
from visual_events_cli.dds.bridge_process import DdsBridgeProcess, DdsBridgeProcessConfig
from visual_events_cli.runtime import RuntimeFactories, RuntimeUnavailable
from visual_events_cli.service_client import VisualEventsServiceClient


BRIDGE_BIN_ENV = "VISUAL_EVENTS_DDS_BRIDGE_BIN"
ProcessFactory = Callable[[DdsBridgeProcessConfig], DdsBridgeProcess]


def bridge_runtime_factories(
    *,
    bridge_bin: str | None = None,
    process_factory: ProcessFactory | None = None,
) -> RuntimeFactories:
    resolved_bridge_bin = bridge_bin or os.environ.get(BRIDGE_BIN_ENV)
    if not resolved_bridge_bin:
        raise RuntimeUnavailable(f"{BRIDGE_BIN_ENV} is required for DDS bridge runtime")

    make_process = process_factory or DdsBridgeProcess
    shared_process: DdsBridgeProcess | None = None

    def get_process(config: Any) -> DdsBridgeProcess:
        nonlocal shared_process
        if shared_process is None:
            shared_process = make_process(
                DdsBridgeProcessConfig(
                    bridge_bin=str(resolved_bridge_bin),
                    dds_domain=int(config.dds.domain),
                    dds_network=str(config.dds.network),
                    camera_topic=str(config.camera.image_topic),
                    head_state_topic=str(config.head_state.topic),
                    gaze_topic=str(config.gaze_target.topic),
                    logical_camera_name=str(config.camera.name),
                )
            )
        return shared_process

    return RuntimeFactories(
        image_subscriber=lambda config: BridgeDdsImageSubscriber(get_process(config)),
        head_state_subscriber=lambda config: BridgeDdsHeadStateSubscriber(
            get_process(config),
            stale_ms=int(config.head_state.stale_ms),
            stationary_yaw_vel_rad_s=float(config.head_state.stationary_yaw_vel_rad_s),
            stationary_pitch_vel_rad_s=float(
                config.head_state.stationary_pitch_vel_rad_s
            ),
        ),
        gaze_publisher=lambda config: BridgeDdsGazeTargetPublisher(get_process(config)),
        service_client=_service_client_factory,
        botified_writer=_botified_writer_factory,
    )


def _service_client_factory(config: Any) -> VisualEventsServiceClient:
    return VisualEventsServiceClient(
        config.service.url,
        response_timeout_ms=int(config.service.response_timeout_ms),
    )


def _botified_writer_factory(config: Any) -> BotifiedStdoutWriter | None:
    if not config.botified.enabled or not config.botified.stdout:
        return None
    return BotifiedStdoutWriter(max_queue_size=int(config.botified.stdout_queue_max))
