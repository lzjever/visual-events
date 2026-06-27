from __future__ import annotations

from typing import Any

from visual_events_cli.dds.bridge_process import DdsBridgeProcess
from visual_events_cli.frame_pump import HeadMotion, InputFrame


class BridgeDdsImageSubscriber:
    def __init__(self, process: DdsBridgeProcess) -> None:
        self._process = process

    def start(self) -> None:
        self._process.start()

    def close(self) -> None:
        self._process.close()

    def poll_latest(self) -> InputFrame | None:
        return self._process.poll_latest_camera()


class BridgeDdsHeadStateSubscriber:
    def __init__(
        self,
        process: DdsBridgeProcess,
        *,
        stale_ms: int = 250,
        stationary_yaw_vel_rad_s: float = 0.03,
        stationary_pitch_vel_rad_s: float = 0.03,
    ) -> None:
        self._process = process
        self._stale_ms = int(stale_ms)
        self._stationary_yaw_vel_rad_s = float(stationary_yaw_vel_rad_s)
        self._stationary_pitch_vel_rad_s = float(stationary_pitch_vel_rad_s)

    def start(self) -> None:
        self._process.start()

    def close(self) -> None:
        self._process.close()

    def current_motion(self, now_ms: int) -> HeadMotion:
        sample = self._process.latest_head_state()
        if sample is None:
            return HeadMotion(state="unknown")
        return sample.to_head_motion(
            now_ms=now_ms,
            stale_ms=self._stale_ms,
            stationary_yaw_vel_rad_s=self._stationary_yaw_vel_rad_s,
            stationary_pitch_vel_rad_s=self._stationary_pitch_vel_rad_s,
        )


class BridgeDdsGazeTargetPublisher:
    def __init__(self, process: DdsBridgeProcess) -> None:
        self._process = process

    def start(self) -> None:
        self._process.start()

    def close(self) -> None:
        self._process.close()

    def publish(self, payload: Any) -> None:
        self._process.send_gaze_target(payload)
