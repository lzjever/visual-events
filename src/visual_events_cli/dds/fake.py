from __future__ import annotations

from typing import Any

from visual_events_cli.dds.types import CameraJpegMessage, HeadStateSample
from visual_events_cli.frame_pump import HeadMotion, InputFrame


class FakeDdsImageSubscriber:
    def __init__(self) -> None:
        self._latest: InputFrame | None = None
        self._closed = False

    def start(self) -> None:
        self._ensure_open()

    def close(self) -> None:
        self._closed = True

    def push(self, message: CameraJpegMessage) -> None:
        self._ensure_open()
        frame = message.to_input_frame()
        if frame is not None:
            self._latest = frame

    def poll_latest(self) -> InputFrame | None:
        self._ensure_open()
        frame = self._latest
        self._latest = None
        return frame

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("closed")


class FakeDdsHeadStateSubscriber:
    def __init__(
        self,
        *,
        stale_ms: int = 250,
        stationary_yaw_vel_rad_s: float = 0.03,
        stationary_pitch_vel_rad_s: float = 0.03,
    ) -> None:
        self._latest: HeadStateSample | None = None
        self._closed = False
        self._stale_ms = stale_ms
        self._stationary_yaw_vel_rad_s = stationary_yaw_vel_rad_s
        self._stationary_pitch_vel_rad_s = stationary_pitch_vel_rad_s

    def start(self) -> None:
        self._ensure_open()

    def close(self) -> None:
        self._closed = True

    def push(self, sample: HeadStateSample) -> None:
        self._ensure_open()
        self._latest = sample

    def current_motion(self, now_ms: int) -> HeadMotion:
        self._ensure_open()
        if self._latest is None:
            return HeadMotion(state="unknown")
        return self._latest.to_head_motion(
            now_ms=now_ms,
            stale_ms=self._stale_ms,
            stationary_yaw_vel_rad_s=self._stationary_yaw_vel_rad_s,
            stationary_pitch_vel_rad_s=self._stationary_pitch_vel_rad_s,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("closed")


class FakeDdsGazeTargetPublisher:
    def __init__(self) -> None:
        self._payloads: list[Any] = []
        self._closed = False

    def start(self) -> None:
        self._ensure_open()

    def close(self) -> None:
        self._closed = True

    def publish(self, payload: Any) -> None:
        self._ensure_open()
        self._payloads.append(payload)

    def latest(self) -> Any | None:
        if not self._payloads:
            return None
        return self._payloads[-1]

    def all(self) -> list[Any]:
        return list(self._payloads)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("closed")
