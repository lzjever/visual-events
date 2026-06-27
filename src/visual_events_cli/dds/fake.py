from __future__ import annotations

from typing import Any

from visual_events_cli.dds.types import CameraJpegMessage, HeadStateSample
from visual_events_cli.frame_pump import HeadMotion, InputFrame


class _FakeDdsLifecycle:
    def __init__(self) -> None:
        self._started = False
        self._closed = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self) -> None:
        self._ensure_open()
        self._started = True

    def close(self) -> None:
        self._closed = True

    def _ensure_started(self) -> None:
        self._ensure_open()
        if not self._started:
            raise RuntimeError("not started")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("closed")


class FakeDdsImageSubscriber(_FakeDdsLifecycle):
    def __init__(self) -> None:
        super().__init__()
        self._latest: InputFrame | None = None

    def push(self, message: CameraJpegMessage) -> None:
        self._ensure_started()
        frame = message.to_input_frame()
        if frame is not None:
            self._latest = frame

    def poll_latest(self) -> InputFrame | None:
        self._ensure_started()
        frame = self._latest
        self._latest = None
        return frame


class FakeDdsHeadStateSubscriber(_FakeDdsLifecycle):
    def __init__(
        self,
        *,
        stale_ms: int = 250,
        stationary_yaw_vel_rad_s: float = 0.03,
        stationary_pitch_vel_rad_s: float = 0.03,
    ) -> None:
        super().__init__()
        self._latest: HeadStateSample | None = None
        self._stale_ms = stale_ms
        self._stationary_yaw_vel_rad_s = stationary_yaw_vel_rad_s
        self._stationary_pitch_vel_rad_s = stationary_pitch_vel_rad_s

    def push(self, sample: HeadStateSample) -> None:
        self._ensure_started()
        self._latest = sample

    def current_motion(self, now_ms: int) -> HeadMotion:
        self._ensure_started()
        if self._latest is None:
            return HeadMotion(state="unknown")
        return self._latest.to_head_motion(
            now_ms=now_ms,
            stale_ms=self._stale_ms,
            stationary_yaw_vel_rad_s=self._stationary_yaw_vel_rad_s,
            stationary_pitch_vel_rad_s=self._stationary_pitch_vel_rad_s,
        )


class FakeDdsGazeTargetPublisher(_FakeDdsLifecycle):
    def __init__(self) -> None:
        super().__init__()
        self._payloads: list[Any] = []

    def publish(self, payload: Any) -> None:
        self._ensure_started()
        self._payloads.append(payload)

    def latest(self) -> Any | None:
        if not self._payloads:
            return None
        return self._payloads[-1]

    def all(self) -> list[Any]:
        return list(self._payloads)
