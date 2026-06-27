from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from visual_events_cli.frame_pump import HeadMotion, InputFrame

_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8)})


@dataclass(frozen=True)
class CameraJpegMessage:
    camera: str
    timestamp_ms: int
    width: int
    height: int
    encoding: str
    data: bytes

    def to_input_frame(self) -> InputFrame | None:
        if self.encoding != "JPEG":
            return None
        if not _is_int(self.timestamp_ms):
            return None
        if not (_is_positive_int(self.width) and _is_positive_int(self.height)):
            return None
        if not isinstance(self.data, bytes):
            return None
        if _parse_jpeg_dimensions(self.data) != (self.width, self.height):
            return None

        return InputFrame(
            camera=self.camera,
            timestamp_ms=self.timestamp_ms,
            width=self.width,
            height=self.height,
            jpeg=self.data,
        )


@dataclass(frozen=True)
class HeadStateSample:
    timestamp_ms: int
    valid: bool
    yaw_rad: float
    pitch_rad: float
    yaw_vel_rad_s: float
    pitch_vel_rad_s: float

    def to_head_motion(
        self,
        *,
        now_ms: int,
        stale_ms: int,
        stationary_yaw_vel_rad_s: float,
        stationary_pitch_vel_rad_s: float,
    ) -> HeadMotion:
        if not self._is_usable(now_ms=now_ms, stale_ms=stale_ms):
            return _unknown_motion()

        if not (
            _is_finite_number(stationary_yaw_vel_rad_s)
            and _is_finite_number(stationary_pitch_vel_rad_s)
        ):
            return _unknown_motion()

        yaw_vel = float(self.yaw_vel_rad_s)
        pitch_vel = float(self.pitch_vel_rad_s)
        yaw_threshold = float(stationary_yaw_vel_rad_s)
        pitch_threshold = float(stationary_pitch_vel_rad_s)
        state = (
            "moving"
            if abs(yaw_vel) > yaw_threshold or abs(pitch_vel) > pitch_threshold
            else "stationary"
        )
        return HeadMotion(
            state=state,
            yaw_vel_rad_s=yaw_vel,
            pitch_vel_rad_s=pitch_vel,
        )

    def _is_usable(self, *, now_ms: int, stale_ms: int) -> bool:
        if not _is_int(self.timestamp_ms) or not _is_int(now_ms) or not _is_int(stale_ms):
            return False
        if type(self.valid) is not bool or not self.valid:
            return False
        if not all(
            _is_finite_number(value)
            for value in (
                self.yaw_rad,
                self.pitch_rad,
                self.yaw_vel_rad_s,
                self.pitch_vel_rad_s,
            )
        ):
            return False
        age_ms = now_ms - self.timestamp_ms
        return 0 <= age_ms <= stale_ms


def _unknown_motion() -> HeadMotion:
    return HeadMotion(state="unknown")


def _parse_jpeg_dimensions(jpeg: bytes) -> tuple[int, int] | None:
    if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
        return None

    index = 2
    payload_len = len(jpeg)
    while index < payload_len - 2:
        if jpeg[index] != 0xFF:
            return None

        while index < payload_len and jpeg[index] == 0xFF:
            index += 1
        if index >= payload_len:
            return None

        marker = jpeg[index]
        index += 1

        if marker == 0x00:
            return None
        if marker == 0xD9:
            break
        if marker in _STANDALONE_MARKERS:
            continue

        if index + 2 > payload_len:
            return None
        segment_length = int.from_bytes(jpeg[index : index + 2], "big")
        index += 2
        if segment_length < 2:
            return None

        segment_payload_length = segment_length - 2
        segment_end = index + segment_payload_length
        if segment_end > payload_len:
            return None

        if marker in _SOF_MARKERS:
            if segment_payload_length < 5:
                return None
            height = int.from_bytes(jpeg[index + 1 : index + 3], "big")
            width = int.from_bytes(jpeg[index + 3 : index + 5], "big")
            if not (_is_positive_int(width) and _is_positive_int(height)):
                return None
            return width, height

        if marker == 0xDA:
            break

        index = segment_end

    return None


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))
