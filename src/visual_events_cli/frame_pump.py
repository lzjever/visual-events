from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from visual_events_cli.botified_output import BotifiedEventMapper
from visual_events_cli.target_mapper import (
    make_invalid_gaze_target,
    map_visual_state_to_gaze_target,
)


@dataclass(frozen=True)
class InputFrame:
    camera: str
    timestamp_ms: int
    width: int
    height: int
    jpeg: bytes


@dataclass(frozen=True)
class HeadMotion:
    state: str
    yaw_vel_rad_s: float | None = None
    pitch_vel_rad_s: float | None = None


class LatestFrameSlot:
    def __init__(self) -> None:
        self._frame: InputFrame | None = None

    def push(self, frame: InputFrame) -> None:
        self._frame = frame

    def pop_latest(self) -> InputFrame | None:
        frame = self._frame
        self._frame = None
        return frame


class FramePump:
    def __init__(
        self,
        *,
        latest_frame_slot: LatestFrameSlot,
        service_client: Any,
        gaze_publisher: Any,
        head_motion_provider: Callable[[], HeadMotion],
        botified_writer: Any | None = None,
        stale_after_ms: int = 250,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._latest_frame_slot = latest_frame_slot
        self._service_client = service_client
        self._gaze_publisher = gaze_publisher
        self._head_motion_provider = head_motion_provider
        self._botified_writer = botified_writer
        self._stale_after_ms = int(stale_after_ms)
        self._clock_ms = clock_ms or _wall_clock_ms
        self._next_frame_id = 1
        self._botified_mapper = BotifiedEventMapper()
        self._last_metadata: dict[str, Any] | None = None
        self._last_fresh_publish_timestamp_ms: int | None = None
        self._last_sent_metadata: dict[str, Any] | None = None
        self._last_sent_timestamp_ms: int | None = None
        self._stale_published_for_key: tuple[Any, ...] | None = None

    async def process_one(self, now_ms: int) -> bool:
        frame = self._latest_frame_slot.pop_latest()
        if frame is None:
            return False

        frame_id = self._next_frame_id
        self._next_frame_id += 1
        header = self._build_header(frame, frame_id)
        self._last_sent_metadata = _metadata_from_input_frame(frame, frame_id)
        self._last_sent_timestamp_ms = int(now_ms)
        result = await self._service_client.request_frame(header, frame.jpeg)

        visual_state = getattr(result, "visual_state", None)
        if isinstance(visual_state, dict):
            self._publish_fresh_visual_state(
                visual_state,
                publish_timestamp_ms=int(self._clock_ms()),
            )
            self._last_sent_metadata = None
            self._last_sent_timestamp_ms = None
            self._write_botified_frames(visual_state)
        return True

    def publish_stale_now(self, publish_timestamp_ms: int) -> bool:
        metadata = self._last_metadata or self._last_sent_metadata
        if metadata is None:
            return False

        key = self._metadata_key(metadata)
        if self._stale_published_for_key == key:
            return False

        payload = make_invalid_gaze_target(
            "stale",
            camera=str(metadata["camera"]),
            frame_id=int(metadata["frame_id"]),
            frame_timestamp_ms=int(metadata["frame_timestamp_ms"]),
            image_size=(
                int(metadata["image_width"]),
                int(metadata["image_height"]),
            ),
            publish_timestamp_ms=publish_timestamp_ms,
            stale_after_ms=self._stale_after_ms,
        )
        self._gaze_publisher.publish(payload)
        self._stale_published_for_key = key
        return True

    def check_stale_deadline(self, now_ms: int) -> bool:
        stale_reference_ms = self._last_fresh_publish_timestamp_ms
        if stale_reference_ms is None:
            stale_reference_ms = self._last_sent_timestamp_ms
        if stale_reference_ms is None:
            return False
        if now_ms - stale_reference_ms < self._stale_after_ms:
            return False
        return self.publish_stale_now(publish_timestamp_ms=now_ms)

    def _build_header(self, frame: InputFrame, frame_id: int) -> dict[str, Any]:
        head_motion = self._head_motion_provider()
        return {
            "type": "frame",
            "schema_version": 1,
            "camera": frame.camera,
            "frame_id": frame_id,
            "timestamp_ms": int(frame.timestamp_ms),
            "encoding": "jpeg",
            "width": int(frame.width),
            "height": int(frame.height),
            "head_motion": {
                "state": head_motion.state,
                "yaw_vel_rad_s": head_motion.yaw_vel_rad_s,
                "pitch_vel_rad_s": head_motion.pitch_vel_rad_s,
            },
        }

    def _publish_fresh_visual_state(
        self,
        visual_state: dict[str, Any],
        *,
        publish_timestamp_ms: int,
    ) -> None:
        payload = map_visual_state_to_gaze_target(
            visual_state,
            publish_timestamp_ms=publish_timestamp_ms,
            stale_after_ms=self._stale_after_ms,
        )
        self._gaze_publisher.publish(payload)
        self._last_metadata = _metadata_from_visual_state(visual_state)
        self._last_fresh_publish_timestamp_ms = int(publish_timestamp_ms)
        self._stale_published_for_key = None

    def _write_botified_frames(self, visual_state: dict[str, Any]) -> None:
        if self._botified_writer is None:
            return
        for frame in self._botified_mapper.frames_from_visual_state(visual_state):
            if hasattr(self._botified_writer, "enqueue"):
                self._botified_writer.enqueue(frame)
            else:
                self._botified_writer.write(frame)

    def _metadata_key(self, metadata: dict[str, Any]) -> tuple[Any, ...]:
        return (
            metadata["camera"],
            metadata["frame_id"],
            metadata["frame_timestamp_ms"],
            metadata["image_width"],
            metadata["image_height"],
        )


def _metadata_from_visual_state(visual_state: dict[str, Any]) -> dict[str, Any]:
    image_size = visual_state.get("image_size")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        image_size = (0, 0)

    return {
        "camera": str(visual_state.get("camera", "")),
        "frame_id": _as_int(visual_state.get("frame_id"), -1),
        "frame_timestamp_ms": _as_int(visual_state.get("frame_timestamp_ms"), 0),
        "image_width": _as_int(image_size[0], 0),
        "image_height": _as_int(image_size[1], 0),
    }


def _metadata_from_input_frame(frame: InputFrame, frame_id: int) -> dict[str, Any]:
    return {
        "camera": str(frame.camera),
        "frame_id": int(frame_id),
        "frame_timestamp_ms": int(frame.timestamp_ms),
        "image_width": int(frame.width),
        "image_height": int(frame.height),
    }


def _wall_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default
