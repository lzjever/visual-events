from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable

from visual_events_cli.botified_output import BotifiedPipeClosed
from visual_events_cli.frame_pump import FramePump, LatestFrameSlot


EXIT_BOTIFIED_PIPE_CLOSED = 3


class RuntimeCoordinator:
    def __init__(
        self,
        *,
        frame_source: Any,
        service_client: Any,
        gaze_publisher: Any,
        head_motion_provider: Callable[[], Any],
        botified_writer: Any | None = None,
        stale_after_ms: int = 250,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._frame_source = frame_source
        self._botified_writer = botified_writer
        self._clock_ms = clock_ms or _wall_clock_ms
        self._latest_frame_slot = LatestFrameSlot()
        self._pump = FramePump(
            latest_frame_slot=self._latest_frame_slot,
            service_client=service_client,
            gaze_publisher=gaze_publisher,
            head_motion_provider=head_motion_provider,
            botified_writer=botified_writer,
            stale_after_ms=stale_after_ms,
            clock_ms=self._clock_ms,
        )
        self._process_task: asyncio.Task[bool] | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._exit_code: int | None = None

    async def run_once(self) -> int | None:
        if self._exit_code is not None:
            return self._exit_code

        frame = await self._read_latest_frame()
        if frame is not None:
            self._latest_frame_slot.push(frame)

        exit_code = self._finish_process_task_if_done()
        if exit_code is not None:
            return exit_code

        if self._process_task is None:
            self._process_task = asyncio.create_task(
                self._pump.process_one(now_ms=int(self._clock_ms()))
            )
            await asyncio.sleep(0)
            exit_code = self._finish_process_task_if_done()
            if exit_code is not None:
                return exit_code

        self._pump.check_stale_deadline(int(self._clock_ms()))

        exit_code = self._finish_botified_drain_task_if_done()
        if exit_code is not None:
            return exit_code
        self._start_botified_drain_task_if_idle()

        return None

    async def _read_latest_frame(self) -> Any | None:
        reader = getattr(self._frame_source, "poll_latest", None)
        if reader is None:
            reader = getattr(self._frame_source, "read_latest", None)
        if reader is None:
            return None

        frame = reader()
        if inspect.isawaitable(frame):
            frame = await frame
        return frame

    def _finish_process_task_if_done(self) -> int | None:
        if self._process_task is None or not self._process_task.done():
            return None

        task = self._process_task
        self._process_task = None
        try:
            task.result()
        except BotifiedPipeClosed:
            return self._handle_botified_pipe_closed()
        return None

    def _finish_botified_drain_task_if_done(self) -> int | None:
        if self._drain_task is None or not self._drain_task.done():
            return None

        task = self._drain_task
        self._drain_task = None
        try:
            task.result()
        except BotifiedPipeClosed:
            return self._handle_botified_pipe_closed()
        return None

    def _start_botified_drain_task_if_idle(self) -> None:
        if self._botified_writer is None:
            return
        if self._drain_task is not None:
            return
        drain = getattr(self._botified_writer, "drain_available", None)
        if drain is None:
            return
        self._drain_task = asyncio.create_task(asyncio.to_thread(drain))

    def _handle_botified_pipe_closed(self) -> int:
        self._pump.publish_stale_now(publish_timestamp_ms=int(self._clock_ms()))
        self._exit_code = EXIT_BOTIFIED_PIPE_CLOSED
        return self._exit_code


def _wall_clock_ms() -> int:
    return time.time_ns() // 1_000_000
