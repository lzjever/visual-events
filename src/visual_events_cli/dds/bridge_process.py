from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from visual_events_cli.dds.bridge_protocol import (
    BridgeErrorFrame,
    BridgeHeadStateFrame,
    ProtocolError,
    decode_bridge_line,
    encode_gaze_target_line,
)
from visual_events_cli.dds.types import CameraJpegMessage, HeadStateSample
from visual_events_cli.frame_pump import InputFrame


_STOP_WRITER = object()


@dataclass(frozen=True)
class DdsBridgeProcessConfig:
    bridge_bin: str
    dds_domain: int
    dds_network: str
    camera_topic: str
    head_state_topic: str
    gaze_topic: str
    logical_camera_name: str
    base_env: Mapping[str, str] | None = None

    def build_env(self, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
        source_env = base_env if base_env is not None else self.base_env
        if source_env is None:
            env = dict(os.environ)
        else:
            env = {str(key): str(value) for key, value in source_env.items()}

        env.update(
            {
                "VISUAL_EVENTS_DDS_DOMAIN": str(self.dds_domain),
                "VISUAL_EVENTS_DDS_NETWORK": str(self.dds_network),
                "VISUAL_EVENTS_CAMERA_TOPIC": str(self.camera_topic),
                "VISUAL_EVENTS_HEAD_STATE_TOPIC": str(self.head_state_topic),
                "VISUAL_EVENTS_GAZE_TOPIC": str(self.gaze_topic),
                "VISUAL_EVENTS_LOGICAL_CAMERA_NAME": str(self.logical_camera_name),
            }
        )
        return env


PopenCallable = Callable[..., Any]


class DdsBridgeProcess:
    def __init__(
        self,
        config: DdsBridgeProcessConfig,
        *,
        popen: PopenCallable | None = None,
        close_timeout_seconds: float = 0.2,
        wall_clock_ms: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        self._config = config
        self._popen = popen or subprocess.Popen
        self._close_timeout_seconds = max(0.0, float(close_timeout_seconds))
        self._wall_clock_ms = wall_clock_ms or _wall_clock_ms
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        self._lock = threading.RLock()
        self._process: Any | None = None
        self._closed = False
        self._fatal_error: str | None = None
        self._monotonic_to_wall_offset_ms: int | None = None
        self._latest_camera: InputFrame | None = None
        self._latest_head: HeadStateSample | None = None
        self._writer_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self.invalid_line_count = 0
        self.dropped_camera_count = 0
        self.dropped_head_count = 0

    @property
    def writer_queue_maxsize(self) -> int:
        return self._writer_queue.maxsize

    @property
    def config(self) -> DdsBridgeProcessConfig:
        return self._config

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("closed")
            self._raise_if_failed_locked()
            if self._process is not None:
                return

            self._monotonic_to_wall_offset_ms = (
                int(self._wall_clock_ms()) - int(self._monotonic_ns()) // 1_000_000
            )
            process = self._popen(
                [self._config.bridge_bin],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._config.build_env(),
                bufsize=0,
            )
            self._process = process
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                name="visual-events-dds-bridge-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                name="visual-events-dds-bridge-stderr",
                daemon=True,
            )
            self._writer_thread = threading.Thread(
                target=self._write_stdin,
                name="visual-events-dds-bridge-stdin",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()
            self._writer_thread.start()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            threads = (
                self._writer_thread,
                self._stdout_thread,
                self._stderr_thread,
            )

        self._put_writer_stop()
        if process is not None:
            self._terminate_process(process)

        for thread in threads:
            if thread is not None:
                thread.join(timeout=self._close_timeout_seconds)

    def poll_latest_camera(self) -> InputFrame | None:
        with self._lock:
            self._ensure_started_and_healthy_locked()
            frame = self._latest_camera
            self._latest_camera = None
            return frame

    def latest_head_state(self) -> HeadStateSample | None:
        with self._lock:
            self._ensure_started_and_healthy_locked()
            return self._latest_head

    def send_gaze_target(self, payload: Any) -> None:
        with self._lock:
            self._ensure_started_and_healthy_locked()
        line = encode_gaze_target_line(payload).encode("utf-8")
        self._put_latest_writer_line(line)

    def _read_stdout(self) -> None:
        process = self._process
        stream = getattr(process, "stdout", None)
        if stream is None:
            self._mark_fatal("DDS bridge stdout pipe is missing")
            return

        while True:
            try:
                line = stream.readline()
            except Exception as exc:
                self._mark_fatal(f"DDS bridge stdout read failed: {exc}")
                return
            if not line:
                self._mark_stdout_eof()
                return
            self._handle_stdout_line(line)

    def _handle_stdout_line(self, line: str | bytes) -> None:
        try:
            frame = decode_bridge_line(
                line,
                logical_camera_name=self._config.logical_camera_name,
            )
        except (ProtocolError, UnicodeError):
            with self._lock:
                self.invalid_line_count += 1
            return

        if isinstance(frame, CameraJpegMessage):
            input_frame = frame.to_input_frame()
            if input_frame is None:
                with self._lock:
                    self.dropped_camera_count += 1
                return
            with self._lock:
                if not self._closed:
                    self._latest_camera = input_frame
            return

        if isinstance(frame, BridgeHeadStateFrame):
            sample = self._head_state_sample_from_bridge_frame(frame)
            with self._lock:
                if not self._closed:
                    self._latest_head = sample
            return

        if isinstance(frame, BridgeErrorFrame) and frame.fatal:
            self._mark_fatal(
                f"DDS bridge fatal error {frame.code}: {frame.message}".strip()
            )

    def _drain_stderr(self) -> None:
        process = self._process
        stream = getattr(process, "stderr", None)
        if stream is None:
            return

        while True:
            try:
                line = stream.readline()
            except Exception:
                return
            if not line:
                return

    def _write_stdin(self) -> None:
        process = self._process
        stream = getattr(process, "stdin", None)
        if stream is None:
            self._mark_fatal("DDS bridge stdin pipe is missing")
            return

        while True:
            item = self._writer_queue.get()
            if item is _STOP_WRITER:
                return
            try:
                stream.write(item)
                stream.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._mark_fatal(f"DDS bridge stdin write failed: {exc}")
                return

    def _head_state_sample_from_bridge_frame(
        self,
        frame: BridgeHeadStateFrame,
    ) -> HeadStateSample:
        offset_ms = self._monotonic_to_wall_offset_ms
        if offset_ms is None:
            offset_ms = int(self._wall_clock_ms()) - int(self._monotonic_ns()) // 1_000_000
        return HeadStateSample(
            timestamp_ms=frame.received_monotonic_ns // 1_000_000 + offset_ms,
            valid=frame.valid,
            yaw_rad=frame.yaw_rad,
            pitch_rad=frame.pitch_rad,
            yaw_vel_rad_s=frame.yaw_vel_rad_s,
            pitch_vel_rad_s=frame.pitch_vel_rad_s,
        )

    def _terminate_process(self, process: Any) -> None:
        try:
            returncode = process.poll()
        except Exception:
            returncode = None

        if returncode is None:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=self._close_timeout_seconds)
                return
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
            except Exception:
                return

        try:
            process.wait(timeout=self._close_timeout_seconds)
        except Exception:
            pass

    def _put_latest_writer_line(self, line: bytes) -> None:
        while True:
            try:
                self._writer_queue.put_nowait(line)
                return
            except queue.Full:
                try:
                    self._writer_queue.get_nowait()
                except queue.Empty:
                    pass

    def _put_writer_stop(self) -> None:
        while True:
            try:
                self._writer_queue.put_nowait(_STOP_WRITER)
                return
            except queue.Full:
                try:
                    self._writer_queue.get_nowait()
                except queue.Empty:
                    pass

    def _ensure_started_and_healthy_locked(self) -> None:
        if self._closed:
            raise RuntimeError("closed")
        if self._process is None:
            raise RuntimeError("not started")
        self._raise_if_failed_locked()

    def _raise_if_failed_locked(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)

        process = self._process
        if process is None:
            return
        try:
            returncode = process.poll()
        except Exception:
            return
        if returncode is not None:
            self._fatal_error = _process_exit_message(returncode)
            raise RuntimeError(self._fatal_error)

    def _mark_fatal(self, message: str) -> None:
        with self._lock:
            if not self._closed and self._fatal_error is None:
                self._fatal_error = message

    def _mark_stdout_eof(self) -> None:
        with self._lock:
            if self._closed or self._fatal_error is not None:
                return
            process = self._process

        returncode: int | None = None
        if process is not None:
            try:
                returncode = process.poll()
            except Exception:
                returncode = None
            if returncode is None and self._close_timeout_seconds > 0:
                try:
                    returncode = process.wait(timeout=self._close_timeout_seconds)
                except subprocess.TimeoutExpired:
                    returncode = None
                except Exception:
                    returncode = None

        if returncode is None:
            message = "DDS bridge stdout closed unexpectedly"
        else:
            message = _process_exit_message(returncode)
        self._mark_fatal(message)


def _process_exit_message(returncode: int) -> str:
    if returncode == 0:
        return "DDS bridge process exited unexpectedly with code 0"
    return f"DDS bridge process exited with code {returncode}"


def _wall_clock_ms() -> int:
    return time.time_ns() // 1_000_000
