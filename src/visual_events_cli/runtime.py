from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from visual_events_cli.botified_output import BotifiedPipeClosed
from visual_events_cli.frame_pump import FramePump, LatestFrameSlot


EXIT_BOTIFIED_PIPE_CLOSED = 3
BOTIFIED_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 0.01
RUNTIME_UNAVAILABLE_MESSAGE = "Step 4 DDS adapters not implemented"


class RuntimeUnavailable(Exception):
    """Raised when the production runtime cannot be constructed."""


ResourceFactory = Callable[[Any], Any]


def _none_resource_factory(_config: Any) -> None:
    return None


@dataclass(frozen=True)
class RuntimeFactories:
    image_subscriber: ResourceFactory
    head_state_subscriber: ResourceFactory
    gaze_publisher: ResourceFactory
    service_client: ResourceFactory
    botified_writer: ResourceFactory = _none_resource_factory


def default_runtime_factories() -> RuntimeFactories:
    return RuntimeFactories(
        image_subscriber=_unimplemented_resource_factory,
        head_state_subscriber=_unimplemented_resource_factory,
        gaze_publisher=_unimplemented_resource_factory,
        service_client=_unimplemented_resource_factory,
        botified_writer=_none_resource_factory,
    )


def run_runtime(
    config: Any,
    *,
    factories: RuntimeFactories | None = None,
    clock_ms: Callable[[], int] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    max_ticks: int | None = None,
    sleep_seconds: float = 0.01,
) -> int:
    clock = clock_ms or _wall_clock_ms
    runtime_factories = factories or default_runtime_factories()
    resources = _ResourceRegistry()

    try:
        image_subscriber = _create_and_start(
            runtime_factories.image_subscriber,
            config,
            resources,
        )

        head_subscriber = _create_and_start(
            runtime_factories.head_state_subscriber,
            config,
            resources,
        )

        gaze_publisher = _create_and_start(
            runtime_factories.gaze_publisher,
            config,
            resources,
        )

        service_client = _wrap_service_client_for_logging(
            resources.register(runtime_factories.service_client(config)),
            config,
        )
        botified_writer = resources.register(runtime_factories.botified_writer(config))
        coordinator = RuntimeCoordinator(
            frame_source=image_subscriber,
            service_client=service_client,
            gaze_publisher=gaze_publisher,
            head_motion_provider=lambda: head_subscriber.current_motion(
                now_ms=int(clock())
            ),
            botified_writer=botified_writer,
            stale_after_ms=int(config.gaze_target.stale_ms),
            clock_ms=clock,
        )
        return asyncio.run(
            _run_runtime_loop(
                coordinator,
                stop_requested=stop_requested,
                max_ticks=max_ticks,
                sleep_seconds=sleep_seconds,
                resources=resources,
            )
        )
    except BaseException:
        _close_resources_from_sync(resources, suppress_errors=True)
        raise


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
        botified_shutdown_drain_timeout_seconds: float = (
            BOTIFIED_SHUTDOWN_DRAIN_TIMEOUT_SECONDS
        ),
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
        self._drain_task: asyncio.Future[None] | None = None
        self._botified_shutdown_drain_timeout_seconds = max(
            0.0,
            float(botified_shutdown_drain_timeout_seconds),
        )
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
        self._drain_task = _run_in_daemon_thread(drain)

    def _handle_botified_pipe_closed(self) -> int:
        self._pump.publish_stale_now(publish_timestamp_ms=int(self._clock_ms()))
        self._exit_code = EXIT_BOTIFIED_PIPE_CLOSED
        return self._exit_code

    @property
    def exit_code(self) -> int | None:
        return self._exit_code

    async def shutdown(self) -> None:
        await self._shutdown_task("_process_task")
        await self._shutdown_botified_drain_task()

    async def _shutdown_task(self, attribute_name: str) -> None:
        task = getattr(self, attribute_name)
        if task is None:
            return

        setattr(self, attribute_name, None)
        if not task.done():
            task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        except BotifiedPipeClosed:
            self._handle_botified_pipe_closed()

    async def _shutdown_botified_drain_task(self) -> None:
        task = self._drain_task
        if task is None:
            return

        self._drain_task = None
        if not task.done():
            await self._observe_botified_drain_task_for_shutdown(task)

        if task.done():
            self._finish_botified_drain_task_result(task)
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except BotifiedPipeClosed:
            self._handle_botified_pipe_closed()

    async def _observe_botified_drain_task_for_shutdown(
        self,
        task: asyncio.Future[None],
    ) -> None:
        timeout_seconds = self._botified_shutdown_drain_timeout_seconds
        if timeout_seconds <= 0:
            await asyncio.sleep(0)
            return

        await asyncio.wait({task}, timeout=timeout_seconds)

    def _finish_botified_drain_task_result(
        self,
        task: asyncio.Future[None],
    ) -> None:
        try:
            task.result()
        except BotifiedPipeClosed:
            self._handle_botified_pipe_closed()


def _wall_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _wrap_service_client_for_logging(service_client: Any, config: Any) -> Any:
    jsonl_path = getattr(getattr(config, "logging", None), "jsonl_path", None)
    if jsonl_path is None:
        return service_client
    return _FrameRequestLoggingServiceClient(service_client, Path(jsonl_path))


class _FrameRequestLoggingServiceClient:
    def __init__(self, service_client: Any, jsonl_path: Path) -> None:
        self._service_client = service_client
        self._jsonl_path = jsonl_path
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    async def request_frame(self, header: dict[str, Any], jpeg: bytes) -> Any:
        self._write_frame_request(header)
        return await self._service_client.request_frame(header, jpeg)

    def _write_frame_request(self, header: dict[str, Any]) -> None:
        payload = {
            "type": "frame_request",
            "frame_id": header.get("frame_id"),
            "timestamp_ms": header.get("timestamp_ms"),
            "camera": header.get("camera"),
            "head_motion": header.get("head_motion"),
        }
        with self._jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _run_in_daemon_thread(function: Callable[[], None]) -> asyncio.Future[None]:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()

    def set_result() -> None:
        if not future.done():
            future.set_result(None)

    def run() -> None:
        try:
            function()
        except BaseException as exc:

            def set_exception(exc: BaseException = exc) -> None:
                if not future.done():
                    future.set_exception(exc)

            _notify_loop_threadsafe(loop, set_exception)
        else:
            _notify_loop_threadsafe(loop, set_result)

    thread = threading.Thread(
        target=run,
        name="visual-events-botified-drain",
        daemon=True,
    )
    thread.start()
    return future


def _notify_loop_threadsafe(
    loop: asyncio.AbstractEventLoop,
    callback: Callable[[], None],
) -> None:
    try:
        loop.call_soon_threadsafe(callback)
    except RuntimeError:
        pass


def _unimplemented_resource_factory(_config: Any) -> Any:
    raise RuntimeUnavailable(RUNTIME_UNAVAILABLE_MESSAGE)


def _create_and_start(
    factory: ResourceFactory,
    config: Any,
    resources: "_ResourceRegistry",
) -> Any:
    resource = factory(config)
    try:
        resource.start()
    except BaseException:
        _close_single_resource_from_sync(resource, suppress_errors=True)
        raise
    return resources.register(resource)


async def _run_runtime_loop(
    coordinator: RuntimeCoordinator,
    *,
    stop_requested: Callable[[], bool] | None,
    max_ticks: int | None,
    sleep_seconds: float,
    resources: "_ResourceRegistry",
) -> int:
    suppress_cleanup_errors = False
    result: int | None = None
    ticks = 0
    try:
        while True:
            if stop_requested is not None and stop_requested():
                result = 0
                break
            if max_ticks is not None and ticks >= max_ticks:
                result = 0
                break

            exit_code = await coordinator.run_once()
            ticks += 1
            if exit_code is not None:
                result = int(exit_code)
                break

            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)
            else:
                await asyncio.sleep(0)
    except BaseException:
        suppress_cleanup_errors = True
        raise
    finally:
        await _shutdown_and_close_resources(
            coordinator,
            resources,
            suppress_errors=suppress_cleanup_errors,
        )

    if coordinator.exit_code is not None:
        return int(coordinator.exit_code)
    return int(result or 0)


class _ResourceRegistry:
    def __init__(self) -> None:
        self._resources: list[Any] = []
        self._closed = False

    def register(self, resource: Any) -> Any:
        if resource is not None:
            self._resources.append(resource)
        return resource

    async def close_all(self, *, suppress_errors: bool) -> None:
        if self._closed:
            return

        self._closed = True
        first_error: BaseException | None = None
        while self._resources:
            resource = self._resources.pop()
            try:
                await _close_resource(resource)
            except BaseException as exc:
                if not suppress_errors and first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error


async def _shutdown_and_close_resources(
    coordinator: RuntimeCoordinator,
    resources: _ResourceRegistry,
    *,
    suppress_errors: bool,
) -> None:
    first_error: BaseException | None = None
    try:
        await coordinator.shutdown()
    except BaseException as exc:
        if not suppress_errors:
            first_error = exc

    try:
        await resources.close_all(suppress_errors=suppress_errors)
    except BaseException as exc:
        if not suppress_errors and first_error is None:
            first_error = exc

    if first_error is not None:
        raise first_error


async def _close_resource(resource: Any) -> None:
    close = _close_method(resource)
    if close is None:
        return

    result = close()
    if inspect.isawaitable(result):
        await result


def _close_method(resource: Any) -> Callable[[], Any] | None:
    for method_name in ("close", "_close_connection"):
        method = getattr(resource, method_name, None)
        if callable(method):
            return method
    return None


def _close_single_resource_from_sync(resource: Any, *, suppress_errors: bool) -> None:
    try:
        asyncio.run(_close_resource(resource))
    except BaseException:
        if not suppress_errors:
            raise


def _close_resources_from_sync(
    resources: _ResourceRegistry,
    *,
    suppress_errors: bool,
) -> None:
    try:
        asyncio.run(resources.close_all(suppress_errors=suppress_errors))
    except BaseException:
        if not suppress_errors:
            raise
