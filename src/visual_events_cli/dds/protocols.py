from __future__ import annotations

from typing import Any, Protocol

from visual_events_cli.frame_pump import HeadMotion, InputFrame


class DdsImageSubscriber(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def poll_latest(self) -> InputFrame | None: ...


class DdsHeadStateSubscriber(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def current_motion(self, now_ms: int) -> HeadMotion: ...


class DdsGazeTargetPublisher(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def publish(self, payload: Any) -> None: ...

