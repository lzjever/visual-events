from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DdsQosProfile:
    reliability: str
    durability: str
    history: str
    depth: int
    deadline_ms: int
    lifespan_ms: int
    liveliness_lease_ms: int


CAMERA_JPEG_QOS = DdsQosProfile(
    reliability="best_effort",
    durability="volatile",
    history="keep_last",
    depth=1,
    deadline_ms=150,
    lifespan_ms=300,
    liveliness_lease_ms=1000,
)

HEAD_STATE_QOS = DdsQosProfile(
    reliability="best_effort",
    durability="volatile",
    history="keep_last",
    depth=1,
    deadline_ms=150,
    lifespan_ms=250,
    liveliness_lease_ms=500,
)

GAZE_TARGET_QOS = DdsQosProfile(
    reliability="best_effort",
    durability="volatile",
    history="keep_last",
    depth=1,
    deadline_ms=150,
    lifespan_ms=250,
    liveliness_lease_ms=500,
)

