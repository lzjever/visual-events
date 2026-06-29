from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FORBIDDEN_AGENT_FIELDS = frozenset(
    {
        "track_id",
        "bbox",
        "bbox_xyxy",
        "point_uv",
        "test_hint",
        "source_scene",
        "source_frame",
    }
)
RESOLVE_TARGET_STATUSES = frozenset({"resolved", "ambiguous", "not_found"})


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _BaseTarget(_StrictModel):
    intent: str = Field(min_length=1)
    referent_text: str = Field(min_length=1)


class PersonTarget(_BaseTarget):
    kind: Literal["person"]


class SceneTarget(_BaseTarget):
    kind: Literal["scene"]


class ResolveTarget(_BaseTarget):
    kind: Literal["person", "scene", "object"]


class _BaseMemoryRequest(_StrictModel):
    camera: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_low_level_fields(self) -> "_BaseMemoryRequest":
        forbidden_path = _find_forbidden_agent_field(self.model_dump())
        if forbidden_path is not None:
            field_path = ".".join(forbidden_path)
            raise ValueError(
                "agent-facing memory payload must not include low-level field "
                f"{field_path}"
            )
        return self


class TeachPersonRequest(_BaseMemoryRequest):
    target: PersonTarget
    profile: dict[str, Any]

    def to_internal_request(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "target": {"mode": "attention_target"},
            "profile": dict(self.profile),
        }


class TeachSceneRequest(_BaseMemoryRequest):
    target: SceneTarget
    memory: dict[str, Any]

    def to_internal_request(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "target": {"mode": "scene"},
            "memory": dict(self.memory),
        }


class ResolveTargetRequest(_BaseMemoryRequest):
    target: ResolveTarget
    profile: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None

    def to_internal_request(self) -> dict[str, Any]:
        if self.target.kind == "object":
            raise ValueError("object target kind is not supported internally")
        return {
            "camera": self.camera,
            "target": {"mode": _internal_mode_for_kind(self.target.kind)},
        }


def unsupported_target_kind_response() -> dict[str, Any]:
    return {
        "ok": True,
        "status": "not_found",
        "error_code": "unsupported_target_kind",
        "retryable": False,
        "ask_user_hint": False,
        "ambiguity_type": "unsupported_target_kind",
        "candidates": [],
    }


def validate_resolve_target_response(response: dict[str, Any]) -> None:
    status = response.get("status")
    if status not in RESOLVE_TARGET_STATUSES:
        raise ValueError(
            "resolve-target response status must be one of "
            f"{sorted(RESOLVE_TARGET_STATUSES)}"
        )


def _internal_mode_for_kind(kind: Literal["person", "scene"]) -> str:
    if kind == "person":
        return "attention_target"
    return "scene"


def _find_forbidden_agent_field(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            if key_text in FORBIDDEN_AGENT_FIELDS:
                return next_path
            nested = _find_forbidden_agent_field(item, path=next_path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_forbidden_agent_field(item, path=(*path, str(index)))
            if nested is not None:
                return nested
    return None
