from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FORBIDDEN_AGENT_FIELDS = frozenset(
    {
        "track_id",
        "bbox",
        "bbox_xyxy",
        "point_uv",
        "keypoints",
        "embedding",
        "crop",
        "crop_ref",
        "test_hint",
        "source_scene",
        "source_frame",
        "source_frame_ref",
        "request_snapshot_ref",
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


class _RejectLowLevelFieldsModel(_StrictModel):
    @model_validator(mode="after")
    def _reject_low_level_fields(self) -> "_RejectLowLevelFieldsModel":
        forbidden_path = _find_forbidden_agent_field(self.model_dump())
        if forbidden_path is not None:
            field_path = ".".join(forbidden_path)
            raise ValueError(
                "agent-facing memory payload must not include low-level field "
                f"{field_path}"
            )
        return self


class _BaseMemoryRequest(_RejectLowLevelFieldsModel):
    camera: str = Field(min_length=1)
    stream_ref: str = Field(min_length=1)

    def _base_internal_request(self) -> dict[str, Any]:
        return {"camera": self.camera, "stream_ref": self.stream_ref}


class TeachPersonRequest(_BaseMemoryRequest):
    target: PersonTarget
    profile: dict[str, Any]

    def to_internal_request(self) -> dict[str, Any]:
        return {
            **self._base_internal_request(),
            "target": self.target.model_dump(),
            "profile": dict(self.profile),
        }


class IdentifyCurrentRequest(_BaseMemoryRequest):
    target: PersonTarget
    scope: Literal["active_target"] = "active_target"
    timeout_ms: int = Field(default=500, ge=1, le=1000)

    def to_internal_request(self) -> dict[str, Any]:
        return {
            **self._base_internal_request(),
            "target": self.target.model_dump(),
            "scope": self.scope,
            "timeout_ms": self.timeout_ms,
        }


class TeachSceneRequest(_BaseMemoryRequest):
    target: SceneTarget
    memory: dict[str, Any]

    def to_internal_request(self) -> dict[str, Any]:
        return {
            **self._base_internal_request(),
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
        if self.target.kind == "person":
            target: dict[str, Any] = self.target.model_dump()
        else:
            target = {"mode": "scene"}
        return {**self._base_internal_request(), "target": target}


class ConversationSummaryRequest(_RejectLowLevelFieldsModel):
    summary: str = Field(min_length=1)
    source: str | None = Field(default=None, min_length=1)
    source_conversation_id: str | None = Field(default=None, min_length=1)

    def to_internal_request(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class LinkExternalUserRequest(_RejectLowLevelFieldsModel):
    person_id: str = Field(min_length=1)
    external_user_ref: str = Field(min_length=1)

    def to_internal_request(self) -> dict[str, Any]:
        return self.model_dump()


class MergeAnonymousPersonRequest(_RejectLowLevelFieldsModel):
    anonymous_id: str = Field(min_length=1)
    person_id: str | None = Field(default=None, min_length=1)
    profile: dict[str, Any] | None = None
    merge_reason: str | None = Field(default=None, min_length=1)

    def to_internal_request(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class CorrectIdentityRequest(_RejectLowLevelFieldsModel):
    memory_match_id: str = Field(min_length=1)
    wrong_person_id: str = Field(min_length=1)

    def to_internal_request(self) -> dict[str, Any]:
        return self.model_dump()


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
