# val-data Manifest v1

This document defines the repo-tracked schema contract for an ignored
`val-data/manifest.json`. The manifest is dataset evidence for PC local E2E
reports. It is not committed with the dataset.

## Top-Level Fields

Required fields:

| Field | Type | Requirement |
| --- | --- | --- |
| `schema_version` | integer | Must be `schema_version: 1`. |
| `fps` | number | Positive numeric source frame rate. |
| `scene_count` | integer | Must be a positive integer and match the actual scene directory inventory. |
| `frame_count` | integer | Must be a positive integer and match the total JPEG frame inventory. |
| `scenes` | array | One entry per actual scene directory, with no duplicates. |
| `oracle` | object | Oracle source contract skeleton. |

Each `scenes[]` entry requires:

| Field | Type | Requirement |
| --- | --- | --- |
| `scene_name` | string | Non-empty scene directory name. |
| `frame_count` | integer | Must match the generated scene inventory. |
| `scene_sha256` | string | 64-hex scene summary digest from the generated inventory. |

The validator rejects duplicate scene names, missing actual scenes, unknown
manifest scenes, per-scene frame count mismatches, and per-scene digest
mismatches.

## Oracle Contract Skeleton

The `oracle` object records where the authoritative expected timelines come
from. Required fields:

| Field | Type | Requirement |
| --- | --- | --- |
| `expected_event_timeline.source` | string | Non-empty source identifier. |
| `expected_event_timeline.version` | string | Non-empty version identifier. |
| `expected_attention_target_timeline.source` | string | Non-empty source identifier. |
| `expected_attention_target_timeline.rule` | string | Non-empty attention rule identifier. |

This schema only proves that the expected oracle contract is present. It does not evaluate event correctness, attention target correctness, latency, soak, fault handling, release readiness, or hardware behavior.
A valid manifest does not complete the full PC GA gate.
