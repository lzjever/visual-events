# Gaze Target V1 DDS Contract

| key | value |
| --- | --- |
| topic | /visual_events/gaze_target |
| dds_type | visual_events::msg::dds_::GazeTargetV1_ |
| reliability | best_effort |
| durability | volatile |
| history | keep_last_1 |
| deadline_ms | 150 |
| lifespan_ms | 250 |
| liveliness_lease_ms | 500 |

## Semantics

`GazeTargetV1_` publishes the selected visual target for a camera frame. It is an observation for downstream consumers, not a head-control command.

| field | type | notes |
| --- | --- | --- |
| schema_version | uint32 | Contract version, currently `1`. |
| camera | string | Logical camera name after source-to-logical mapping. |
| frame_id | int64 | Camera frame identifier associated with this target. |
| frame_timestamp_ms | int64 | Frame timestamp in milliseconds after source validation or receive-time fallback. |
| publish_timestamp_ms | int64 | Publisher wall-clock timestamp in milliseconds. |
| valid | bool | `true` when the target fields describe a current tracked target. |
| state | string | Closed set: `tracking`, `lost`, `stale`, `disabled`. |
| target_track_id | int64 | Stable tracker id for the selected target, or `-1` when invalid. |
| target_u | float32 | Pixel x coordinate in the image, zero when invalid. |
| target_v | float32 | Pixel y coordinate in the image, zero when invalid. |
| target_norm_x | float32 | Normalized x coordinate: `target_u / image_width - 0.5`, zero when invalid. |
| target_norm_y | float32 | Normalized y coordinate: `target_v / image_height - 0.5`, zero when invalid. |
| image_width | uint32 | Image width in pixels. |
| image_height | uint32 | Image height in pixels. |
| confidence | float32 | Confidence in `[0.0, 1.0]`, zero when invalid. |
| reason | string | Selector reason for valid observations; invalid samples use `lost`, `stale`, or `disabled`. |
| stale_after_ms | uint32 | Receiver freshness limit for this observation. |

State semantics:

- `tracking`: a current attention target is available and target fields are valid.
- `lost`: a fresh frame was processed, but no usable attention target is available.
- `stale`: the source frame or server response has exceeded the freshness window.
- `disabled`: gaze target publishing is intentionally disabled by configuration or operator action.
