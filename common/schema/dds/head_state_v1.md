# Head State V1 DDS Contract

| key | value |
| --- | --- |
| topic | /robot/head_state |
| dds_type | visual_events::msg::dds_::HeadStateV1_ |
| reliability | best_effort |
| durability | volatile |
| history | keep_last_1 |
| deadline_ms | 150 |
| lifespan_ms | 250 |
| liveliness_lease_ms | 500 |

## Semantics

`HeadStateV1_` is a read-only observation of current head state. It does not request or imply robot motion.

| field | type | notes |
| --- | --- | --- |
| schema_version | uint32 | Contract version, currently `1`. |
| timestamp_ms | int64 | Measurement timestamp in milliseconds. |
| valid | bool | `false` when the state is stale, unavailable, or contains non-finite values. |
| yaw_rad | float64 | Head yaw angle in radians. Unknown values are represented with `valid=false`. |
| pitch_rad | float64 | Head pitch angle in radians. Unknown values are represented with `valid=false`. |
| yaw_vel_rad_s | float64 | Yaw angular rate in radians per second. |
| pitch_vel_rad_s | float64 | Pitch angular rate in radians per second. |

Consumers must treat stale or non-finite values as unknown. The current GA acceptance gate uses PC-simulated `HeadStateV1_` input; real robot/RK/field/release validation must prove the same contract separately as post-GA validation and cannot be inferred from PC evidence.
