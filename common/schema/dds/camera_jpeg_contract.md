# Camera JPEG DDS Contract

| key | value |
| --- | --- |
| topic | /camera/image/jpeg |
| dds_type | unitree_camera::msg::dds_::CameraFrame_ |
| encoding | JPEG |
| reliability | best_effort |
| durability | volatile |
| history | keep_last_1 |
| deadline_ms | 150 |
| lifespan_ms | 300 |
| liveliness_lease_ms | 1000 |

## Payload

`CameraFrame_` is the source JPEG frame contract consumed by Visual Events.

| field | type | notes |
| --- | --- | --- |
| timestamp_ns | unsigned long long | Source capture timestamp in nanoseconds when trustworthy. If absent, zero, or non-monotonic, receivers fall back to receive wall-clock time. |
| camera_name | string | Source camera identifier. Unitree commonly publishes `image`; receivers map this source name to a configured logical camera. |
| width | unsigned long | Source image width in pixels after JPEG decode. Frames with missing, zero, or mismatched dimensions are invalid. |
| height | unsigned long | Source image height in pixels after JPEG decode. Frames with missing, zero, or mismatched dimensions are invalid. |
| encoding | string | Expected to be exactly `JPEG`. Other encodings are invalid for this DDS contract. |
| step | unsigned long | Encoded row stride value from the source message. Receivers treat inconsistent or unusable values as invalid-frame evidence rather than deriving identity from it. |
| data | sequence&lt;octet&gt; | JPEG-encoded bytes. Empty or undecodable payloads are invalid frames. |

## Receiver Semantics

`CameraFrame_` does not contain a `frame_id`; receivers must not invent source frame identity from this DDS payload. Receivers maintain invalid-frame counters for missing fields, repeated or non-monotonic timestamps, unsupported encodings, malformed dimensions, empty payloads, or undecodable JPEG bytes.

Receivers keep only the latest frame per logical camera and drop older frames. Staleness is measured by receiver monotonic time, not source wall-clock time, so clock jumps do not make frames fresh. The target QoS is best effort, volatile, keep-last depth 1 with a 150 ms deadline, 300 ms lifespan, and 1000 ms liveliness lease.
