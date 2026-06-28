# Visual Events Protocol v1

日期：2026-06-26

本文档是 `visual-events-server` 的 wire protocol 事实来源。V1 只有一条主通道：WebSocket streaming。

S0-S8 当前实现范围是 server WebSocket 协议、mock/真实推理 `visual_state`、项目内 ByteTrack-style IoU/TTL tracker baseline、attention selector、semantic events、`val-data` E2E/perf/soak、runtime smoke 和 opt-in metrics evidence。本文中提到的 robot CLI、DDS、gaze target 和 Botified frame 行为是未来客户端侧约定，不是当前 server 实现范围。

DDS contract entrypoints:

- `common/schema/dds/camera_jpeg_contract.md`
- `common/schema/dds/gaze_target_v1.md`
- `common/schema/dds/head_state_v1.md`

## 1. 连接

- URL：`ws://<host>:<port>/v1/stream`
- 客户端：未来 `visual-events-cli`；S0/S1 可由 `tools/replay_val_data.py` 按同一 wire protocol 发送测试帧
- 服务端：`visual-events-server`
- 每条连接只处理一个 camera stream。
- 第一帧确定该连接的 `camera`；后续 frame 的 `camera` 不同是严重协议错误，服务端返回 `invalid_header`、`retryable=false` 后关闭连接。
- 每条连接最多一个 in-flight frame。

Backpressure：

1. CLI 从 DDS 持续接收 JPEG，但本地只保留最新有效帧。
2. CLI 发送一帧后等待同 `frame_id` 的 `visual_state` 或 timeout。
3. 等待期间到达的新 DDS 帧只替换本地 latest，不进入 WebSocket 队列。
4. timeout 后 CLI 关闭连接并重连。
5. `gaze_target.stale_ms` watchdog 独立运行；即使 one in-flight frame 还没有到 `response_timeout_ms`，最近有效 gaze target 到达 stale deadline 时也必须发布一次 `valid=false,state=stale`。

## 2. 客户端帧消息

客户端到服务端使用 binary WebSocket message：

```text
uint32_be header_len
header_json_utf8
jpeg_bytes
```

限制：

- `header_len` 最大 16 KiB。
- `jpeg_bytes` 最大 2 MiB。
- `encoding` 只支持 `jpeg`。
- 非法 message 触发 error response；严重协议错误后服务端关闭连接。

Header 必填字段：

```json
{
  "type": "frame",
  "schema_version": 1,
  "camera": "front",
  "frame_id": 1024,
  "timestamp_ms": 1710000000000,
  "encoding": "jpeg",
  "width": 1280,
  "height": 720
}
```

`frame_id` 是 CLI 生成的 per-connection monotonic transport identity。`CameraFrame_` DDS 输入没有源 `frame_id`，CLI 不得使用 DDS `timestamp_ns`/`timestamp_ms` 作为 identity；CLI 将 DDS source timestamp（缺失或不可用时使用 receive fallback）填入 WebSocket header `timestamp_ms`，server 原样回显为 `visual_state.frame_timestamp_ms`。这个 timestamp 只用于时序/freshness，不是 frame identity。CLI 重连后可以从新的 monotonic sequence 开始，服务端状态仍按 WebSocket connection 隔离。

Header 可选字段：

```json
{
  "head_motion": {
    "state": "stationary",
    "yaw_vel_rad_s": 0.0,
    "pitch_vel_rad_s": 0.0
  }
}
```

`head_motion.state` 取值：

- `stationary`: 当前头部近似静止。
- `moving`: 当前头部正在运动。
- `unknown`: 无法判断。

V1 规则：`moving` 或 `unknown` 时，服务端暂停 `person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot` 这类运动敏感事件。没有 `head_motion` 字段等价于 `unknown`。

## 3. 服务端状态消息

服务端到客户端使用 JSON text WebSocket message。

正常响应：

```json
{
  "type": "visual_state",
  "schema_version": 1,
  "camera": "front",
  "frame_id": 1024,
  "frame_timestamp_ms": 1710000000000,
  "server_timestamp_ms": 1710000000082,
  "image_size": [1280, 720],
  "tracks": [
    {
      "track_id": 7,
      "class": "person",
      "bbox_xyxy": [320, 120, 520, 600],
      "bbox_area_ratio": 0.18,
      "center_uv": [420, 360],
      "head_uv": [421, 205],
      "velocity_uv_s": [35, -4],
      "age_ms": 2400,
      "lost_ms": 0,
      "confidence": 0.86,
      "pose_confidence": 0.72
    }
  ],
  "attention": {
    "target_track_id": 7,
    "target_uv": [421, 205],
    "reason": "largest_stable_person",
    "confidence": 0.86
  },
  "scene_context": {
    "engagement_state": "available",
    "attention_available": true,
    "target_track_id": 7,
    "no_engage_reasons": [],
    "target_reacquired": null
  },
  "scene_flags": {
    "has_person": true,
    "person_count": 1,
    "largest_person_stable": true,
    "someone_near_center": true
  },
  "semantic_events": []
}
```

无人或无稳定目标时：

```json
{
  "type": "visual_state",
  "schema_version": 1,
  "camera": "front",
  "frame_id": 1025,
  "frame_timestamp_ms": 1710000000100,
  "server_timestamp_ms": 1710000000180,
  "image_size": [1280, 720],
  "tracks": [],
  "attention": null,
  "scene_context": {
    "engagement_state": "no_target",
    "attention_available": false,
    "target_track_id": null,
    "no_engage_reasons": ["no_visible_person"],
    "target_reacquired": null
  },
  "scene_flags": {
    "has_person": false,
    "person_count": 0,
    "largest_person_stable": false,
    "someone_near_center": false
  },
  "semantic_events": []
}
```

Track 字段规则：

- `tracks` 是必填数组；S3 只包含 `class="person"` 的 track。
- `tracks` 可以包含当前帧未匹配但仍在 lost TTL 内保留的 track。这类 track 的 `lost_ms > 0`，`bbox_xyxy`、`center_uv`、`head_uv`、`confidence`、`pose_confidence` 使用最近一次有效观测值；超过 TTL 后从数组移除。
- `scene_flags.has_person` 和 `scene_flags.person_count` 只统计当前帧匹配/可见的 track，即 `lost_ms == 0`。lost track 不计入人数。
- `scene_context` 是必填 object，由 server `EventEngine` 随 `semantic_events` 一起输出，processor 只透传，不复算 engagement 业务规则。最小字段为 `engagement_state`、`attention_available`、`target_track_id`、`no_engage_reasons`、`target_reacquired`。
- 当前最小 engagement 规则：无人时输出 `engagement_state="no_target"`、`attention_available=false`、`target_track_id=null`、`no_engage_reasons=["no_visible_person"]`、`target_reacquired=null`；有稳定、近距离、头部静止且非 fast passing 的 attention target 时输出 `engagement_state="available"`、`attention_available=true`、`target_track_id=<attention target id>`、`no_engage_reasons=[]`。有可见人但暂不可 engage 时输出 `engagement_state="no_engage_target"`，`no_engage_reasons` 可包含 `unstable`、`too_far`、`camera_motion_not_stationary`、`passing_fast`。
- `scene_context.target_reacquired` 为 `null` 或一帧 reacquire object。object 只包含 `runtime_person_slot`、`reacquired_from_track_id`、`reacquired_to_track_id`、`reacquire_elapsed_ms`、`reacquire_center_distance_px`、`reacquire_area_ratio`。它是当前 server runtime 的短期 evidence，一帧 pulse，不代表长期身份，不跨 WebSocket 连接或 server 重启保留。
- S5 可输出 `semantic_events`；无事件帧必须输出空数组 `[]`。
- 每个 track object 必须包含：`track_id` integer、`class` string、`bbox_xyxy` number[4]、`bbox_area_ratio` number、`center_uv` number[2]、`head_uv` number[2]、`velocity_uv_s` number[2]、`age_ms` integer、`lost_ms` integer、`confidence` number、`pose_confidence` number。
- `pose_confidence` 是 keypoint confidence 的平均值；没有 keypoint 或没有 confidence 时为 `0.0`。
- pose keypoints 可在 server 内部随 `TrackSnapshot` 保留，用于 `person_waving`；wire protocol 的 `tracks[]` 不输出 keypoints。
- `head_uv` 优先使用 COCO nose/eyes/ears 中有效点的中心；没有有效 face keypoint 时 fallback 到 bbox 水平中心和 `bbox_top + 0.28 * bbox_height`。
- `velocity_uv_s` 单位是像素/秒，来自同一 track 的近期有效观测；历史不足或时间间隔过小时可输出 `[0.0, 0.0]` 或保留上一速度，避免尖峰。
- `age_ms`、`lost_ms` 不得为负数。服务端发现 `timestamp_ms` 或 `frame_id` 倒退时应 reset 当前连接的 tracker。

Attention 字段规则：

- `attention` 为 `null`，或包含 `target_track_id` integer、`target_uv` number[2]、`reason` string、`confidence` number。
- `target_track_id` 必须引用当前 `tracks` 数组中的目标；短暂 lost hold 期间可引用 `lost_ms > 0` 且仍在 `tracks` 中的 lost track。
- `target_uv` 使用图像像素坐标，必须是有限数，并位于 `[0,width] x [0,height]` 范围内。
- `confidence` 表示 selector 对当前目标的置信度；当前实现使用目标 track 的检测置信度并 clamp 到 `[0,1]`。
- `scene_flags.largest_person_stable=true` 仅表示当前 attention 对应 visible stable target；lost hold 期间为 `false`。

## 4. 坐标与时间

- 像素坐标使用输入图像坐标系。
- 原点在左上角。
- `x` 向右增大，`y` 向下增大。
- bbox 使用 `[x1, y1, x2, y2]`，`x2/y2` 表示右下边界；宽高按 `x2 - x1`、`y2 - y1` 计算，端到端实现统一转 float。
- `center_uv`、`head_uv`、`target_uv` 使用 `[x, y]` 像素坐标。
- `bbox_area_ratio = bbox_area / image_area`。
- 时间单位全部为毫秒，字段名以 `_ms` 结尾。
- 速度 `velocity_uv_s` 使用像素/秒。

## 5. Semantic Event

`semantic_events` 是 `visual_state` 的数组字段。

```json
{
  "type": "semantic_event",
  "event_id": "front:evt_000456",
  "event": "person_waving",
  "camera": "front",
  "track_id": 7,
  "confidence": 0.86,
  "duration_ms": 900,
  "lifecycle_state": "confirmed",
  "evidence": {
    "runtime_person_slot": 3,
    "wrist_x_span_px": 84.0,
    "wrist_x_span_bbox_ratio": 0.42,
    "wrist_y_relative_to_shoulder_px": 18.0,
    "wave_duration_ms": 900,
    "keypoint_min_confidence": 0.72
  },
  "text": "有人在机器人前方挥手"
}
```

V1 event 枚举：

- `person_appeared`
- `person_left`
- `person_passing_by`
- `person_approaching_robot`
- `person_stopped_near_robot`
- `person_waving`
- `attention_target_changed`

运动敏感事件：

- `person_passing_by`
- `person_approaching_robot`
- `person_stopped_near_robot`

这些事件只在 `head_motion.state=stationary` 时触发。`moving`、`unknown` 或缺失 `head_motion` 时，服务端不累积这些运动敏感事件的条件。

`person_appeared` 只对 salient target 触发：优先使用当前 `attention.target_track_id`；没有 attention 时，从 visible stable tracks 中选择面积和置信度最高的 person。它不会对背景中所有 person 逐个刷屏。

所有对外输出的 semantic event 都是 confirmed 事实，必须带 `lifecycle_state: "confirmed"` 和 `evidence`。`evidence` 只包含 JSON 标量或简单 list/dict；数值必须是 finite，不输出 `NaN` 或 `Infinity`。

当 semantic event 来自 alias 窗口内的 reacquired track 时，`evidence` 可选包含同一组 reacquire keys：`runtime_person_slot`、`reacquired_from_track_id`、`reacquired_to_track_id`、`reacquire_elapsed_ms`、`reacquire_center_distance_px`、`reacquire_area_ratio`。这些 key 是可选短期 runtime evidence，不加入下面的 event-specific required evidence table。

v0.2 event-specific required evidence keys：

| 事件 | required evidence keys |
| --- | --- |
| `person_appeared` | `runtime_person_slot`、`visible_duration_ms`、`bbox_area_ratio`、`salient_reason` |
| `person_left` | `runtime_person_slot`、`lost_duration_ms`、`last_bbox_area_ratio` |
| `person_passing_by` | `runtime_person_slot`、`dx_ratio`、`avg_vx_px_s`、`crossed_side_bands`、`camera_motion_state`、`passing_speed_class` |
| `person_approaching_robot` | `runtime_person_slot`、`bbox_area_ratio_start`、`bbox_area_ratio_end`、`area_growth_ratio`、`area_delta`、`camera_motion_state` |
| `person_stopped_near_robot` | `runtime_person_slot`、`bbox_area_ratio`、`speed_px_s_p95`、`stationary_duration_ms`、`camera_motion_state` |
| `person_waving` | `runtime_person_slot`、`wrist_x_span_px`、`wrist_x_span_bbox_ratio`、`wrist_y_relative_to_shoulder_px`、`wave_duration_ms`、`keypoint_min_confidence` |
| `attention_target_changed` | `previous_track_id`、`target_track_id`、`switch_reason` |

`runtime_person_slot` 只在当前 server runtime 内有效，用于短期合并和调试；不代表跨连接身份。

服务端负责：

- rising-edge 触发。
- 同类事件全局 5s cooldown。
- 同 track 同事件 5s 去重。
- 同帧可输出多个不同事件，按固定顺序：`person_appeared`、`person_left`、`person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`、`person_waving`、`attention_target_changed`。
- 生成稳定 `event_id`。

CLI 只负责：

- 做 Botified notification gate：allowlist、`event_id` 幂等、pending/coalescing、same-key gap、global/burst limit、低价值事件静默。
- 做 stdout writer 背压处理。
- 不重新实现视觉规则，不根据 attention 高频状态生成 Botified 事件；只消费 server 输出的 event type、event evidence、`scene_context`、`track_id`/`event_id`。

Botified 通知频率由 server semantic event engine 的 rising-edge、cooldown、dedupe 与 CLI notification gate 共同约束，并由 PC/现场 report gate 验收。

Botified stdout allowlist：

- `person_appeared`
- `person_left`
- `person_passing_by`
- `person_approaching_robot`
- `person_stopped_near_robot`
- `person_waving`

`attention_target_changed` 只保留在 `visual_state.semantic_events` 和诊断 artifact 中，用于调试/PC gate，不输出到 Botified stdout，避免把高频注视闭环泄漏到 Botified。

## 6. Error Message

服务端可返回 JSON error：

```json
{
  "type": "error",
  "schema_version": 1,
  "frame_id": 1024,
  "code": "invalid_frame",
  "message": "jpeg payload is invalid",
  "retryable": true
}
```

V1 error code：

- `invalid_header`
- `invalid_frame`
- `frame_too_large`
- `unsupported_encoding`
- `backend_unavailable`
- `internal_error`

CLI 行为：

- `retryable=true`: 丢弃当前帧，继续下一帧。
- `retryable=false`: 关闭连接并按重连策略恢复。

## 7. 断线与状态清理

- WebSocket 断开后，服务端丢弃该连接的 tracker 和 event state。
- CLI 重连后从新的 `frame_id` 继续发送。
- CLI 重连后的前 1s 内不输出 Botified frame，避免旧状态造成事件突发。
- 断线期间 CLI 不使用过期 `attention` 发布有效 gaze target；必须在 250ms 内发布一次 invalid/stale sample，DDS lifespan 只作为后备失效保护。
- `gaze_target.stale_ms` 与 `service.response_timeout_ms` 是两个计时器。前者决定下游何时必须看到 stale，后者决定 WebSocket request 何时失败重连；不能因为仍有一个 in-flight request 未 timeout 而延迟 stale sample。

## 8. Botified Frame 输出

本节描述未来 robot CLI 的客户端输出约定；`visual-events-server` 不直接输出 Botified frame。

CLI stdout 默认只输出 Botified allowlist frame：

```text
<botified>{"id":"visual:front:evt_000456","urgency":"normal","timeout_secs":8,"request":"event=person_waving camera=front track_id=7 confidence=0.86 duration_ms=900 text=有人在机器人前方挥手 visual_context={\"visual_context\":{\"event_target\":{\"track_id\":7,\"runtime_person_slot\":3,\"visible_now\":true,\"matches_attention_target\":true,\"event_age_ms\":82,\"position\":\"center\",\"size\":\"mid\",\"bbox_area_ratio\":0.1042},\"trigger_evidence\":{\"runtime_person_slot\":3,\"wave_duration_ms\":900},\"current_scene\":{\"camera\":\"front\",\"frame_age_ms\":82,\"person_count\":1,\"attention_target\":{\"track_id\":7,\"position\":\"center\",\"size\":\"mid\",\"center_uv\":[420.0,360.0],\"bbox_area_ratio\":0.1042},\"other_people_count\":0,\"engagement_state\":\"available\",\"no_engage_reasons\":[]}}}","expect":"ack"}</botified>
```

顶层 payload 只包含 `id`、`urgency`、`timeout_secs`、`request`、`expect`。`request` 中包含 `event=... track_id=... visual_context=<minified-json>`。`visual_context` 是紧凑 JSON wrapper，包含 `event_target`、`trigger_evidence`、`current_scene`；不包含图片、crop、embedding、身份或全量 tracks。

日志、调试状态、性能指标走 stderr 或文件。`--debug-json-stdout` 只能用于手工调试，不能用于 Botified task。

stdout 写入不得阻塞 gaze stale watchdog。CLI 必须使用 bounded queue/drop/coalescing，并固定 BrokenPipe 行为：尽力发布一次 `valid=false,state=stale` 后受控非 0 退出；stdout 慢、pipe close 或 Botified 未读取时，CLI 不能无界排队，也不能延迟 `valid=false,state=stale` gaze target。
