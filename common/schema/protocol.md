# Visual Events Protocol v1

日期：2026-06-26

本文档是 `visual-events-server` 的 wire protocol 事实来源。V1 只有一条主通道：WebSocket streaming。

S0-S3 当前实现范围是 server WebSocket 协议、mock/推理 `visual_state`、项目内 ByteTrack-style IoU/TTL tracker baseline 和 `tools/replay_val_data.py` 回放工具。本文中提到的 robot CLI、DDS、gaze controller 和 Botified frame 行为是未来客户端侧约定，不是当前 server 实现范围。

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
    "confidence": 0.82
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
- S3 不输出 attention 或 semantic events：`attention` 为 `null`，`semantic_events` 为空数组。后续 S4/S5 才填充这些字段。
- 每个 track object 必须包含：`track_id` integer、`class` string、`bbox_xyxy` number[4]、`bbox_area_ratio` number、`center_uv` number[2]、`head_uv` number[2]、`velocity_uv_s` number[2]、`age_ms` integer、`lost_ms` integer、`confidence` number、`pose_confidence` number。
- `pose_confidence` 是 keypoint confidence 的平均值；没有 keypoint 或没有 confidence 时为 `0.0`。
- `head_uv` 优先使用 COCO nose/eyes/ears 中有效点的中心；没有有效 face keypoint 时 fallback 到 bbox 水平中心和 `bbox_top + 0.28 * bbox_height`。
- `velocity_uv_s` 单位是像素/秒，来自同一 track 的近期有效观测；历史不足或时间间隔过小时可输出 `[0.0, 0.0]` 或保留上一速度，避免尖峰。
- `age_ms`、`lost_ms` 不得为负数。服务端发现 `timestamp_ms` 或 `frame_id` 倒退时应 reset 当前连接的 tracker。

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

这些事件只在 `head_motion.state=stationary` 时触发。

服务端负责：

- rising-edge 触发。
- 全局 5s cooldown。
- 同 track 同事件去重。
- 生成稳定 `event_id`。

CLI 只负责：

- 按 `event_id` 做 Botified 输出幂等保护。
- 把事件事实转换为 Botified request frame。

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
- 断线期间 gaze controller 使用保持或回中策略，不使用过期 `attention`。

## 8. Botified Frame 输出

本节描述未来 robot CLI 的客户端输出约定；`visual-events-server` 不直接输出 Botified frame。

CLI stdout 默认只输出 Botified frame：

```text
<botified>{"id":"visual:front:evt_000456","urgency":"normal","timeout_secs":8,"request":"视觉事件：有人在机器人前方挥手。track_id=7, confidence=0.86。请根据当前上下文决定是否回应；处理完成后回复 ack。","expect":"ack"}</botified>
```

日志、调试状态、性能指标走 stderr 或文件。`--debug-json-stdout` 只能用于手工调试，不能用于 Botified task。
