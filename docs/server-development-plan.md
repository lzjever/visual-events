# Visual Events Server 开发计划

日期：2026-06-26

## 1. 目标

本计划只覆盖 `visual-events-server` 的开发。当前阶段不开发正式机器人端 CLI。

Server 的目标是接收符合 [protocol.md](../common/schema/protocol.md) 的 JPEG 帧，完成 person pose 推理、追踪、事件规则和注视目标计算，并返回 10Hz `visual_state`。首个产品场景是商店门口揽客，V1 必须支持：

- `person_appeared`
- `person_left`
- `person_passing_by`
- `person_approaching_robot`
- `person_stopped_near_robot`
- `person_waving`
- `attention_target_changed`

`val-data/` 是本阶段端到端验证的强制数据源。任何 server 端 handoff 都必须用 `val-data/` 跑通协议、推理、追踪、事件、attention 和性能测试。`val-data/` 已在 `.gitignore` 中，不允许提交到 Git。缺失 `val-data/` 时 E2E 必须 fail，不能自动降级成 mock pass。

## 2. 非目标

本阶段不做：

- 不开发正式 robot CLI、DDS 输入、Botified frame 输出、头部控制。
- 不训练模型。
- 不做人脸识别、身份识别、长期记忆。
- 不做通用动作识别模型。
- 不做多摄像头融合。
- 不做 RK3588 正式移植；只保留 `InferBackend` 边界。
- 不做管理后台或可视化大屏。

允许开发一个简单测试工具，例如 `tools/replay_val_data.py`，用于把 `val-data/` JPEG 序列按协议发给 server 做端到端测试。这个工具不是产品 CLI，不接 DDS，不输出 Botified frame，不承担机器人本体职责。

开发原则：

- KISS：一个 server、一条 WebSocket 协议、一套 `visual_state` schema、一套事件规则。
- DRY：事件生成、cooldown、同 track 去重、attention target 选择只在 server 实现一次。
- YAGNI：不训练模型，不做人脸识别，不做 ReID，不做数据库，不做多摄像头，不做治理后台。
- 可验证优先：每个里程碑必须能用 `val-data/` 或 replay 工具验证。

## 3. 输入数据

当前 `val-data/` 目录结构：

| 场景目录 | 帧数 | 约时长 | 用途 |
| --- | ---: | ---: | --- |
| `pci_stand` | 74 | 7.54s | 站立/停留，验证 `person_stopped_near_robot` 和 attention 稳定性 |
| `pic_1_l_to_r` | 94 | 9.97s | 左到右路过，验证 `person_passing_by` |
| `pic_1_r_to_l` | 80 | 8.40s | 右到左路过，验证 `person_passing_by` 的方向无关性 |
| `pic_hello` | 52 | 5.83s | 打招呼/挥手，验证 `person_waving` |
| `pic_leave` | 113 | 11.98s | 离开，验证 `person_left` |
| `pic_persone_walk_in` | 79 | 8.39s | 走入/靠近，验证 `person_approaching_robot` |
| `pic_walk_in_stop` | 84 | 8.50s | 走入后停留，验证 `person_approaching_robot` 到 `person_stopped_near_robot` 的状态转换 |

数据约束：

- 文件是 1280x720 JPEG。
- 回放顺序按文件名排序。
- 文件名中的数字按纳秒时间戳解析，转成 `timestamp_ms`；无法解析时使用回放序号和默认 10Hz 时间。
- E2E 测试默认发送 `head_motion.state=stationary`。
- 必须额外跑一组 `head_motion.state=unknown`，验证运动敏感事件不会触发。

## 4. 架构

V1 技术栈：

- Python server。
- `uv` 做包管理和环境管理。
- FastAPI/Starlette WebSocket + Uvicorn。
- Pytest 做单元/集成测试。
- Ultralytics headless package `ultralytics-opencv-headless` + `YOLOv8n-pose` 做 V1 inference backend。
- 不同时维护 `websockets` 裸实现、gRPC 或第二套 HTTP streaming 协议。

V1 server 结构：

```text
src/
  visual_events_server/
    app.py
    config.py
    protocol.py
    processor.py
    inference/
      base.py
      ultralytics_pose.py
    tracking/
      byte_tracker.py
    attention/
      selector.py
    events/
      history.py
      engine.py
    metrics.py
tools/
  replay_val_data.py
  run_val_data_e2e.py
tests/
  unit/
  integration/
runtime/        # ignored: release venv, runtime cache, local config/model cache
artifacts/      # ignored: replay/e2e/perf outputs
```

模块职责：

| 模块 | 职责 |
| --- | --- |
| `protocol` | 解析 binary WebSocket frame，校验 header/JPEG，序列化 `visual_state` 和 error |
| `inference` | 加载 `YOLOv8n-pose`，输出项目内部 `PoseDetections` |
| `tracking` | 项目内 ByteTrack-style IoU/TTL tracker baseline，只追踪 person，输出稳定 `track_id`、速度、age、lost |
| `attention` | 选择最大稳定人物和 `target_uv` |
| `events` | 基于 track history 生成 V1 `semantic_events` |
| `metrics` | 输出 latency、FPS、事件统计、错误统计 |
| `tools/replay_val_data.py` | 测试回放客户端，不是产品 CLI |

环境约束：

- runtime package 只放在 `src/visual_events_server/`。
- `tools/` 和 `tests/` 是开发/验证目录，不属于正式 robot CLI。
- 开发环境使用项目内 `.venv/` 和 `.uv-cache/`。
- release/runtime 环境使用项目内 `runtime/venv/` 和 `runtime/cache/uv/`。
- inference runtime cache 必须收敛在 `runtime/cache/*`；server 设置 `YOLO_CONFIG_DIR`、`TORCH_HOME`、`XDG_CACHE_HOME`、`MPLCONFIGDIR`，不改 `HOME`。
- `val-data/`、`runtime/`、`artifacts/`、模型缓存和测试产物不得进入 Git。
- `uv.lock` 必须进入 Git，用于 release/runtime 的 `--frozen` 安装。

S0/S1 本地开发命令：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv sync --group dev
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --group dev pytest -q
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run visual-events-server --host 127.0.0.1 --port 8765
```

开发真实推理命令：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv sync --group dev --extra inference
```

release/runtime 最小部署命令：

```bash
UV_CACHE_DIR=runtime/cache/uv UV_PROJECT_ENVIRONMENT=runtime/venv uv sync --frozen --no-dev --no-editable --extra inference
runtime/venv/bin/visual-events-server --config runtime/config.toml
```

release 产物应保持简单：Python runtime venv、server 代码、锁定依赖、本地 config、后续模型缓存和运行输出都留在 `runtime/` 或 `artifacts/`，不写入 Git。

真实推理配置示例：

```toml
[server]
runtime_dir = "runtime"

[inference]
backend = "ultralytics"
model_path = "runtime/models/yolov8n-pose.pt"
device = "0"
imgsz = 640
conf = 0.25
```

GPU server 可省略 `device` 让 Ultralytics 自动选择，或显式写 `device = "0"`。CPU 仅作为本地无 GPU 调试 fallback，例如 `device = "cpu"`。

模型权重不进入 Git。默认模型路径是 `runtime/models/yolov8n-pose.pt`；也可用 `[inference].model_path` 显式指定。server 不调用裸 `YOLO("yolov8n-pose.pt")`，模型缺失时真实 backend 启动失败并报告 config error，不做隐式下载。

KISS 约束：

- 一个 WebSocket endpoint：`/v1/stream`。
- 一个运行入口：`visual-events-server --config <path>`。
- 一个 protocol schema：`common/schema/protocol.md`。
- 一个推理模型 baseline：`YOLOv8n-pose`。
- 一个 tracker baseline：项目内 ByteTrack-style IoU/TTL tracker baseline；不使用 Ultralytics `model.track()`，不依赖上游 ByteTrack package。
- 一个事件引擎：服务端负责 rising-edge、cooldown、同 track 去重。
- 不把 Ultralytics result 对象传到 tracking/events；进入 server 内部后统一转换为项目自己的结构。

## 5. 协议实现

Server 必须实现 [protocol.md](../common/schema/protocol.md)：

- binary request：`uint32_be header_len + header_json_utf8 + jpeg_bytes`
- JSON response：`visual_state` 或 `error`
- 每条连接一个 camera stream
- 第一帧确定连接的 `camera`；后续帧 `camera` 不同必须返回 `invalid_header`、`retryable=false`，并关闭连接
- 每条连接最多一个 in-flight frame
- 非法 header、非法 JPEG、超限 payload 返回 `error`
- 断线后丢弃该连接的 tracker/event state

Server 不直接读取 DDS，也不输出 Botified frame。

## 6. 事件规则

V1 事件由 `EventEngine` 生成。

通用规则：

- 每个 track 保留最近 2-3 秒 history。
- 全局 cooldown：5s。
- 同 track 同事件 cooldown 内只输出一次。
- 事件必须 rising-edge 触发。
- `event_id` 格式：`<camera>:evt_<monotonic_counter>`。

事件定义：

| 事件 | 规则 |
| --- | --- |
| `person_appeared` | 新 track 稳定出现至少 2 帧 |
| `person_left` | track lost 超过 TTL 后触发 |
| `person_passing_by` | 横向位移明显，从画面一侧通过，未进入近区停留 |
| `person_approaching_robot` | bbox 面积或高度持续增大，并向中心/近区移动至少 0.5s |
| `person_stopped_near_robot` | bbox 足够大且中心速度低，持续至少 1.5s |
| `person_waving` | 手腕在肩部附近或以上，并在 1-2s 内出现横向方向变化 |
| `attention_target_changed` | attention target 稳定切换 |

运动敏感事件：

- `person_passing_by`
- `person_approaching_robot`
- `person_stopped_near_robot`

这些事件只在 `head_motion.state=stationary` 时触发。`moving`、`unknown` 或缺失 `head_motion` 时不触发。

## 7. Attention

`AttentionSelector` 每帧选择一个目标：

```text
score = bbox_area_ratio * confidence * stability_score
```

规则：

- 优先选择最大稳定 person。
- 当前目标仍存在时保持。
- 新目标面积至少大于当前目标 25%，并持续 0.5s，才允许切换。
- 当前目标短暂丢失时保持 0.5-0.8s。
- 无稳定目标时 `attention=null`。

`target_uv` 优先级：

1. pose 头部关键点中心。
2. bbox fallback：`x=bbox_center_x`，`y=bbox_top + 0.28 * bbox_height`。

V1 server 只输出 `attention`，不生成头部控制命令。

## 8. 开发里程碑

### S0 Server Skeleton

产出：

- Python package skeleton。
- 配置加载。
- `/healthz` 或等价健康检查。
- `/v1/stream` WebSocket endpoint。
- protocol encode/decode。

验收：

- mock frame 能得到 mock `visual_state`。
- 非法 header/JPEG/过大 payload 返回 protocol error。

### S1 Val-data Replay Tool

产出：

- `tools/replay_val_data.py`
- 支持发送单个场景目录或全部 `val-data/`。
- 支持 `--fps 10`、`--head-motion stationary|moving|unknown`、`--save-jsonl <path>`。
- 按文件名时间戳排序读取 JPEG。
- 使用协议规定的 binary envelope。
- 每次只允许一个 in-flight frame。

验收：

- 能按协议回放 `val-data/*/*.jpeg`。
- 能保存每帧 `visual_state` 到 JSONL。
- 这个工具不依赖 DDS，不输出 Botified frame。

示例：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/replay_val_data.py \
  --server ws://127.0.0.1:8765/v1/stream \
  --data-dir val-data/pic_hello \
  --camera front \
  --fps 10 \
  --head-motion stationary \
  --save-jsonl artifacts/e2e/pic_hello/visual_state.jsonl
```

### S2 Inference

产出：

- `UltralyticsPoseBackend`。
- `PoseDetections` 内部结构。
- person bbox/keypoints/confidence 输出。
- `visual_state.scene_flags.has_person/person_count` 来自 detections。
- S2 不生成稳定 `track_id`，`tracks` 可以为空。

验收：

- `val-data/` 每个场景至少 95% 有效帧能完成推理并返回 `visual_state`。
- 有人场景中 `scene_flags.has_person/person_count` 能反映 person detections。
- server GPU 模式显存目标 < 4GB。
- 单帧错误不会中断连接。

### S3 Tracking

产出：

- 项目内 ByteTrack-style IoU/TTL tracker baseline。
- 每条 WebSocket 连接独立 tracker/history；inference backend/model 可以全局共享。
- Track history。
- 速度、age、lost_ms。
- `visual_state.tracks` 和 tracking-derived `scene_flags.has_person/person_count`；`attention` 仍为 `null`，`semantic_events` 仍为空。
- 低置信 detection 不创建新 track，但可更新已有 track；真实 ByteTrack-style rescue 要求 inference backend 的 `conf` 不高于 `tracking.low_conf`，否则低置信候选不会进入 tracker。

验收：

- `val-data/` 回放中可见 person track 不应频繁换 ID。
- 短暂漏检不删除、不重分配 track，并输出 `lost_ms > 0` 的 lost state；S3 不触发 `person_left`。
- `tools/replay_val_data.py` summary 输出 S3 tracking smoke 指标：`track_frames`、`duplicate_track_id_frames`、`single_visible_id_switches`、`adjacent_track_matches`、`association_id_switches`、`visible_counts_by_id`、`track_schema_errors`、`age_monotonic_violations`。
- `largest_bbox_track_switches`、`largest_bbox_track_id`、`largest_bbox_track_coverage`、`largest_bbox_track_max_gap_ms` 只作为 S4 attention/target 诊断，不作为 S3 tracking ID 稳定性 gate。

### S4 Attention

产出：

- 最大稳定人物选择。
- `target_uv`。
- 目标滞回和短暂丢失保持。

验收：

- `pci_stand`、`pic_walk_in_stop` 中 attention 稳定存在。
- 多帧中目标不因 bbox 小幅波动频繁切换。

### S5 Event Engine

产出：

- V1 七类事件。
- cooldown、rising-edge、同 track 去重。
- 运动敏感事件 gating。

验收：

- `pic_1_l_to_r`、`pic_1_r_to_l` 触发 `person_passing_by`。
- `pic_persone_walk_in` 触发 `person_approaching_robot`。
- `pic_walk_in_stop` 触发 `person_approaching_robot` 后可触发 `person_stopped_near_robot`。
- `pci_stand` 触发 `person_stopped_near_robot`。
- `pic_hello` 触发 `person_waving`。
- `pic_leave` 触发 `person_left`。
- `head_motion.state=unknown` 回放时不触发 `person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`。
- `head_motion.state=moving` 回放时同样不触发上述三类运动敏感事件。

### S6 E2E Gate

产出：

- `tools/run_val_data_e2e.py`
- E2E JSON report。
- 性能报告。

验收：

- 全量 `val-data/` 必须跑完。
- 生成 per-scene 事件摘要、latency 统计、错误帧统计。
- 失败时返回非零 exit code。

## 9. 测试计划

### Unit

- protocol parser。
- JPEG validation。
- geometry：bbox area、center、head fallback。
- attention hysteresis。
- event cooldown/rising-edge。
- motion-sensitive event gating。

### Integration

- WebSocket stream request/response。
- one-in-flight 行为。
- error response。
- reconnect 后 state 清理。
- inference -> tracking -> attention -> events。

### E2E with val-data

必须运行：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/run_val_data_e2e.py \
  --server ws://127.0.0.1:8765/v1/stream \
  --data-dir val-data \
  --out artifacts/e2e
```

必测场景：

- `val-data/pci_stand`
- `val-data/pic_1_l_to_r`
- `val-data/pic_1_r_to_l`
- `val-data/pic_hello`
- `val-data/pic_leave`
- `val-data/pic_persone_walk_in`
- `val-data/pic_walk_in_stop`

每个场景输出：

- `visual_state.jsonl`
- `summary.json`
- `metrics.json`
- `summary.md`

输出产物必须落在 `artifacts/` 下，不能写回 `val-data/`。

### Performance

统计范围：server 收到完整 frame 到发出 response。

目标：

- 输出频率：回放 10Hz 时 `visual_state` >= 9Hz。
- GPU server latency：P95 < 120ms，P99 < 200ms。
- 显存：< 4GB。
- 单场景连接不中断，error frame 比例 < 1%。
- 全量 `val-data/` 循环 5 分钟：无崩溃、无明显内存增长。

## 10. 验证矩阵

| 覆盖面 | 用例 | 数据 | 验收门槛 |
| --- | --- | --- | --- |
| 协议 | 合法 JPEG envelope | 任意 `val-data` 目录 | 每帧返回同 `frame_id` 的 `visual_state`，无协议错误 |
| 协议 | header 过大、非法 JSON、非 JPEG、unsupported encoding | 构造帧 | 返回协议定义的 `error.code`，服务不崩溃 |
| 协议 | one in-flight | replay client 强制等待响应 | server 不积压队列，乱序响应为 0 |
| 推理 | person bbox/keypoints | 全部 `val-data` | 有人场景中 `scene_flags.has_person` 非空率 >= 85%，`person_count` 来自 detections |
| 推理 | image_size/坐标合法性 | 全部 `val-data` | 内部 `PoseDetections` bbox 坐标在图像范围内，`bbox_area` 在图像面积范围内；S3 `tracks[].bbox_area_ratio` 合法 |
| 追踪 | 可见 person ID 稳定性 | `pci_stand`、`pic_walk_in_stop` | replay summary 中 `track_frames > 0`、`visible_counts_by_id` 非空、`single_visible_id_switches == 0`、`association_id_switches == 0`、`duplicate_track_id_frames == 0`、`track_schema_errors == 0`、`age_monotonic_violations == 0` |
| 追踪 | 路过轨迹连续 | `pic_1_l_to_r`、`pic_1_r_to_l` | 可见 person track 横向速度方向与场景一致；相邻 bbox 匹配 `adjacent_track_matches > 0` 且 `association_id_switches == 0` |
| 事件 | `person_passing_by` | `pic_1_l_to_r`、`pic_1_r_to_l` | 每段触发 1 次，不能触发 `person_stopped_near_robot` |
| 事件 | `person_approaching_robot` | `pic_persone_walk_in` | 触发 1 次，触发点应在 bbox 面积/高度连续增大之后 |
| 事件 | `person_stopped_near_robot` | `pic_walk_in_stop`、`pci_stand` | 稳定停留后触发 1 次；cooldown 内不重复 |
| 事件 | `person_waving` | `pic_hello` | 触发 1 次；其他非挥手目录不得频繁误报 |
| 事件 | `person_left` | `pic_leave` | track 丢失 TTL 后触发 1 次 |
| 事件抑制 | 运动敏感事件抑制 | 路过/靠近/停留目录，`head_motion=moving/unknown` | 不触发 `passing_by/approaching/stopped` |
| Attention | 最大稳定人物 | 全部 `val-data` | 存在稳定 visible person 或短暂 lost hold 时，`attention.target_track_id` 指向 selector 目标；无稳定目标且无 lost hold 时允许 `attention=null` |
| Attention | 注视点合法 | 全部 `val-data` | `target_uv` 在图像范围内 |
| 性能 | server GPU E2E | 全部 `val-data` 循环 5 分钟 | P95 < 120ms，P99 < 200ms |
| 回归 | 固定数据回放 | 全部 `val-data` | 事件类型、数量、顺序稳定；触发帧偏差 <= 3 帧或 <= 300ms |

## 11. E2E 验收矩阵

| val-data 场景 | 必须观察到 | 不应观察到 |
| --- | --- | --- |
| `pci_stand` | `person_appeared`, `person_stopped_near_robot`, stable `attention` | repeated event spam |
| `pic_1_l_to_r` | `person_passing_by` | `person_stopped_near_robot` |
| `pic_1_r_to_l` | `person_passing_by` | `person_stopped_near_robot` |
| `pic_hello` | `person_waving` | repeated `person_waving` within cooldown |
| `pic_leave` | `person_left` | premature `person_left` before lost TTL |
| `pic_persone_walk_in` | `person_approaching_robot` | `person_passing_by` |
| `pic_walk_in_stop` | `person_approaching_robot`, then `person_stopped_near_robot` | repeated cooldown spam |

第二轮 gating：

- 用同一批数据以 `--head-motion unknown` 回放。
- 不允许出现 `person_passing_by`、`person_approaching_robot`、`person_stopped_near_robot`。
- 仍允许 `person_appeared`、`person_left`、`person_waving`、`attention_target_changed`。

## 12. 测试产物格式

每次 E2E 只保留三类产物，避免过重治理：

原始响应：

```text
artifacts/e2e/<case>/visual_state.jsonl
```

每行一个 server 响应：

```json
{"frame_id":12,"latency_ms":43.2,"response":{"type":"visual_state"}}
```

场景汇总：

```text
artifacts/e2e/<case>/summary.json
```

示例字段：

```json
{
  "case": "pic_hello",
  "frames_sent": 52,
  "frames_ok": 52,
  "errors": 0,
  "hz": 9.8,
  "latency_ms": {"p50": 41, "p95": 88, "p99": 116},
  "events": [{"event": "person_waving", "count": 1}],
  "attention": {"valid_frames": 49, "invalid_uv": 0},
  "pass": true
}
```

性能报告：

```text
artifacts/perf/server_perf.json
```

必须包含 decode、preprocess、infer、postprocess、tracking、events、total latency、显存峰值。

## 13. Handoff 要求

Server handoff 必须包含：

- 可运行 `visual-events-server`。
- `tools/replay_val_data.py` 和 `tools/run_val_data_e2e.py`。
- 单元测试和集成测试。
- 全量 `val-data/` E2E 报告。
- 性能报告。
- 模型权重和授权说明。
- 已知失败场景和阈值说明。

不满足以下任一项，不可 handoff：

- 未跑 `val-data/` 全量 E2E。
- `val-data/` 缺失时用 mock 测试代替 E2E。
- `val-data/` 被加入 Git。
- 运动敏感事件在 `head_motion=unknown` 或 `moving` 时仍触发。
- server 输出不符合 [protocol.md](../common/schema/protocol.md)。
- server 内部事件规则依赖正式 robot CLI。
- 未输出 `summary.json` 和 `server_perf.json`。

## 14. 必须避免的歧义

- “server 端开发”不包含正式 robot CLI。
- replay client 是测试工具，不是产品 CLI。
- `val-data/` 必须用于 E2E 验证，但不得提交到 Git。
- 高频 `visual_state` 是 server 输出，不等于 Botified 事件。
- `semantic_events` 由 server 生成；未来 CLI 只做 `event_id` 幂等输出，不重新实现规则。
- `head_motion` 缺失等价于 `unknown`。
- `moving/unknown` 时不触发运动敏感事件。
- `approaching_robot` 是基于图像 bbox/运动趋势的近似，不代表真实 3D 距离估计。
- `passing_by` 是门店揽客语义，不要求世界坐标轨迹。
- `stopped_near_robot` 是基于大 bbox + 低速停留的近似，不承诺真实距离。
- `person_waving` 是 pose 规则，不是动作识别模型。
- V1 不做人脸识别，也不输出“看向机器人”的强判断。
- RK3588 是未来迁移方向，不是当前 server 开发验收条件。

## 15. 风险

| 风险 | 处理 |
| --- | --- |
| `val-data/` 没有人工标注 | V1 使用场景级期望和事件摘要验收；必要时后续增加轻量 annotation |
| pose 模型漏检导致事件漏报 | 先调阈值和 track TTL，不训练模型 |
| Tracker ID switch 影响事件 | 加 track 稳定帧数和 cooldown，避免一抖就发事件 |
| 头部运动状态缺失 | 按协议视为 `unknown`，禁用运动敏感事件 |
| 挥手规则误报 | 规则保守，要求关键点可见和短时间方向变化 |
| Ultralytics 授权 | 授权未确认前仅做内部 POC/性能验证 |
| RK3588 后续迁移 | 保持 `InferBackend`，不把 Ultralytics result 泄漏到业务层 |

## 16. 参考

- [产品设计](product-design.md)
- [开发与测试计划](development-test-plan.md)
- [协议草案](../common/schema/protocol.md)
