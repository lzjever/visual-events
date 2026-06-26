# visual-events

视觉事件推理服务。当前 repo 已包含 S0-S6 server baseline：WebSocket wire protocol parser/serializer、mock `visual_state` endpoint、`val-data` replay/E2E 工具、S2 推理 backend 边界和 Ultralytics pose adapter、S3 项目内 ByteTrack-style IoU/TTL tracker baseline、S4 attention selector、S5 semantic events，以及 S6 `tools/run_val_data_e2e.py` E2E/perf gate。首个产品场景是商店门口揽客机器人。

本 repo 计划包含两个运行单元，并包含开发/验证工具：

- `visual-events-server`: 已有 S0-S6 baseline。局域网推理服务，接收机器人侧 JPEG 帧，输出 10Hz `visual_state` 和低频 `semantic_events`。
- `visual-events-cli`: 未来运行单元。Botified 启动的机器人后台 CLI，从 DDS 抓取图像，调用服务端，消费高频注视目标，并把低频语义事件转换成 Botified frame。
- `tools/replay_val_data.py`: 已有开发/验证工具。按 server wire protocol 回放 `val-data` JPEG，不接 DDS，不输出 Botified frame。
- `tools/run_val_data_e2e.py`: 已有 S6 E2E/perf gate。对运行中的 server 回放全量 `val-data`，输出 ignored artifacts。

核心分层：

```text
DDS JPEG @10Hz
  -> robot CLI
  -> WebSocket stream over LAN
  -> inference service
  -> visual_state / attention @10Hz
  -> robot CLI
      -> gaze controller
      -> Botified semantic event frames
```

设计文档：

- [产品设计](docs/product-design.md)
- [开发与测试计划](docs/development-test-plan.md)
- [Server 开发计划](docs/server-development-plan.md)
- [Server handoff](docs/server-handoff.md)
- [协议草案](common/schema/protocol.md)

当前基线决策：

- 包管理：`uv`。
- runtime package：`src/visual_events_server/`。
- 开发/验证目录：`tools/`、`tests/`。
- 输入/输出频率：10Hz。
- 模型：`YOLOv8n-pose` 作为 V1 baseline。
- 真实推理依赖：开发使用 `uv sync --group dev --extra inference`；release 使用 `uv sync --frozen --no-dev --no-editable --extra inference`。当前 extra 使用 PyTorch cu128 wheel，面向 5090D GPU server。
- Inference release note：模型权重默认 `runtime/models/yolov8n-pose.pt`，不入 Git，必须由 release/runtime 外部准备；真实 backend 只加载显式 `model_path`，缺失时启动失败而不会隐式下载。
- 追踪：项目内 ByteTrack-style IoU/TTL tracker baseline，不使用 Ultralytics `model.track()`。
- 高频通道：WebSocket streaming。
- 低频语义事件：Botified `<botified>...</botified>` request frame。
- V1 事件：出现、离开、路过、靠近、停留、挥手、注视目标变化。
- RK3588 兼容：从第一版开始保留 `InferBackend` 边界，未来替换为 RKNN backend。

开发原则：

- KISS：一个输入源、一个服务协议、一个高频状态 schema、一个低频事件出口。
- DRY：schema、几何计算、事件冷却和注视目标选择只实现一次。
- YAGNI：不训练模型，不做通用动作识别，不做人脸识别，不做多摄像头融合，不做后台治理平台。
- 低频事件交给 agent 决策；高频注视控制在机器人本体本地闭环完成。
