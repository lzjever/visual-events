# visual-events

视觉事件推理服务。当前 repo 已包含 S0-S8 server baseline：WebSocket wire protocol parser/serializer、mock `visual_state` endpoint、`val-data` replay/E2E 工具、S2 推理 backend 边界和 Ultralytics pose adapter、S3 项目内 ByteTrack-style IoU/TTL tracker baseline、S4 attention selector、S5 semantic events、支持 S6/S6.1/S6.3/S8 E2E/perf/soak、semantic event first-trigger/timeline gate 和 opt-in server metrics evidence 的 `tools/run_val_data_e2e.py`，以及 release/runtime smoke verification 工具 `tools/run_runtime_smoke.py`。首个产品场景是商店门口揽客机器人。

当前 GA acceptance/pass/fail authority 是 PC 本地模拟：synthetic DDS image/head-state publishers、runtime server/CLI、DDS gaze subscriber/stdout collector、`val-data` full PC E2E，以及必要轻量稳定性和 latency checks。manifest/evidence/strict gate 等已有工作只作为有限证据保留。真机实际运行、真实 robot camera DDS、真实 head-state source、physical head pointing、HIL/real closed loop、现场测试或 owner sign-off 不阻塞 GA；RK3588/board/real robot/field validation 属于 GA 之后的硬件适配/现场验证，PC evidence 只能声称 PC-simulated GA passed。

本 repo 计划包含两个运行单元，并包含开发/验证工具：

- `visual-events-server`: 已有 S0-S8 baseline。局域网推理服务，接收机器人侧 JPEG 帧，输出 10Hz `visual_state` 和低频 `semantic_events`；metrics 默认关闭，显式配置后写 ignored JSONL。
- `visual-events-cli`: 未来运行单元。Botified 启动的机器人后台 CLI，从 DDS 抓取图像，调用服务端，把高频注视目标发布为 DDS gaze target，并把低频语义事件转换成 Botified frame。CLI 不直接操纵运控。
- `tools/replay_val_data.py`: 已有开发/验证工具。按 server wire protocol 回放 `val-data` JPEG，不接 DDS，不输出 Botified frame。
- `tools/run_val_data_e2e.py`: 已有 S6/S8 E2E/perf gate，支持 S6.1 5-minute soak evidence gate、S6.3 semantic event first-trigger/timeline gate 和 opt-in server metrics JSONL aggregation。对运行中的 server 回放 `stationary` 全量、`unknown` 全量 suppression、`moving` targeted suppression gates，输出 ignored artifacts。
- `tools/run_runtime_smoke.py`: 已有 release/runtime verification 工具。它会同步 `runtime/venv`、启动 release/runtime server，并通过 `/healthz` 校验新进程身份；它不是产品 CLI，也不能替代 `val-data` E2E/soak。

核心分层：

```text
DDS JPEG @10Hz
  -> robot CLI
  -> WebSocket stream over LAN
  -> inference service
  -> visual_state / attention @10Hz
  -> robot CLI
      -> DDS gaze target publisher
      -> Botified semantic event frames
  -> head/motion owner subscribes gaze target
```

设计文档：

- [产品设计](docs/product-design.md)
- [开发与测试计划](docs/development-test-plan.md)
- [Server 开发计划](docs/server-development-plan.md)
- [Server handoff](docs/server-handoff.md)
- [GA 后续开发计划](docs/ga-development-plan.md)
- [协议草案](common/schema/protocol.md)
- DDS contracts: [camera JPEG](common/schema/dds/camera_jpeg_contract.md), [gaze target v1](common/schema/dds/gaze_target_v1.md), [head state v1](common/schema/dds/head_state_v1.md)

当前基线决策：

- 包管理：`uv`。
- runtime package：`src/visual_events_server/`。
- 开发/验证目录：`tools/`、`tests/`。
- 输入/输出频率：10Hz。
- 模型：`YOLOv8n-pose` 作为 V1 baseline。
- 真实推理依赖：开发使用 `uv sync --group dev --extra inference`；release 使用 `uv sync --frozen --no-dev --no-editable --extra inference --reinstall-package visual-events-server`，确保交付验证时当前项目 wheel 刷新到 `runtime/venv`。当前 extra 使用 PyTorch cu128 wheel，面向 5090D GPU server；当前 handoff 验证配置路径为 `runtime/config/s2.toml`。
- Inference release note：模型权重默认 `runtime/models/yolov8n-pose.pt`，不入 Git，必须由 release/runtime 外部准备；真实 backend 只加载显式 `model_path`，缺失时启动失败而不会隐式下载。
- 追踪：项目内 ByteTrack-style IoU/TTL tracker baseline，不使用 Ultralytics `model.track()`。
- 高频通道：WebSocket streaming + CLI 发布 DDS gaze target。
- 低频语义事件：Botified `<botified>...</botified>` request frame。
- V1 事件：出现、离开、路过、靠近、停留、挥手、注视目标变化。
- S6.3 event gate：在 `val-data/` 上检查 expected first trigger frame tolerance <= 3 frames、forbidden scene events，以及 `pic_walk_in_stop` 的 approaching-before-stopped ordering；它不是 dense per-frame manual annotation。
- S8 metrics evidence：server metrics 默认关闭；通过 `[metrics].jsonl_path` 或 `--metrics-jsonl <path>` 开启后写 ignored JSONL，E2E runner 只在显式 `--server-metrics-jsonl <path>` 时聚合 phase latency、RSS 和 PyTorch CUDA allocated/reserved VRAM evidence。
- RK3588 兼容：从第一版开始保留 `InferBackend` 边界，未来替换为 RKNN backend。

开发原则：

- KISS：一个输入源、一个服务协议、一个高频状态 schema、一个 DDS gaze target 输出、一个低频事件出口。
- DRY：schema、几何计算、事件冷却和注视目标选择只实现一次。
- YAGNI：不训练模型，不做通用动作识别，不做人脸识别，不做多摄像头融合，不做后台治理平台。
- 治理克制：只在直接保护核心运行边界或用户明确要求时添加 report/audit/gate；TDD 只覆盖核心功能和高风险集成，不为测试工具、报告骨架、文档文字堆测试。
- 低频事件交给 agent 决策；高频注视 target 由 CLI 发布 DDS，真实头部动作由运控/头控 owner 本地闭环完成。
