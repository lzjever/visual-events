# visual-events

视觉事件推理服务。服务端接收机器人侧 JPEG 帧，输出 10Hz `visual_state`、注意力目标和低频 `semantic_events`；CLI 把服务端结果投影到机器人侧 gaze target 和 Botified 事件上下文。首个产品场景是商店门口揽客机器人。

## Demo

用户可见 demo 只使用两个入口：

```bash
uv run --extra inference python tools/run_visual_demo.py --data-dir val-data
uv run --extra inference python tools/run_memory_demo.py --data-dir val-data
```

打开 `artifacts/demo/visual/index.html` 查看 visual demo；打开 `artifacts/demo/memory/index.html` 查看 memory demo。两个 demo 默认使用真实模型，报告只描述本次 demo 事实，避免外推到其他运行环境。

## Runtime

安装开发依赖：

```bash
uv sync --group dev --extra inference
```

服务端入口：

```bash
uv run --extra inference visual-events-server --config configs/pc-ga-server.toml
```

CLI 入口：

```bash
uv run visual-events-cli --config <cli-config.toml>
```

模型权重不进入 Git。默认真实模型路径由配置提供；公开 demo 使用 `runtime/models/yolov8n-pose.pt`、`runtime/models/face-buffalo-s/` 和 `runtime/models/scene-mobileclip2-s0/`。

## Core Path

```text
DDS JPEG @10Hz
  -> visual-events-cli
  -> WebSocket stream
  -> visual-events-server
  -> visual_state / attention / semantic_events
  -> visual-events-cli
      -> DDS gaze target
      -> Botified event context
```

核心约束：

- 一个输入源：机器人侧 JPEG 帧。
- 一个服务协议：WebSocket streaming。
- 一个高频状态：`visual_state`。
- 一个注视输出：DDS gaze target。
- 一个低频事件出口：Botified event context。
- 身份、记忆和示教走 server-side memory / identity service；CLI 只投影，不做人脸匹配、不写库。

## Docs

- [Real Model Evidence Demo Development Plan](docs/real-model-evidence-demo-development-plan.md)
- [Identity Overlay active plan](docs/identity-overlay-product-development-plan.md)
- [协议草案](common/schema/protocol.md)
- DDS contracts: [camera JPEG](common/schema/dds/camera_jpeg_contract.md), [gaze target v1](common/schema/dds/gaze_target_v1.md), [head state v1](common/schema/dds/head_state_v1.md)

旧资料在 [docs/legacy/](docs/legacy/) 中，仅供追溯，不作为当前设计约束。
