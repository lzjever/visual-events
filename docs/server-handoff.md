# Visual Events Server Handoff

日期：2026-06-26

## 1. Scope / Status

本 handoff 只覆盖 `visual-events-server`。当前 server 已完成 S0-S6 baseline：WebSocket protocol、真实/Mock inference backend 边界、Ultralytics pose adapter、项目内 ByteTrack-style IoU/TTL tracker baseline、attention selector、semantic events、`val-data/` E2E runner 和轻量 perf report。

当前阶段没有正式 robot CLI：不接 DDS，不输出 Botified frame，不做头部控制闭环。`tools/replay_val_data.py` 和 `tools/run_val_data_e2e.py` 是开发/验证工具，不是产品 CLI。

## 2. Commands

开发环境：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv sync --group dev
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --group dev pytest -q
```

开发真实推理依赖：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv sync --group dev --extra inference
```

Release/runtime 同步：

```bash
UV_CACHE_DIR=runtime/cache/uv UV_PROJECT_ENVIRONMENT=runtime/venv uv sync --frozen --no-dev --no-editable --extra inference
```

Server 启动：

```bash
runtime/venv/bin/visual-events-server --config runtime/config.toml
```

开发环境也可直接启动：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run visual-events-server --host 127.0.0.1 --port 8765
```

S6 E2E：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/run_val_data_e2e.py \
  --server ws://127.0.0.1:8765/v1/stream \
  --data-dir val-data \
  --out artifacts/e2e
```

## 3. Runtime / Cache / Model Policy

Non-git local paths:

- `val-data/`: local validation data, required for real E2E, never committed.
- `runtime/`: local runtime venv, config, model files, and inference caches, never committed.
- `runtime/venv/`: release/runtime virtual environment.
- `runtime/cache/uv/`: release `uv` cache.
- `runtime/cache/yolo/`: `YOLO_CONFIG_DIR`.
- `runtime/cache/torch/`: `TORCH_HOME`.
- `runtime/cache/xdg/`: `XDG_CACHE_HOME`.
- `runtime/cache/matplotlib/`: `MPLCONFIGDIR`.
- `runtime/models/`: model weights, never committed.
- `artifacts/`: E2E/perf outputs, never committed.

The server sets inference cache environment variables before constructing the Ultralytics model. It does not change `HOME`.

## 4. Model Manifest

| Field | Value |
| --- | --- |
| Name | `YOLOv8n-pose` |
| Runtime path | `runtime/models/yolov8n-pose.pt` |
| SHA-256 | `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212` |
| Source / provenance | Official Ultralytics pretrained baseline |
| Fine-tuning | Not fine-tuned by this repo |
| Git policy | Downloaded/prepared outside Git; do not commit the weight file |
| Startup behavior | No automatic download; missing file fails real backend startup with a config error |

The real backend must load the explicit configured `model_path`. It must not call `YOLO("yolov8n-pose.pt")`, because that can trigger implicit upstream download/cache behavior.

## 5. License Status

This section is an engineering handoff note, not legal advice.

Official references:

- Ultralytics licensing: <https://www.ultralytics.com/license>
- Ultralytics docs: <https://docs.ultralytics.com/>

As of this handoff, the local status is internal POC only unless one of these is confirmed:

- The repository/product satisfies AGPL-3.0 obligations for Ultralytics YOLO code/models.
- An Ultralytics Enterprise License covering this product/use is confirmed.

The Ultralytics license page states that Ultralytics YOLO trained models are AGPL-3.0 by default, and presents Enterprise licensing as the path for commercial/proprietary embedding of Ultralytics YOLO code/models. Do not claim that Enterprise licensing is already satisfied. Treat product release with Ultralytics YOLO code/models as blocked until licensing is resolved by the responsible owner.

## 6. E2E / Perf Evidence

Latest ignored artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/e2e/report.json` | `7ec9c1725286390c5f5b4fdb67e757095607c9df94aebc134d0b0f5f55c26003` |
| `artifacts/perf/server_perf.json` | `a01b6056c3c9889dec7a00f33fe6440c01e0df8fd01ec3f887d06cd5ed6fb29a` |

Latest S6 realtime report:

- Cases: 14
- Frames: 1152
- Errors: 0
- Aggregate Hz: 9.797
- Aggregate latency p95: 23.26 ms
- Aggregate latency p99: 24.03 ms
- Server phase latency, VRAM, and memory: unavailable by design in the S6 runner

Important caveat: the current S6 gate uses aggregate perf. Per-case latency is diagnostic. Decode/preprocess/infer/postprocess/tracking/events phase metrics, VRAM, and memory are unavailable until a future server metrics module exists.

## 7. Known Limitations / Failure Scenarios

- No server DDS integration, Botified output, formal robot CLI, or gaze controller.
- No face identity, face recognition, long-term identity, or gaze/eye-contact judgement.
- Motion-sensitive events are suppressed when `head_motion.state` is `moving`, `unknown`, or missing: `person_passing_by`, `person_approaching_robot`, `person_stopped_near_robot`.
- Event gates are scene-level smoke checks from `val-data/`; they are not manual frame-level annotations.
- No confirmed 5-minute soak result in this handoff unless a matching artifact is added later.
- RK3588 is not validated; only the `InferBackend` boundary is preserved for future migration.
- Product release remains blocked until Ultralytics model/license status is resolved.
- The pose baseline can miss people or keypoints; V1 response is to tune thresholds/tracker TTL and add validation evidence, not to train a new model in this repo.

Current thresholds from the development plan:

- 10Hz replay should produce `visual_state` at >= 9Hz.
- GPU server aggregate latency target: p95 < 120 ms, p99 < 200 ms.
- Error frame ratio target: < 1%.
- Regression trigger-frame tolerance: <= 3 frames or <= 300 ms.

## 8. Handoff Checklist

Pass only if all required items are true:

- `visual-events-server` starts with the release/runtime environment.
- `runtime/models/yolov8n-pose.pt` exists locally and matches the manifest hash.
- Real backend startup fails clearly if the configured model file is missing.
- Ultralytics cache env vars resolve under `runtime/cache/*` before model construction.
- `val-data/` is present locally for E2E, but is not committed.
- Full `val-data/` E2E was run against the real server.
- `artifacts/e2e/report.json` and `artifacts/perf/server_perf.json` exist for the run being handed off.
- Motion-sensitive events are suppressed for `head_motion=unknown` and `head_motion=moving`.
- Server output conforms to `common/schema/protocol.md`.
- Handoff notes include the model manifest, license status, known limitations, and perf caveats.

Fail handoff if any of these are true:

- Mock tests are used as a substitute for real `val-data/` E2E.
- `val-data/`, `runtime/`, model weights, caches, or `artifacts/` are added to Git.
- Server event rules depend on a formal robot CLI.
- Product/release materials claim Ultralytics Enterprise licensing without confirmation.
