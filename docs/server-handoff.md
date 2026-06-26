# Visual Events Server Handoff

日期：2026-06-26

## 1. Scope / Status

本 handoff 只覆盖 `visual-events-server`。当前 server 已完成 S0-S6 baseline：WebSocket protocol、真实/Mock inference backend 边界、Ultralytics pose adapter、项目内 ByteTrack-style IoU/TTL tracker baseline、attention selector、semantic events、`val-data/` E2E runner 和轻量 perf report。当前 handoff 已有 S6.1/S7 5-minute soak pass evidence；对应 artifacts 仍在 ignored paths 下，不提交到 Git。

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
UV_CACHE_DIR=runtime/cache/uv UV_PROJECT_ENVIRONMENT=runtime/venv \
  uv sync --frozen --no-dev --no-editable --extra inference \
  --reinstall-package visual-events-server
```

`--reinstall-package visual-events-server` is intentional for development/handoff verification: when the project version is unchanged, it forces the current project wheel to refresh inside `runtime/venv`. It does not change the dependency lock, `--frozen`, `--no-dev`, or `--no-editable` policy.

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

S6.1/S7 5-minute soak evidence gate：

```bash
SERVER_PID=<visual-events-server pid>

UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/run_val_data_e2e.py \
  --server ws://127.0.0.1:8765/v1/stream \
  --data-dir val-data \
  --out artifacts/e2e \
  --camera front \
  --fps 10 \
  --response-timeout-ms 30000 \
  --soak-seconds 300 \
  --server-pid "$SERVER_PID" \
  --soak-memory-growth-max-mb 64 \
  --soak-sample-interval-s 10
```

Soak requires realtime playback. Do not add `--no-realtime`; the runner rejects that combination. Soak also requires `--response-timeout-ms` and `--server-pid` so a hung request or unreadable RSS sample fails with artifact evidence instead of silently passing.

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
| `artifacts/e2e/report.json` | `265c510db1a984f2f059494ed8aedf8643227b6795011eb6c6ba7dd4f28668d7` |
| `artifacts/perf/server_perf.json` | `0d46913be696a380b53f9c467fff6e637dfefcd24c09b88a34bd875ef5b5b1bf` |

Latest S6 realtime warm-up/full-matrix report:

- Cases: 14
- Frames: 1152 sent, 1152 ok
- Errors: 0
- Aggregate Hz: 9.935054661031874
- Aggregate latency p95: 23.527881130576134 ms
- Aggregate latency p99: 24.289363995194435 ms
- Error rate: 0.0
- Server phase latency, VRAM, and memory: unavailable by design in the S6 runner

Latest S6.1/S7 soak evidence:

- Status: pass.
- Command environment: current source checkout with `.venv` server, `UV_CACHE_DIR=.uv-cache`, `UV_PROJECT_ENVIRONMENT=.venv`.
- Command parameters: `--camera front --fps 10 --response-timeout-ms 30000 --soak-seconds 300 --server-pid 507273 --soak-memory-growth-max-mb 64 --soak-sample-interval-s 10`.
- Report fields: `overall_pass == true`, `soak.enabled == true`, `soak.passed == true`.
- Soak duration: target 300s, elapsed 347.9469155697152s.
- Soak loops/cases: 3 loops, 42 cases.
- Soak frames: 3456 sent, 3456 ok, 0 errors.
- Soak Hz: 9.935543594743722.
- Soak latency: p95 23.538413923233747 ms, p99 24.733862839639187 ms.
- Soak error rate: 0.0.
- Soak RSS: start 1683.86328125 MB, end 1683.23828125 MB, growth 0.09375 MB, max growth 64 MB, samples 5.
- Soak artifacts: `artifacts/e2e/soak/loop_0001/...` through `artifacts/e2e/soak/loop_0003/...`; these are ignored artifacts and must not be committed.

Important caveat: the current S6 gate uses aggregate perf. Per-case latency is diagnostic. Decode/preprocess/infer/postprocess/tracking/events phase metrics, VRAM, and top-level memory metrics are unavailable until a future server metrics module exists. `server_perf.json.vram.available == false` is expected; it does not verify VRAM < 4GB. Soak RSS is process-level evidence from `/proc/<pid>/status`, not a complete metrics pipeline.

## 7. Known Limitations / Failure Scenarios

- No server DDS integration, Botified output, formal robot CLI, or gaze controller.
- No face identity, face recognition, long-term identity, or gaze/eye-contact judgement.
- Motion-sensitive events are suppressed when `head_motion.state` is `moving`, `unknown`, or missing: `person_passing_by`, `person_approaching_robot`, `person_stopped_near_robot`.
- Event gates are scene-level smoke checks from `val-data/`; they are not manual frame-level annotations.
- S6.1 soak pass evidence exists for this handoff, but it was collected on the current source `.venv` server and does not validate release/runtime packaging by itself.
- RK3588 is not validated; only the `InferBackend` boundary is preserved for future migration.
- Product release remains blocked until Ultralytics model/license status is resolved.
- The pose baseline can miss people or keypoints; V1 response is to tune thresholds/tracker TTL and add validation evidence, not to train a new model in this repo.

Current thresholds from the development plan:

- 10Hz replay should produce `visual_state` at >= 9Hz.
- GPU server aggregate latency target: p95 < 120 ms, p99 < 200 ms.
- Error frame ratio target: < 1%.
- S6.1/S7 soak target: `--soak-seconds 300`, `soak.hz >= 9`, `soak.total_latency_ms.p95 < 120`, `soak.total_latency_ms.p99 < 200`, `soak.error_rate < 1%`, RSS growth <= configured `soak.rss_mb.max_growth` default 64 MB.
- VRAM < 4GB is a future GPU capacity / metrics evidence item, not a S6.1 soak pass condition.
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
- S6.1/S7 soak passed, `artifacts/e2e/soak/loop_0001/...` exists, and both reports contain `soak.enabled == true` and `soak.passed == true`.
- Motion-sensitive events are suppressed for `head_motion=unknown` and `head_motion=moving`.
- Server output conforms to `common/schema/protocol.md`.
- Handoff notes include the model manifest, license status, known limitations, and perf caveats.

Fail handoff if any of these are true:

- Mock tests are used as a substitute for real `val-data/` E2E.
- `val-data/`, `runtime/`, model weights, caches, or `artifacts/` are added to Git.
- Server event rules depend on a formal robot CLI.
- Product/release materials claim Ultralytics Enterprise licensing without confirmation.
- A 5-minute soak pass is claimed without matching `artifacts/e2e/soak/...` files and matching `soak` summaries in both report files.
