# Visual Events Server Handoff

日期：2026-06-27

## 1. Scope / Status

本 handoff 只覆盖 `visual-events-server`。当前 server 已完成 S0-S8 baseline：WebSocket protocol、真实/Mock inference backend 边界、Ultralytics pose adapter、项目内 ByteTrack-style IoU/TTL tracker baseline、attention selector、semantic events、`val-data/` E2E runner、轻量 perf report、release/runtime smoke verification、S6.3 semantic event first-trigger/timeline gate，以及 S8 opt-in server metrics evidence。`tools/run_val_data_e2e.py` 已支持 stationary 全量、unknown 全量 suppression、moving targeted suppression gates，并在 `val-data/` 上检查 expected first trigger frame tolerance <= 3 frames、forbidden scene events 和 `pic_walk_in_stop` event ordering。`tools/run_runtime_smoke.py` 已验证 `runtime/venv` server 可启动，并通过 `/healthz` 证明本次 server process identity。当前 handoff 已有 runtime server S8 full matrix、metrics aggregation 和 5-minute soak pass evidence；moving suppression 由 warm-up/full matrix 的 `__head_moving` artifacts 证明，不在 soak loop 内重复。对应 artifacts 仍在 ignored paths 下，不提交到 Git。

当前阶段没有正式 robot CLI：不接 DDS，不输出 Botified frame，不做头部控制闭环。`tools/replay_val_data.py`、`tools/run_val_data_e2e.py` 和 `tools/run_runtime_smoke.py` 是开发/验证工具，不是产品 CLI。`tools/run_runtime_smoke.py` 只验证 release/runtime 启动，不替代 `val-data` E2E/soak。

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
runtime/venv/bin/visual-events-server --config runtime/config/s2.toml
```

Release/runtime smoke：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/run_runtime_smoke.py \
  --config runtime/config/s2.toml \
  --port 8767
```

The smoke tool runs the release sync with repo-local `runtime/cache/uv` and `runtime/venv`, starts `runtime/venv/bin/visual-events-server`, polls `/healthz`, verifies `healthz_pid == server_pid`, writes `artifacts/runtime-smoke/report.json`, and stops the server. It is not the product CLI and is not a substitute for the `val-data` E2E/soak gates.

开发环境也可直接启动：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run visual-events-server --host 127.0.0.1 --port 8765
```

S6 E2E：

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/run_val_data_e2e.py \
  --server ws://127.0.0.1:8767/v1/stream \
  --data-dir val-data \
  --out artifacts/e2e
```

该命令运行 full matrix：stationary 全量 7 scene、unknown 全量 suppression 7 scene、moving targeted suppression 5 scene。stationary `all` gate 包含 S6.3 first-trigger/timeline checks；moving targeted cases 使用 `__head_moving` artifact 目录并只跑 events gate。

S6.1/S7/S8 5-minute soak evidence gate：

```bash
SERVER_PID=<visual-events-server pid>

UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/run_val_data_e2e.py \
  --server ws://127.0.0.1:8767/v1/stream \
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

Soak requires realtime playback. Do not add `--no-realtime`; the runner rejects that combination. Soak also requires `--response-timeout-ms` and `--server-pid` so a hung request or unreadable RSS sample fails with artifact evidence instead of silently passing. The warm-up/full matrix includes targeted `__head_moving` cases; the 300s soak loop intentionally excludes moving cases and only loops stationary + unknown.

S8 server metrics are disabled by default. Enable them with config:

```toml
[metrics]
jsonl_path = "artifacts/perf/server_metrics.jsonl"
```

or with server CLI:

```bash
runtime/venv/bin/visual-events-server \
  --config runtime/config/s2.toml \
  --host 127.0.0.1 \
  --port 8767 \
  --metrics-jsonl artifacts/perf/server_metrics.jsonl
```

The server writes one ignored JSONL `frame_metrics` line per successfully processed frame and does not change the `visual_state` wire protocol. The E2E runner consumes metrics only from an explicit `--server-metrics-jsonl <path>`; omitted means the old unavailable behavior. If that path is provided but has no usable `total` phase samples, E2E fails with `server_metrics_unavailable`; the runner clears any stale metrics file at start.

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
| `artifacts/runtime-smoke/report.json` | `58552d81b7f0a46741f626ddce373f5e50e7672ef7a5d1d3550aa9eae7750439` |
| `artifacts/e2e/report.json` | `67deba187e11bf53382f379b039b840f7847d65695546fa615eb13caa3274d49` |
| `artifacts/perf/server_perf.json` | `563a51402baa45db3662921b49a3e131364bf8b3b46f1e31214e8fd59b52fa41` |
| `artifacts/perf/server_metrics.jsonl` | `9b059a2458111d6e6af575c164f89d1dfcca40b0cfb0c572fc1f1e922ac2170a` |

Latest S8 runtime smoke evidence:

- Status: pass; `passed == true`.
- Command:

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --group dev python tools/run_runtime_smoke.py --config runtime/config/s2.toml --port 8767
```

- Sync env: `runtime/cache/uv`, `runtime/venv`.
- Server command: `/home/galbot/works/visual-events/runtime/venv/bin/visual-events-server --config /home/galbot/works/visual-events/runtime/config/s2.toml --host 127.0.0.1 --port 8767`.
- Server pid in smoke report: 840437; healthz pid: 840437; `healthz_identity_verified == true`; elapsed 1.0514707476831973s; stopped by the smoke tool.
- Artifact: `artifacts/runtime-smoke/report.json`; ignored and must not be committed.

Latest runtime server S8 E2E/soak invocation:

- Server command: `runtime/venv/bin/visual-events-server --config runtime/config/s2.toml --host 127.0.0.1 --port 8767 --metrics-jsonl artifacts/perf/server_metrics.jsonl`.
- Server pid: 841084.
- Runner command:

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --group dev python tools/run_val_data_e2e.py --server ws://127.0.0.1:8767/v1/stream --data-dir val-data --out artifacts/e2e --camera front --fps 10 --response-timeout-ms 30000 --soak-seconds 300 --server-pid 841084 --soak-memory-growth-max-mb 64 --soak-sample-interval-s 10 --server-metrics-jsonl artifacts/perf/server_metrics.jsonl
```

Latest runtime server S8 warm-up/full-matrix report:

- Status: pass; `overall_pass == true`.
- Cases: 19 total = 7 stationary `all` + 7 unknown `events` + 5 moving targeted `events`.
- Moving targeted cases passed: `pci_stand__head_moving`, `pic_1_l_to_r__head_moving`, `pic_1_r_to_l__head_moving`, `pic_persone_walk_in__head_moving`, `pic_walk_in_stop__head_moving`.
- Frames: 1563 sent, 1563 ok
- Errors: 0
- Aggregate Hz: 9.9368711648818
- Aggregate latency: p50 22.232437040656805 ms, p95 23.635465651750565 ms, p99 24.868441745638847 ms
- Error rate: 0.0
- Semantic event timeline gate: pass.
- `pic_1_r_to_l` stationary events: `attention_target_changed:1`, `person_appeared:2`, `person_passing_by:1`; first `person_passing_by` = 47, expected first = 47; no `person_waving`; `timing_errors == 0`; `forbidden == {}`; `order_violations == 0`.
- `pic_hello`: first `person_waving` = 12, expected first = 12.
- `pic_walk_in_stop`: first `person_approaching_robot` = 9, first `person_stopped_near_robot` = 63; approaching-before-stopped ordering passes.
- Waving rule fix: waving wrists must be clearly above the shoulder, closing the prior `pic_1_r_to_l` walking-arm-swing false positive while preserving `pic_hello` `person_waving` at frame 12.

Latest runtime server S8 soak evidence:

- Status: pass.
- Report fields: `overall_pass == true`, `soak.enabled == true`, `soak.passed == true`.
- Moving suppression evidence: present in warm-up/full matrix `__head_moving` artifacts; moving cases are intentionally excluded from the soak loop.
- Soak duration: target 300s; elapsed 347.898024847731s; passed.
- Soak cases: 42 cases = 3 loops * 14 stationary + unknown cases.
- Soak frames: 3456 sent, 3456 ok, 0 errors.
- Soak Hz: 9.936939815800379.
- Soak latency: p50 22.34238525852561 ms, p95 23.670367896556854 ms, p99 24.51172610744834 ms.
- Soak error rate: 0.0.
- Soak RSS: start 1677.96484375 MB, end 1690.7890625 MB, growth 12.82421875 MB <= 64 MB, samples 5.
- Soak artifacts: `artifacts/e2e/soak/loop_0001/...` through `artifacts/e2e/soak/loop_0003/...`; these are ignored artifacts and must not be committed.

Latest S8 server metrics evidence:

- JSONL samples: 5019 `frame_metrics` lines = 1563 full matrix frames + 3456 soak frames.
- Phase latency summary from `artifacts/perf/server_perf.json`:

| Phase | Count | P50 ms | P95 ms | P99 ms |
| --- | ---: | ---: | ---: | ---: |
| `decode` | 5019 | 3.253 | 3.54 | 3.752 |
| `infer` | 5019 | 6.469 | 7.01 | 7.591 |
| `postprocess` | 5019 | 0.17 | 0.214 | 0.253 |
| `tracking` | 5019 | 0.065 | 0.099 | 0.11 |
| `attention` | 5019 | 0.011 | 0.015 | 0.017 |
| `events` | 5019 | 0.137 | 0.294 | 0.326 |
| `response` | 5019 | 0.008 | 0.011 | 0.013 |
| `total` | 5019 | 10.385 | 11.213 | 11.929 |

- Memory summary: available true; count 5019; min 1755070464 bytes; max 1775976448 bytes; last 1772920832 bytes.
- VRAM summary: available true; count 5019; device `0`; device consistent true; max allocated 13242368 bytes; max reserved 327155712 bytes; both <= 4 GiB; `vram_4gib == true`.

Important caveat: S8 metrics are available only when explicitly enabled and are evidence artifacts, not a full observability pipeline. Per-case latency is diagnostic. VRAM evidence is process PyTorch CUDA allocated/reserved memory, not RK3588 validation and not total board memory validation. `val-data/`, `runtime/`, `artifacts/`, metrics JSONL, model files, and caches remain ignored/untracked and must not be committed.

## 7. Known Limitations / Failure Scenarios

- No server DDS integration, Botified output, formal robot CLI, or gaze controller.
- No face identity, face recognition, long-term identity, or gaze/eye-contact judgement.
- Motion-sensitive events are suppressed when `head_motion.state` is `moving`, `unknown`, or missing: `person_passing_by`, `person_approaching_robot`, `person_stopped_near_robot`.
- `head_motion=unknown` suppression is full 7-scene coverage; `head_motion=moving` suppression is targeted 5-scene coverage and only uses the events gate.
- Event gates are first-trigger/timeline gated on `val-data/`; they are still not dense per-frame manual annotations.
- Release/runtime is verified by runtime smoke plus runtime server E2E/soak evidence; the runtime artifacts remain local ignored files and are not committed.
- RK3588 is not validated; only the `InferBackend` boundary is preserved for future migration.
- Product release remains blocked until Ultralytics model/license status is resolved.
- The pose baseline can miss people or keypoints; V1 response is to tune thresholds/tracker TTL and add validation evidence, not to train a new model in this repo.

Current thresholds from the development plan:

- 10Hz replay should produce `visual_state` at >= 9Hz.
- GPU server aggregate latency target: p95 < 120 ms, p99 < 200 ms.
- Error frame ratio target: < 1%.
- S6.1/S7 soak target: `--soak-seconds 300`, `soak.hz >= 9`, `soak.total_latency_ms.p95 < 120`, `soak.total_latency_ms.p99 < 200`, `soak.error_rate < 1%`, RSS growth <= configured `soak.rss_mb.max_growth` default 64 MB.
- S8 metrics target: when explicitly enabled, phase metrics have usable `total` samples and PyTorch CUDA allocated/reserved VRAM evidence is <= 4 GiB.
- S6.3 first-trigger tolerance: <= 3 frames.

## 8. Handoff Checklist

Pass only if all required items are true:

- `visual-events-server` starts with the release/runtime environment.
- `artifacts/runtime-smoke/report.json` exists for the run being handed off and contains `passed == true`.
- The runtime smoke report shows `healthz_pid == server_pid` and `healthz_identity_verified == true`; HTTP 200 alone is not sufficient.
- The runtime smoke report shows sync env under `runtime/cache/uv` and `runtime/venv`, and server command under `runtime/venv/bin/visual-events-server`.
- `runtime/models/yolov8n-pose.pt` exists locally and matches the manifest hash.
- Real backend startup fails clearly if the configured model file is missing.
- Ultralytics cache env vars resolve under `runtime/cache/*` before model construction.
- `val-data/` is present locally for E2E, but is not committed.
- Full matrix `val-data/` E2E was run against the real runtime server.
- `artifacts/e2e/report.json` and `artifacts/perf/server_perf.json` exist for the run being handed off.
- Moving targeted suppression artifacts exist for the run being handed off: `artifacts/e2e/<scene>__head_moving/...` for the five targeted scenes.
- S6.3 semantic event first-trigger/timeline gate passed: expected first trigger frame tolerance <= 3 frames, forbidden scene events empty, and `pic_walk_in_stop` ordering valid.
- S6.1/S7 soak passed, `artifacts/e2e/soak/loop_0001/...` exists, and both reports contain `soak.enabled == true` and `soak.passed == true`.
- S8 metrics evidence, if claimed, was produced by a metrics-enabled server and runner `--server-metrics-jsonl`; phase latency, RSS, and VRAM availability are explicit in `server_perf.json`.
- Motion-sensitive events are suppressed for `head_motion=unknown` full coverage and `head_motion=moving` targeted coverage.
- Server output conforms to `common/schema/protocol.md`.
- Handoff notes include the model manifest, license status, known limitations, and perf caveats.

Fail handoff if any of these are true:

- Mock tests are used as a substitute for real `val-data/` E2E.
- `val-data/`, `runtime/`, model weights, caches, or `artifacts/` are added to Git.
- Release/runtime is claimed without a passing `artifacts/runtime-smoke/report.json`.
- Release/runtime E2E/soak is claimed from a source `.venv` server instead of `runtime/venv/bin/visual-events-server`.
- Server event rules depend on a formal robot CLI.
- S6.3 event correctness is claimed without first-trigger/timeline evidence.
- Product/release materials claim Ultralytics Enterprise licensing without confirmation.
- A 5-minute soak pass is claimed without matching `artifacts/e2e/soak/...` files and matching `soak` summaries in both report files.
- S8 phase/RSS/VRAM evidence is claimed without the ignored server metrics JSONL and matching `server_perf.json` aggregation.
