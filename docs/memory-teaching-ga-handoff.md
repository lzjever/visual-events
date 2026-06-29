# Memory Teaching GA Handoff

本文是 memory-teaching GA 的最小复现说明，面向开发和交付团队。本 handoff 只覆盖 PC 本地 runner 证据；不声明真机、RK、现场、HIL、真实机器人摄像头 DDS 或 owner sign-off 已通过。

## 能力边界

- PC 本地可复现：fake/full contract 和 local-smoke/real model runner。
- 支持示教当前交互对象本人、第三人介绍中的被指向人，以及整图 scene memory。
- `target.kind=object` 当前是 negative-only：预期 unsupported / no-write。
- region/object memory 不作为当前可用能力；不要把 object 请求降级为 scene/region 写库。
- CLI 只投影 memory semantic event，不负责示教动作或新增 agent-facing teach 能力。

## 标准命令

Fake/full contract：

```bash
uv run python tools/run_memory_teaching_ga_e2e.py --data-dir val-data --out <out>
```

Local-smoke/real model：

```bash
uv run python tools/run_memory_teaching_ga_e2e.py --data-dir val-data --out <out> --local-smoke --embedding-backend local --person-model-path runtime/models/face-buffalo-s --scene-model-path runtime/models/scene-mobileclip2-s0 --inference-backend ultralytics --pose-model-path runtime/models/yolov8n-pose.pt
```

## 通过标准

查看 `<out>/report.json`。

Fake/full contract 应满足：

- `ok=true`
- `scene_count=15`
- checks 中 `all_scenes_replayed`、self、third-person、scene、object no-write、supporting contracts、CLI projection、bounded recognition 均为 true。

Local-smoke/real model 应满足：

- `ok=true`
- `real_model_evidence=true`
- self/scene/third-person local checks 均为 true。
- bounded recognition 为 true。

Third-person 重点字段：

- `resolution_reason=pose_pointing_to_person`
- `introducer_ref != resolver_target_ref`
- `pose_stability_window.failure_reason=null`
- no debug fixture：`debug_test_channel_enabled=false`、`fixture_inputs_consumed=[]`、`debug_fixture_used_for_target_resolution=false`

Visual evidence：

- `visual_evidence_index` 写入 report。
- `visual-evidence/index.html` 可人工打开。
- fake/full contract 通常生成 4 张 overlay：self、third-person、scene、object negative。
- local-smoke 通常生成 3 张 overlay：self、third-person、scene。object negative 不在 local-smoke 主流程中执行。

## Artifact 和 Git Hygiene

- 所有输出写到命令指定的 `<out>` 下；以本次 `<out>/report.json` 为准。
- `<out>/runtime/`、memory artifacts、visual evidence、`val-data/` 和模型目录不提交到 Git。
- `runtime/models/...` 需由本机或交付环境准备；模型权重不入 Git。

## Troubleshooting

- 缺少 `val-data`：runner 会失败；准备本地 ignored `val-data/` 后重跑。
- 缺少 local model path：local-smoke preflight 会失败；确认命令中的三个 model path 存在。
- 旧 artifact 混淆：历史 `artifacts/...` 可能是 stale；当前命令的 `<out>` 才是 authoritative output。
- `StarletteDeprecationWarning` 当前是 non-blocking warning，不表示 runner 失败。
