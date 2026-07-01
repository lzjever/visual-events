# Real Model Evidence Demo Development Plan

## 1. 背景

当前 evidence 工具已经能生成可视化结果，但用户可见心智过重：

- 通用视觉 demo 需要用户理解 server、replay JSONL 和生成器之间的关系。
- memory / teaching demo 的真实模型路径藏在开发内部命令里，用户很难判断结果是否来自真实脸部识别模型。
- 历史输出目录和脚本名混入了阶段性开发心智，会让人误以为 demo 不是正式真实模型结果。

本计划的目标是把 demo 收敛成两个稳定入口，默认全部使用真实模型。公开文档只保留当前入口和当前输出目录。测试可以使用 test double，但它们不属于产品 demo 心智。

## 2. 产品目标

交付两个独立 demo 入口：

```bash
uv run --extra inference python tools/run_visual_demo.py --data-dir val-data
uv run --extra inference python tools/run_memory_demo.py --data-dir val-data
```

两个入口职责固定：

| 入口 | 输出目录 | 展示能力 |
| --- | --- | --- |
| `tools/run_visual_demo.py` | `artifacts/demo/visual/` | detection、tracking、attention、semantic events、no-event/no-engage reason |
| `tools/run_memory_demo.py` | `artifacts/demo/memory/` | known person、familiar unknown、用户示教、指向目标解析、场景记忆、事件 identity_context |

用户打开的入口固定为：

```text
artifacts/demo/visual/index.html
artifacts/demo/memory/index.html
```

报告文件只描述本次 demo 事实，避免外推到其他运行环境。

本计划采用 clean break：开发完成后，README 和 active docs 只指向这两个入口。

## 3. 命名和心智规则

公开路径和 README 主命令只使用以下词：

- visual demo
- memory demo
- real model
- known person
- familiar unknown
- scene activated

历史测试/阶段性入口不再作为公开入口，相关文件直接归档、重命名或删除。README 和 active docs 不列替代命令。

内部实现可以复用已有函数，但用户不需要知道这些函数名。开发时可以重命名或删除历史 wrapper；公开文档不描述历史 CLI 参数、目录结构或 artifact 输入。

## 4. 真实模型规则

两个公开入口默认都跑真实模型：

| 能力 | 默认模型路径 |
| --- | --- |
| pose / person detection | `runtime/models/yolov8n-pose.pt` |
| face detection / face embedding | `runtime/models/face-buffalo-s/` |
| scene embedding | `runtime/models/scene-mobileclip2-s0/` |

规则：

1. 找不到模型直接失败，错误信息明确指出缺哪个路径。
2. 不允许从公开入口静默 fallback 到非真实模型后端。
3. `report.json` 和 HTML 顶部必须显示本次使用的模型路径和 runtime label。
4. `report.json` 必须包含：

```json
{
  "real_model_evidence": true,
  "models": {
    "pose": "runtime/models/yolov8n-pose.pt",
    "face": "runtime/models/face-buffalo-s",
    "scene": "runtime/models/scene-mobileclip2-s0"
  }
}
```

## 5. Visual Demo 范围

`run_visual_demo.py --data-dir val-data` 默认执行完整真实模型流程：

1. 启动本地 `visual-events-server`，使用真实 `configs/pc-ga-server.toml` 或等价 real-model config。
2. 复用 `tools.replay_val_data.replay_data_dir()` 回放全量 `val-data`，不新增第二套 WebSocket replay loop。
3. 写出本次 visual state 记录到 `artifacts/demo/visual/visual_state.jsonl`。
4. 复用现有 visual renderer 生成 overlay 图片、scene 页面和根 `index.html`。
5. 运行结束后关闭临时 server。

图片和 HTML 至少展示：

- person bbox / track label
- attention target
- semantic event label
- scene person count
- track count
- no-event 或 no-engage reason（如果 visual_state 中已有相应字段）
- 如果 `visual_state` 已经包含 keypoints，可以画 pose skeleton；不要为了 demo 反向扩大 public visual schema。

公开参数保持最小：

- `--data-dir`
- `--out`，但必须解析到 `artifacts/demo/visual/` 或 `artifacts/demo/memory/` 下
- `--camera`
- `--fps`
- `--no-realtime`

公开 demo 不支持外部 server、历史 JSONL 或历史 replay 输入。需要定位底层 replay 问题时，开发者直接使用 `tools/replay_val_data.py` 或模块级 helper，不把这些模式塞回 demo CLI。

## 6. Memory Demo 范围

新增公开入口 `tools/run_memory_demo.py`。它是 memory / identity / teaching demo 的唯一用户入口。

默认行为：

1. 使用真实 YOLO pose、InsightFace face embedding、scene embedding。
2. 从 `val-data` 发现全部场景和 transcript 消息。
3. 对需要交互的 transcript，在对应图像附近模拟 agent 调用 memory API。
4. 使用热 buffer / attention / pose pointing 解析用户示教目标。
5. 生成 `artifacts/demo/memory/index.html`。

必须展示的 demo 主线：

- 用户说“请记住我，我是 xxx”，系统推断目标是当前交互人，做人脸入库。
- 用户指向第三个人介绍身份，系统用 pose pointing 解析目标，不要求 agent 传 track id / bbox。
- 未知但多次出现的人，触发 `familiar_unknown_present`。
- 场景示教后，再次看到相似场景触发 `scene_activated`。

如果同一次真实模型运行已经产生以下信息，也应在页面中展示摘要，但不要为了 demo 新增产品逻辑：

- 已知人再次出现，事件包含 `known_person_present` 和身份信息。
- 匿名熟客被用户命名后，server 自动合并成 known person，不要求 client 调 merge API。
- 普通人物事件发生时，如果能召回身份，事件附带 `identity_context`。
- current visual snapshot 展示 agent 可读的人物身份摘要和 opaque `target_ref`。

默认页面展示：

- person bbox
- face bbox
- identity label
- familiar unknown label
- pose skeleton
- pointing ray
- selected target candidate
- event label

face crop、face score、match score 放入内部排障附件，不作为首页主视觉信息。

agent-facing snapshot 仍然不暴露 raw track id、bbox、keypoints、embedding、crop path 或 `stream_ref`。

## 7. 输出目录规则

两个入口每次只清理自己管理的输出目录。公开输出必须位于：

```text
artifacts/demo/visual/
artifacts/demo/memory/
```

两个 demo 只要求根目录有 `index.html` 和 `report.json`。内部图片、scene 页面、states、crops、raw 文件的目录结构分别沿用现有 renderer，不强行统一，避免无收益迁移。用户打开 `index.html` 不需要理解内部目录。

公开 `report.json` 使用字段白名单，只表达本次 demo 事实：

- `real_model_evidence`
- `models`
- `outputs`
- `data_dir`
- `scene_count`
- `frame_count`
- `case_count`
- `error_count`
- `cases`

公开 HTML 只能链接 `artifacts/demo/visual/` 或 `artifacts/demo/memory/` 下的 demo 图片、case 页面和必要摘要。内部 raw/troubleshooting 文件可以写在输出目录内供开发排障，但不进入公开首页，不作为用户需要理解的 demo 入口。

禁止把新结果写入 `val-data/`、`runtime/models/` 或 Git tracked 路径；显式 `--out` 也必须位于 `artifacts/demo/` 下。

## 8. 实现计划

### Step 1: 收敛公开命令和文档

- README 主路径只保留两个命令：
  - `tools/run_visual_demo.py --data-dir val-data`
  - `tools/run_memory_demo.py --data-dir val-data`
- `docs/identity-overlay-product-development-plan.md` 的 demo 命令同步改为 memory demo 单入口。
- 历史文档继续留在 `docs/legacy/`，但不再作为 active demo 指南。

### Step 2: Visual evidence 默认真实模型

- 新增或改名为 `tools/run_visual_demo.py`，作为唯一 visual demo 入口。
- 默认启动临时本地 server 并在线跑全量 replay。公开入口不读取离线 artifact。
- server lifecycle 复用现有 runtime verification / CLI runner 中的 subprocess 启停模式，不新增第二套复杂进程管理框架。
- server config 使用真实 YOLO pose；缺模型直接失败。
- 在线 replay 继续调用 `tools.replay_val_data.replay_data_dir()`。
- renderer 继续复用 `tools.visual_evidence_helpers`。
- 输出 `report.json`，记录 `real_model_evidence=true`、模型路径、scene count、frame count、error count。

### Step 3: Memory demo 新公开入口

- 新增 `tools/run_memory_demo.py`。
- 公开入口不暴露 runtime 选择、历史输入或阶段性开发模式。
- 默认模型路径固定为：
  - `runtime/models/face-buffalo-s`
  - `runtime/models/scene-mobileclip2-s0`
  - `runtime/models/yolov8n-pose.pt`
- 缺模型 fail fast。
- 在本轮开发中把真实模型执行能力收敛为 `run_real_model_memory_demo()` 或等价清晰名称；用户可见报告、目录和公开函数名都不得出现阶段性开发词。
- 历史 memory evidence 脚本不再作为入口。开发团队可以删除、重命名或移动到 internal 位置。

### Step 4: Memory renderer 补齐真实模型信息

- HTML 顶部展示模型和 runtime label。
- self teach 卡片展示 face bbox、face crop、face score、known person event。
- third-person teach 卡片展示 pose pointing ray、introducer、selected target、target face crop。
- familiar unknown 卡片展示 anonymous id、seen count、observed duration、familiar score。
- scene teach 卡片展示 scene crop、scene match score、scene_activated event。
- event identity 卡片展示普通事件携带的 `identity_context`。
- current snapshot 卡片展示 agent-facing redacted 结果。

只增强现有 renderer；不新增第二套 HTML 框架、dashboard 或报告系统。

### Step 5: 测试替身留在测试内部

- active README、active docs、默认 CLI 不出现非真实模型作为 demo 做法。
- 单元测试需要可控输入时，使用 test double 或 fixture，而不是暴露为 demo 模式。
- 公开 memory demo 不读取任意历史输入；它自己生成本次真实模型结果并渲染。
- 历史 artifact 目录名不作为 active demo 入口。

### Step 6: 输出清理

- 两个生成器运行前清理自己管理的文件和目录，避免历史 scene/image 残留。
- 默认输出目录固定时，可以直接重建 `artifacts/demo/visual/` 或 `artifacts/demo/memory/`。
- 如果用户显式传 `--out`，该路径仍必须位于 `artifacts/demo/` 下，并且只清理该目录内工具管理的固定 children，例如 `images/`、`scenes/`、`crops/`、`raw/`、`index.html`、`report.json`。
- 工具不清空整个 `artifacts/`。

## 9. 测试计划

只测试核心行为，不测试测试工具本身，不做像素级 golden image。

### Unit Tests

- memory CLI 默认使用 local embedding + ultralytics。
- memory CLI 缺模型时 fail fast，错误信息包含缺失路径。
- memory CLI 不提供历史输入 / runtime 选择参数。
- visual CLI 默认走真实在线 replay orchestration。
- 两个 CLI 的输出路径必须位于 `artifacts/demo/` 下，且不能位于 `val-data/`、`runtime/` 或 Git tracked 路径内。

### Lightweight Integration

在本机模型存在时运行：

```bash
uv run --extra inference python tools/run_visual_demo.py --data-dir val-data
uv run --extra inference python tools/run_memory_demo.py --data-dir val-data
```

断言：

- `artifacts/demo/visual/index.html` 存在。
- `artifacts/demo/visual/report.json.real_model_evidence == true`。
- visual report 的 scene/frame 数与本次 `val-data` discovery 自洽，不写死固定数量。
- `artifacts/demo/memory/index.html` 存在。
- `artifacts/demo/memory/report.json.real_model_evidence == true`。
- memory report 记录本次使用的 embedding model。
- memory HTML 至少包含 self known、third-person pointing、familiar unknown、scene activated 四类 demo 摘要。

明确不做：

- 不加 Playwright。
- 不做图片 pixel diff。
- 不做 CSS / 字体 / bbox 颜色测试。
- demo report 只描述本次 demo。
- 不为 helper 或 test double 再写测试。

## 10. 验收标准

开发完成后，交付给开发团队的条件：

- README 主路径只有两个 demo 命令。
- 两个命令默认跑真实模型。
- 缺模型时失败清晰，不 fallback。
- 两个根 HTML 能直接打开并看懂对应功能。
- active docs 不再把历史测试/阶段性入口作为 demo 主路径。
- 公开命令已从 README 和 active docs 收敛为两个 demo wrapper。
- `report.json` 明确写出 real model 和模型路径。
- `git ls-files artifacts runtime val-data` 仍为 0。

## 11. 非目标

- 不开发产品级 dashboard。
- 不新增管理后台、标注平台、profile gallery 或身份纠错 UI。
- 不新增第二套 replay loop。
- 不新增第二套 renderer。
- 不新增第二套身份识别逻辑。
- 不维护历史 evidence 命令入口。
- 不支持历史 artifact 目录自动迁移。
- 不把 demo 通过解释为真机、RK3588、HIL、真实 DDS camera、真实头控闭环或现场通过。
- 不做多摄像头 ReID、通用动作识别或顾客画像平台。
- 不把 report 扩展成治理系统。

## 12. Team Review 结论

产品 review 结论：

- 两个 demo 入口是正确抽象；用户只需要知道 visual demo 和 memory demo。
- 用户可见层必须移除历史测试、阶段性和治理心智。
- demo 顶部必须明确展示真实模型来源，避免误判。
- memory demo 必须覆盖真实脸部识别、熟客识别、示教目标解析和场景记忆的完整故事，而不是只展示单帧 API。

研发 review 结论：

- 不新增复杂 runner，不新增 dashboard。
- 复用现有 replay、renderer、memory runner 能力。
- test double 只属于单元测试，不进入 demo CLI。
- 最小验证只覆盖命令默认行为、模型路径、真实模型结果和输出文件，不做像素级测试。

本计划按 KISS / DRY / YAGNI 收敛为：两个公开入口、两个固定输出目录、默认真实模型、清理用户心智、内部实现复用现有模块。
