# Visual Memory / Scene-Grounded RAG 产品候选特性

日期：2026-06-27

## 1. 背景与原始需求

本文记录一个未来可选特性：让机器人具备“被现场教学”和“基于视觉场景检索记忆”的能力。

原始需求目标是：

- 用户站在机器人摄像头前，可以通过指向场景、指向人、或直接指定当前画面，向机器人介绍信息。
- Agent 可以向视觉推理服务发送“文本 + 检测物位置”或“文本 + 整体画面”的教学指令。
- 服务端根据该指令提取对应视觉区域的特征，例如整图特征、检测框 crop 特征、人脸框特征或 embedding。
- 服务端把文本和视觉特征 pair 存储起来。
- 未来当机器人再次看到相关人、物体、区域或相似整体场景时，服务端可以检索到相关文本信息并返回给 CLI / agent。
- Agent 可以把这些检索结果作为场景驱动 RAG 上下文，让 LLM 负责解释、介绍、回忆聊天记录或面向特定用户对话。

这不是当前 GA baseline 的一部分。它应作为后续“视觉事件平台”之上的候选产品能力，等基础 detection、tracking、event、gaze 和本地 replay 测试稳定后再进入主线评估。

## 2. 产品价值

这个能力可以显著增强机器人交互的自然感，但它和现有事件检测不是同一层能力。

现有 Visual Events 负责回答：

- 眼前有没有人。
- 谁是当前注视目标。
- 这个人是否经过、靠近、停留、挥手。
- 是否应该给 agent 一个低频事件。

Visual Memory / Scene-Grounded RAG 负责回答：

- 眼前这个人、这个区域、这个场景是否和过去被教过的信息相关。
- 有没有可供 LLM 使用的背景文本。
- 当前视觉上下文是否能帮助 agent 做更自然的解释和对话。

在商店门口揽客机器人场景中，典型收益包括：

- 店员可以现场教机器人：“这个区域是新品展示区”、“这位是店长”、“这个海报对应本周活动”。
- 顾客站到某个区域前，agent 可以拿到与该区域相关的介绍资料。
- 已授权的特定用户再次出现时，agent 可以取回与该用户相关的偏好或历史聊天摘要。
- 机器人不需要训练新模型，也不需要把所有知识硬编码进规则。

## 3. 建议产品边界

第一阶段只做“视觉记忆检索”，不做自主学习和身份自动扩张。

服务端可以存储和检索：

- 全图场景 embedding + 文本。
- 指定 bbox / track crop embedding + 文本。
- 可选的、明确授权的人脸 embedding + 文本或用户 profile 引用。

服务端不应该负责：

- 决定机器人要不要开口。
- 生成最终自然语言话术。
- 操纵运控。
- 维护完整聊天系统。
- 在未授权情况下识别路人身份。
- 用视觉相似度自动创建长期人员身份。

Agent 仍然负责：

- 发起教学请求。
- 判断是否使用检索结果。
- 把检索文本组织进 LLM prompt。
- 决定最终回应、解释或动作。

## 4. 最小可行实现

低成本可行，建议分三层推进。

### 4.1 第一层：全图与区域记忆

这是最推荐的 MVP。

能力：

- Agent 发送 `memory_teach` 请求，目标为 `scene`、`bbox`、`track_id` 或 `point_uv`。
- 服务端从最近帧缓存中取图。
- 如果是 `scene`，直接提取整图 image embedding。
- 如果是 `bbox` / `track_id` / `point_uv`，裁剪相关区域并提取 embedding。
- 服务端存储 embedding、文本、时间、来源、目标类型和必要 metadata。
- 后续在稳定场景或 agent 主动请求时进行相似度检索，返回相关文本。

优点：

- 不需要训练。
- 不依赖人脸识别。
- 和现有 detection / tracking / frame cache 能自然复用。
- 风险较低，最适合先做。

### 4.2 第二层：授权用户记忆

这是高价值但高风险能力，应明确晚于第一层。

能力：

- 只有在明确授权或内部测试名单中，才允许存储人脸 embedding。
- 人脸 embedding 只用于匹配已授权用户，不用于识别未知路人。
- 检索结果可以返回用户相关文本、偏好摘要、历史聊天摘要引用。

约束：

- 默认关闭。
- 必须有删除机制。
- 必须有 retention 策略。
- 尽量不存原始人脸 crop。
- 不把 face id 暴露成可被 agent 滥用的无限身份系统。

### 4.3 第三层：指向手势理解

用户提到“通过手指场景或手指人来介绍信息”。这是真实需求，但不建议第一阶段直接做复杂手势 grounding。

第一阶段建议由 agent / CLI 明确传目标：

- `target.mode = "scene"`
- `target.mode = "track_id"`
- `target.mode = "bbox"`
- `target.mode = "point_uv"`

后续再加入真正的 pointing gesture 解析：

- 检测手腕、手指方向或手臂方向。
- 结合 person keypoints 和画面几何估计被指向对象。
- 输出一个置信度较低但可解释的目标候选。

原因是手指指向在真实摄像头中容易受姿态、遮挡、距离和镜头畸变影响。先做显式目标输入可以用较小成本验证产品价值。

## 5. 建议架构

```text
agent / Botified
  -> memory_teach / memory_query request
  -> visual-events-server
      -> recent frame cache
      -> detection / track resolver
      -> crop / scene image extractor
      -> embedding extractor
      -> vector index
      -> metadata store
      -> memory retrieval response
  -> client CLI
  -> agent RAG context
```

核心模块建议：

| 模块 | 职责 |
| --- | --- |
| Frame Cache | 保存最近 N 秒帧，用于教学请求和检索请求定位图像 |
| Target Resolver | 把 `scene`、`bbox`、`track_id`、`point_uv` 解析成图像区域 |
| Embedding Extractor | 提取 scene / crop / face embedding |
| Memory Store | 保存文本、embedding、metadata 和索引 |
| Retrieval Engine | 根据当前视觉 embedding 检索相关记忆 |
| Response Formatter | 输出给 CLI / agent 的结构化 memory context |

第一版实现不需要复杂服务拆分，全部放在 server 进程内即可。

## 6. 模型选择建议

### 6.1 场景与物体区域 embedding

建议使用 CLIP / OpenCLIP 体系的轻量模型。

理由：

- 图像和文本在同一个 embedding 空间，适合“视觉区域 + 文本”pair。
- 不需要训练即可使用。
- 模型体系成熟。
- 可先在 PC 服务端运行，未来再评估 RK3588 上的轻量化部署。

第一阶段可选：

- 使用较小的 OpenCLIP ViT-B 或 MobileCLIP / SigLIP 类轻量模型。
- 先不追求最强语义理解，只验证教学与检索闭环。

### 6.2 人脸 embedding

如果后续做授权用户记忆，可考虑 InsightFace / ArcFace 类模型。

但注意：

- 这属于生物特征处理。
- 产品和合规风险明显高于普通视觉场景 embedding。
- 不应作为默认能力进入第一版。

## 7. 数据存储建议

第一版保持简单：

- SQLite 存 metadata、文本、来源、时间、target 类型、权限字段。
- FAISS 存本地向量索引。
- 文件目录只存必要的 index 文件和数据库文件。

后续如果需要多机器人、多租户、远程管理或更复杂 metadata filter，再考虑 Qdrant 等向量数据库。

不建议第一版直接上复杂数据库服务，因为当前目标是验证产品闭环，不是建设知识平台。

## 8. 请求与响应草案

### 8.1 教学请求

```json
{
  "type": "memory_teach",
  "request_id": "req-001",
  "timestamp_ms": 1780000000000,
  "text": "这是本周新品展示区，主推夏季轻量外套。",
  "target": {
    "mode": "scene"
  },
  "scope": {
    "store_id": "store-a",
    "robot_id": "robot-01"
  },
  "policy": {
    "visibility": "local_store",
    "retention_days": 90
  }
}
```

区域教学：

```json
{
  "type": "memory_teach",
  "request_id": "req-002",
  "text": "这个人是店长，可以优先通知他处理现场问题。",
  "target": {
    "mode": "track_id",
    "track_id": 12,
    "feature_type": "face",
    "consent": "explicit"
  }
}
```

### 8.2 检索响应

```json
{
  "type": "memory_context",
  "timestamp_ms": 1780000000500,
  "query": {
    "mode": "scene",
    "source_frame_id": "frame-123"
  },
  "matches": [
    {
      "memory_id": "mem-001",
      "score": 0.82,
      "text": "这是本周新品展示区，主推夏季轻量外套。",
      "target_type": "scene",
      "created_at_ms": 1779999900000,
      "metadata": {
        "store_id": "store-a",
        "robot_id": "robot-01"
      }
    }
  ]
}
```

Agent 只应把这些结果作为候选上下文，不应把检索结果当成事实绝对正确。

## 9. 触发策略

检索不应该每帧触发。

建议触发条件：

- 新的稳定 attention target 出现。
- `person_stopped_near_robot` confirmed。
- Agent 主动请求当前场景上下文。
- 用户发起教学后立即回查验证。
- 场景明显变化且距离上次检索超过 cooldown。

默认频率建议：

- 低频 memory query，例如 0.2Hz 到 1Hz。
- 每个 track / scene 有 cooldown。
- 结果返回给 agent 前做相似度阈值过滤和数量限制。

## 10. 隐私与安全边界

这个特性最重要的风险不是技术实现，而是隐私边界。

必须明确：

- 普通场景记忆和人脸身份记忆要分开。
- 默认不做人脸长期记忆。
- 未授权顾客不能被创建身份 profile。
- 人脸 embedding 也应视为敏感生物特征数据。
- 必须支持删除用户相关记忆。
- 应有过期时间和本地存储策略。
- 返回给 agent 的内容应尽量是业务文本，不是底层向量或身份内部 ID。

对商店门口机器人，第一版最好只启用：

- 店铺场景记忆。
- 商品区域记忆。
- 活动海报或陈列区域记忆。
- 内部员工授权记忆。

不要默认启用：

- 路人身份识别。
- 顾客无感知 face profile。
- 跨天追踪未知用户。

## 11. 与现有 Visual Events 的关系

Visual Events 仍然保持 KISS：

- detection / tracking / event / gaze 是实时感知主链路。
- low-frequency event 通过 Botified frame 给 agent。
- high-frequency gaze 通过 DDS 给运动侧或注意力侧。
- memory retrieval 是附加上下文，不阻塞主感知链路。

Visual Memory 不应改变事件规则的核心职责。它可以订阅或复用：

- 当前 frame。
- track metadata。
- bbox / keypoints。
- scene context。
- stable attention target。

但它不应该反过来影响 detection、tracking 和 gaze 的实时稳定性。

## 12. 测试与验证

第一版应支持 PC 上完全本地化端到端测试：

- 使用 val-data 或辅助数据 replay 图像流。
- 使用测试工具模拟 agent 发送 `memory_teach`。
- 验证服务端能从正确帧中解析 scene / bbox / track。
- 验证 embedding 和 metadata 被写入本地 store。
- 验证后续相似场景能检索到文本。
- 验证不相似场景不会返回高置信结果。
- 验证 memory retrieval 不影响 10Hz visual event 输出。

建议测试用例：

| 用例 | 验证点 |
| --- | --- |
| 全图教学后回放同一场景 | 能检索到对应文本 |
| bbox 区域教学后回放相似区域 | 能检索到对应文本 |
| track_id 教学但 track 已丢失 | 返回明确错误，不写入错误记忆 |
| 低相似度场景查询 | 不返回或低分返回 |
| 未授权 face memory 请求 | 拒绝写入 |
| 删除 memory 后查询 | 不再返回 |
| 高频事件运行时查询 memory | 不阻塞主链路 |

## 13. 开发阶段建议

建议后续如果进入主线，按以下顺序实现：

1. 定义 `memory_teach` / `memory_query` / `memory_context` 数据结构。
2. 在 server 内实现 recent frame cache 查询能力。
3. 实现 target resolver，先支持 `scene`、`bbox`、`track_id`。
4. 接入一个轻量图文 embedding 模型。
5. 实现 SQLite + FAISS 本地 memory store。
6. 写一个 PC 本地测试工具，模拟 agent 教学和查询。
7. 用 val-data replay 做端到端验证。
8. 加入 similarity threshold、top_k、cooldown 和 allowlist。
9. 再评估是否加入授权 face memory。
10. 最后再评估 pointing gesture grounding。

## 14. 非目标

本候选特性不应在第一阶段做：

- 不训练新模型。
- 不建设大型知识库管理后台。
- 不做复杂多租户权限系统。
- 不做未授权人脸识别。
- 不做跨店铺顾客追踪。
- 不让 memory retrieval 决定机器人动作。
- 不让 LLM 直接写入长期记忆，必须由 agent 发明确教学请求。
- 不把所有历史聊天原文直接塞入视觉服务。

## 15. Review 结论

需求可以理解，并且可以用较小成本做出一个有价值的 MVP。最稳妥的产品路径是先做“全图 / 区域视觉记忆 + 文本检索”，验证场景驱动 RAG 是否能提升机器人讲解和互动质量。

人脸特征与特定用户聊天记录检索在技术上可行，但隐私和产品边界风险更高，必须作为显式授权的后续能力。真正通过手指方向自动判断目标也建议后置，第一阶段用 agent 明确传 `scene`、`bbox`、`track_id` 或 `point_uv`，以最低成本验证完整闭环。
