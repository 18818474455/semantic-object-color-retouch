# 语义物体调色专家 · 开发方案 V3（执行版）

> 本文档基于 `semantic-object-color-retouch-dev-plan-v2.md` 重新整理。V3 的目标是把方案从“能力设想”收束为“可开发、可验收、可蒸馏”的服务端工程方案。
>
> **2026-07-09 更新**：Phase C 拆为 C1（GPT 量化）∥ C2（Reference 自蒸馏，主路径）。请同时阅读：  
> - `semantic-object-color-retouch-dev-plan-v3.1-c2-addendum.md`  
> - `phase-c2-reference-self-distill-design.md`

---

## 0. 一句话定位

构建一个服务端语义调色系统：先识别照片中的物体和区域，再判断每个区域该不该调、怎么调、由哪个执行器调，最终输出多套可执行调色方案。

核心不是单纯判断“整张图偏黄”，而是做到：

```text
识别天空、皮肤、草地、衣服、LED 屏、建筑、文字/logo 等区域
-> 判断区域色彩问题和保护优先级
-> 从冻结 action 词表中选择调色动作
-> 路由到本地 C++ 渲染器或 GPT Image 2
-> QA 不通过自动降级或转人工
```

---

## 1. 关键工程决策

| 决策项 | V3 结论 |
|---|---|
| 部署形态 | 只做服务端批处理，面向影楼、活动、批量交付，不做端侧 |
| 产品目标 | 语义物体调色专家，不训练像素生成模型 |
| 执行器 | 双执行器：本地 C++ 渲染器 + GPT Image 2 |
| 默认执行 | 默认走本地 C++ 渲染器，确定、低成本、可批量 |
| GPT Image 2 定位 | 正式执行器，用于宽容度恢复、清晰度提升、疑难混合光，不再只是探索工具 |
| 蒸馏目标 | `image -> semantic_color_plan.json`，包含区域诊断、action、strength、executor 路由 |
| action 设计 | 冻结封闭词表，VLM/学生模型只能做选择题 |
| 本地引擎 | 复用 BeautySDK `pe_process_image` / ColorCore v2 / Smart Color v2 参数体系 |
| 区域能力 | Stage 0 可用 Python 离线验证 mask 混合，C++ 区域混合能力并行开发 |
| 兜底原则 | 置信度不足默认 `keep/no-edit`，宁可不动，不动错 |

---

## 2. 总体架构

```text
照片输入
  -> 数据清理与元信息
  -> 感知层：VLM + Grounding DINO + SAM2 + 人脸/皮肤资产
  -> 区域清单：object_type / bbox / mask / confidence / protect_level
  -> 区域色彩指标：LAB / HSV / 亮度分位 / 裁切比例 / 清晰度
  -> 语义调色规划：冻结 action 词表内选择
  -> 执行路由：本地 C++ / GPT Image 2 / 两段式
  -> 自动 QA：目标区域、保护区、人脸、OCR、结构一致性
  -> 人工偏好反馈
  -> 蒸馏学生规划模型
```

系统分为五个服务模块：

1. **Perception Service**  
   负责物体检测、分割、区域清单生成。

2. **Color Metrics Service**  
   负责全图和区域内客观色彩指标计算。

3. **Planner Service**  
   负责语义诊断、action 选择、方案生成、executor 路由。

4. **Execution Service**  
   负责调用本地 C++ 渲染器、GPT Image 2 或两段式执行。

5. **QA & Preference Service**  
   负责自动质检、降级策略、人工评审和训练数据回流。

---

## 3. 数据输入与清理

当前原图目录已初步体检：

```text
真实可用 JPG：20,562 张
macOS 资源叉文件：21,894 个 `._*`，必须排除
主要分辨率：长边 2000-4000
已有阿里云人脸检测 manifest：7156 条人脸记录，覆盖 1294 张图片
```

输入清理规则：

- 排除所有 `._*`
- 排除小于阈值的异常图片
- 统一 EXIF orientation
- 保留原图路径，不移动原图
- 为训练/验证生成干净 `manifest.jsonl`

manifest 示例：

```json
{
  "image_id": "000001",
  "source_path": "/Volumes/未命名/大模型/原图1/DAP01805.JPG",
  "width": 4000,
  "height": 2667,
  "split": "stage0",
  "has_face": true
}
```

---

## 4. 感知层设计

### 4.1 教师栈

| 模块 | 推荐 | 职责 |
|---|---|---|
| VLM | **Qwen3-VL-8B-Instruct**（2026-07-09 核实：ModelScope 官方仓库，Apache-2.0，商用无门槛，见下方更新说明） | 全局场景理解、语义判断、门控 |
| 检测 | Grounding DINO / Grounding DINO 1.5 | 按文本提示找天空、草地、衣服、建筑、LED 屏等 |
| 分割 | SAM2 | 将检测框/点提示转成区域 mask |
| 人脸/皮肤 | BeautySDK 现有资产 + 阿里云人脸结果 | 人脸、皮肤区域优先使用现成高可信资产 |
| 色彩指标 | OpenCV / Pillow / scikit-image | 计算全图和 mask 内客观指标 |

> **2026-07-09 更新（VLM 选型落地）**：本节 §4.2 从方案定稿起就要求"VLM 认为该物体在图中存在"作为交叉验证门槛之一，但 `scripts_m2/region_provider_v2.py` 实际实现至今只有**启发式规则**（纹理+饱和度双指标判断"天空合理性"），没有接入任何真实 VLM。ModelScope 图文多模态盘点（`09-行业方案与知识库/App/2026-07-09-ModelScope图文多模态VLM模型盘点.md`）核实了 Qwen3-VL 全系列（2B~32B，含 Instruct/Thinking）License 为 **Apache-2.0，完全商用无门槛**，于是做了 C1c 小实验：用托管的 `qwen3-vl-plus`（通过 API易，本机 16GB 内存跑不动自部署的 8B 权重）复核了 C2 数据集里全部 30 个"启发式认定合理"的天空样本。
>
> **实验结论（详见 `phase-c1c-vlm-sky-gate-results.md`）**：Qwen3-VL 与启发式规则 100% 一致（0 个语义假阳性），包括最初被怀疑是"假天空检测"的 bug 案例——人工复核确认那其实是真实过曝天空，bug 是数值拟合层面的，不是感知层的语义误判。**C1c 原本"替代启发式规则"的动机没有得到数据支持，优先级下调**；§12.0 C1c 状态已更新。

### 4.2 交叉验证原则

进入区域清单的物体必须同时满足：

- VLM 认为该物体在图中存在
- 检测器给出合理 bbox
- SAM2 或现有分割资产能生成可用 mask

例外：

- 人脸/皮肤优先信任现有人脸检测与皮肤分割资产
- OCR 文本/logo 优先用于保护，不一定参与调色

低置信度处理：

```text
region_confidence < 0.6 -> 不做主动调色，只加入保护或忽略
diagnosis_confidence < 0.6 -> action = keep
mask_quality 不合格 -> 禁止局部调色，可只做全局方案
```

### 4.3 区域清单 Schema

```json
{
  "image_id": "000001",
  "scene_type": "outdoor_event_portrait",
  "regions": [
    {
      "region_id": "r_sky_001",
      "object_type": "sky",
      "role": "background",
      "bbox": [0.0, 0.0, 1.0, 0.32],
      "mask_path": "masks/000001/r_sky_001.png",
      "confidence": 0.91,
      "mask_quality": 0.86,
      "editability": "high",
      "protect_level": "medium"
    },
    {
      "region_id": "r_skin_001",
      "object_type": "skin",
      "role": "primary_subject",
      "bbox": [0.42, 0.22, 0.61, 0.56],
      "mask_path": "masks/000001/r_skin_001.png",
      "confidence": 0.88,
      "mask_quality": 0.82,
      "editability": "medium",
      "protect_level": "high"
    }
  ]
}
```

---

## 5. 区域色彩指标

每个区域必须生成客观指标，规划器不能只靠 VLM 主观描述。

### 5.1 指标列表

全图与每个 mask 内都计算：

- RGB 均值和分位数
- LAB 均值、a/b 偏移
- HSV 色相/饱和度/亮度分布
- 亮度 1/5/50/95/99 分位
- `clip_high_pct`
- `clip_low_pct`
- saturation / colorfulness
- 局部对比度
- sharpness_proxy
- white-balance proxy
- green/magenta cast proxy

### 5.2 routing_metrics

V3 明确把执行器路由所需指标纳入诊断输出：

```json
{
  "routing_metrics": {
    "clip_high_pct": 6.2,
    "clip_low_pct": 0.4,
    "sharpness_proxy": 0.31,
    "mixed_light_score": 0.78,
    "local_preview_quality": 0.62
  }
}
```

这些字段直接影响是否走 GPT Image 2。

---

## 6. 冻结 Action 词表

### 6.1 设计原则

- 先冻结词表，再开始标注
- VLM 和学生模型只能选择枚举，不自由生成动作
- 每个 action 绑定两套模板：
  - 本地 C++ 渲染器参数模板
  - GPT Image 2 提示词片段
- action 不只描述“颜色”，还描述“语义合法性”

### 6.2 词表 V1.0 草案

```text
sky:
  keep
  slight_clean
  natural_daylight_blue
  deep_blue
  keep_overcast_mood
  keep_sunset
  keep_night

skin:
  keep
  remove_yellow_green_cast
  remove_red_orange_cast
  clean_natural
  warm_healthy

grass_tree:
  keep
  fresh_natural_green
  reduce_yellow_cast
  darker_cinematic_green

water:
  keep
  cleaner_blue_cyan

clothing:
  keep
  neutralize_white

led_stage:
  keep
  reduce_face_color_pollution

food:
  keep
  warm_appetizing

global:
  none
  white_balance_correct
  exposure_lift
  exposure_reduce
  contrast_soften
  contrast_boost
  latitude_recovery
  clarity_boost
```

### 6.3 语义门控规则

关键规则：

- `sky.natural_daylight_blue` 只允许在日间天空、灰蓝/低饱和天空、非日落、非夜景、非舞台背景时出现
- 舞台、LED、演出照片默认不强行中性化环境色，只处理人物脸部污染
- 皮肤最高保护级，禁止塑料皮、禁止改变脸型和肤质
- 衣服、logo、文字默认保护
- 食物可偏暖，但禁止灰、脏、过冷
- `global.latitude_recovery` 和 `global.clarity_boost` 只能路由到 GPT Image 2
- 低置信度统一 `keep`

### 6.4 Action 模板示例

```json
{
  "action": "sky.natural_daylight_blue",
  "local_params": {
    "engine": "pe_process_image_v2",
    "region_blend": {
      "mask": "sky_mask",
      "feather_px": 8,
      "edge_protect": true
    },
    "hsl_blue": {
      "hue": 6,
      "sat": 18,
      "lum": -4
    },
    "temp_shift": -300
  },
  "gpt_prompt_fragment": "Only adjust the sky region. Make the sky a natural daylight blue with realistic brightness and soft saturation. Preserve cloud shape, building edges, trees, horizon, and the original lighting direction."
}
```

---

## 7. 方案生成器

每张图输出 3-5 个方案，但所有方案必须由冻结 action 组合而来。

### 7.1 方案类型

| 方案 | 目标 | 默认执行器 |
|---|---|---|
| `safe_natural` | 自然白平衡、肤色干净、低风险批量交付 | 本地 C++ |
| `commercial_clean` | 更亮、更干净、更讨喜，适合活动/家庭/学校交付 | 本地 C++ |
| `object_enhancement` | 对天空、草地、白衣等目标区域做语义强化 | 本地 C++，失败升级 GPT |
| `latitude_recovery` | 找回高光/暗部、提升清晰度和宽容度 | GPT Image 2 |
| `restore_original_intent` | 舞台、夜景、日落保留氛围，只去污染 | 本地 C++ 为主 |

### 7.2 方案 JSON

```json
{
  "image_id": "000001",
  "plans": [
    {
      "plan_id": "p1",
      "name": "safe_natural",
      "executor": "local_cpp",
      "risk": "low",
      "routing_reason": "color and tone only; no severe clipping",
      "two_stage": false,
      "region_actions": [
        {
          "region_id": "r_skin_001",
          "object_type": "skin",
          "action": "skin.remove_yellow_green_cast",
          "strength": 0.45,
          "confidence": 0.86
        },
        {
          "region_id": "global",
          "object_type": "global",
          "action": "global.white_balance_correct",
          "strength": 0.35,
          "confidence": 0.91
        }
      ],
      "preserve": [
        "identity",
        "skin_texture",
        "clothing_color",
        "logos",
        "text",
        "background_structure",
        "composition"
      ]
    },
    {
      "plan_id": "p4",
      "name": "latitude_recovery",
      "executor": "gpt_image_2",
      "risk": "medium",
      "routing_reason": "clip_high_pct=6.2 exceeds threshold and sharpness_proxy=0.31 is low",
      "two_stage": true,
      "local_pre_correction": {
        "white_balance": "auto",
        "exposure": 0.3
      },
      "region_actions": [
        {
          "region_id": "global",
          "object_type": "global",
          "action": "global.latitude_recovery",
          "strength": 0.7,
          "confidence": 0.88
        }
      ],
      "gpt_image_prompt": "[lock_block] + [action_fragments] + [quality_block]"
    }
  ]
}
```

---

## 8. 双执行器与路由

### 8.1 本地 C++ 渲染器

默认执行器。

适合：

- 白平衡校正
- 曝光/对比度/饱和度调整
- HSL 局部调色
- 轻度蓝天、绿草、白衣校正
- 大批量低成本交付

约束：

- 只能重新分布原图已有信息
- 不能恢复死白高光和死黑暗部真实细节
- 当前区域 mask 混合能力待补齐

Stage 0 临时实现：

```text
用 Python 离线复现 pe_process_image 参数 + mask 混合
-> 验证局部调色效果
-> 并行开发 C++ 区域混合能力
```

### 8.2 GPT Image 2

正式第二执行器。

适合：

- 大面积高光/暗部信息丢失
- 画质和清晰度不足
- 舞台/LED/混合光疑难图
- 本地参数化调色看起来假的结果
- 两段式本地校正后再恢复宽容度和清晰度

约束：

- 生成式结果有方差
- 必须强提示词锁定
- 必须逐图 QA
- QA 不过自动降级本地结果

### 8.3 路由规则

默认：

```text
executor = local_cpp
```

升级 GPT Image 2 条件：

```text
clip_high_pct > threshold_high
clip_low_pct > threshold_low
sharpness_proxy < threshold_sharpness
mixed_light_score > threshold_mixed_light
local_preview_quality < threshold_preview
action 包含 global.latitude_recovery 或 global.clarity_boost
```

两段式：

```text
local_pre_correction -> GPT Image 2
```

用于：

- 白平衡明显偏、但又需要宽容度恢复
- 本地可先把色彩拉到合理范围，再交给 GPT 做细节与质感

---

## 9. GPT Image 2 强锁定提示词模板

所有 GPT Image 2 请求必须使用三段式模板，规划器只能填充中间 action 片段。

### 9.1 固定锁定段

```text
This is a photo retouching task, NOT image generation.
Strictly preserve: person identity, facial features, face shape, skin texture,
pores, expression, age, hair, body shape, pose, clothing design and brand colors,
logos, text and signs, background structure, object shapes, composition, and framing.
Do not add, remove, move, or replace any object. Do not beautify or slim faces/bodies.
```

### 9.2 Action 操作段

由 action 模板拼接生成，例如：

```text
Only adjust the sky region. Make the sky a natural daylight blue with realistic brightness and soft saturation. Preserve cloud shape, building edges, trees, horizon, and the original lighting direction.
Correct yellow-green color pollution on skin gently while preserving natural skin texture.
```

### 9.3 画质段

按路由原因选择：

```text
Recover highlight and shadow detail naturally, as if the photo was shot with higher dynamic range. Improve overall clarity and micro-contrast without halos, artificial HDR look, or over-sharpening.
```

禁用自由提示词：

- 不允许 VLM 直接写完整 GPT prompt
- 不允许用户方案绕过锁定段
- 不允许出现“change face / beautify / replace / generate new sky objects”等高风险词

---

## 10. QA 系统

### 10.1 本地输出 QA

检查：

- 目标 mask 内颜色变化是否达到 action 预期
- 保护 mask 内 ΔE 是否超限
- mask 边缘是否有光晕、断裂、脏边
- 整体曝光和色彩是否越界

### 10.2 GPT 输出加强 QA

| 检查项 | 阈值/规则 |
|---|---|
| 人脸身份保持 | face embedding similarity ≥ 0.92 |
| 皮肤纹理 | 高频能量比 ≥ 0.85 |
| 文字/logo | OCR 一致率 ≥ 0.95 |
| 保护区色差 | ΔE ≤ 3 |
| 结构一致性 | 无新增、消失、位移物体 |
| 目标区域变化 | ΔE / hue / saturation 落入 action 预期区间 |
| 边缘伪影 | 天空/树/建筑边缘无明显光晕或断裂 |

### 10.3 降级策略

```text
GPT QA 通过 -> 使用 GPT 输出
GPT QA 不通过 -> 自动降级本地输出
本地输出也不通过 -> 标记人工复核
规划低置信度 -> no-edit
```

QA JSON 示例：

```json
{
  "image_id": "000001",
  "plan_id": "p4",
  "executor": "gpt_image_2",
  "target_success": true,
  "face_similarity": 0.94,
  "skin_texture_ratio": 0.89,
  "ocr_consistency": 0.98,
  "protected_delta_e": 2.1,
  "artifact_risk": "low",
  "pass": true,
  "fallback_used": false
}
```

---

## 11. 数据集与训练目标

### 11.1 不训练像素生成

V3 训练目标明确为：

```text
image -> semantic_color_plan.json
```

学生模型预测：

- scene_type
- object/region 重要性
- 区域 color_state
- action 枚举
- strength
- executor 路由
- plan 排序
- no-edit 判断

### 11.2 数据目录

```text
dataset/
  images/
  masks/
  metrics/
  regions/
  teacher_labels/
  plans/
  edited_candidates/
  qa/
  preferences/
  manifest.jsonl
```

### 11.3 teacher label 示例

```json
{
  "image_id": "000001",
  "scene": {
    "type": "outdoor_event_portrait",
    "lighting": "daylight_overcast",
    "mood": "documentary"
  },
  "regions": [
    {
      "region_id": "r_sky_001",
      "object_type": "sky",
      "confidence": 0.91,
      "mask_path": "masks/000001/r_sky_001.png",
      "color_state": "dull_gray",
      "action": "sky.natural_daylight_blue",
      "strength": 0.7,
      "executor_preference": "local_cpp"
    }
  ],
  "routing": {
    "executor": "local_cpp",
    "upgrade_to_gpt": false,
    "reason": "no severe clipping"
  },
  "plans": [],
  "protected_objects": ["faces", "skin", "clothing", "text", "logos"],
  "reject_conditions": [
    "identity changed",
    "skin became plastic",
    "logo/text changed",
    "sky looks fake",
    "object structure changed"
  ]
}
```

---

## 12. 蒸馏路线

> **V3.1（2026-07-09）**：本节被双轨 Phase C 扩展。C2 Reference 自蒸馏为主路径，不依赖 GPT API。  
> 详见 `semantic-object-color-retouch-dev-plan-v3.1-c2-addendum.md` 与 `phase-c2-reference-self-distill-design.md`。

### 12.0 Phase C 三轨概览（2026-07-09 新增 C1c，同日完成小实验并下调优先级）

| 轨道 | 名称 | 依赖 | 产出 |
|------|------|------|------|
| **C1** | GPT teacher 量化 | API易 `gpt-image-2-all` | per-class Lab 残差（hard-case 标注） |
| **C1c**（实验已完成，优先级下调） | 本地/托管 VLM 语义门控/critic | Qwen3-VL（API易代理的 `qwen3-vl-plus`，本机 16GB 内存跑不动自部署 8B） | 30 样本对比实验：0 语义假阳性，见 `phase-c1c-vlm-sky-gate-results.md` |
| **C2** | Reference 自蒸馏 ★ | 本地 pseudo-target + Smart Color v2 | RegionalParamHead 权重 |

C2 teacher v0 = `color_reference_transfer.py` medium 档；数据从 20 图回归集 bootstrap。

**C1c 结论（2026-07-09 更新）**：最初动机是"§4.2 一直要求 VLM 交叉验证但只有启发式规则实现，怀疑这是 C2 训练数据里那个'假天空'bug 的根因"。用 `qwen3-vl-plus` 对 C2 数据集里全部 30 个"启发式认定合理"的 sky 样本做了对比实验，**结果 100% 一致，0 个语义假阳性**——包括最初怀疑的 bug 案例，人工复核确认那其实是真实过曝天空（数值拟合问题，不是语义识别问题）。**结论：当前数据集下启发式规则可靠，C1c"替代规则"的动机不成立，优先级下调为次要备选**（仍可作为"不依赖 GPT API 的本地/托管 teacher"选项保留，但同一次实验也观察到 `qwen3-vl-plus` 对同一张图 5 次请求全部超时/断连，说明它也不是绝对可靠，这个附带价值也没有被验证）。

---

规模：100 张。

分布：

```text
30 户外/天空
30 人物/活动
20 舞台/LED/混合光
20 暗光/过曝/困难图
```

执行：

1. 生成 clean manifest
2. 跑检测 + SAM2 + 人脸/皮肤 mask
3. 计算区域色彩指标
4. VLM 在冻结 action 词表内选择动作
5. 本地执行器生成 2-3 套方案
6. GPT Image 2 跑 30-40 张，包括所有 `latitude_recovery` 路由图
7. 自动 QA
8. 人工评审：选最好方案、标注不该动却动了、记录 GPT 人物变动

验收线：

```text
mask 可用率 ≥ 80%
语义门控错误率 ≤ 5%
至少一个可用方案的图片占比 ≥ 70%
GPT Image 2 人脸 QA 通过率 ≥ 90%
```

达标后：

- 修订 action 词表
- 锁定 V1.0
- 进入 Stage 1

### Stage 1：伪标签数据集

规模：2,000-5,000 张。

建议分布：

```text
1000 随机
500 人物密集
300 户外/天空/绿植
200 暗光/舞台/LED
其余按失败类型补样
```

输出：

- masks
- region metrics
- teacher labels
- plans
- edited candidates
- QA
- preferences

### Stage 2：学生规划模型

推荐 Hybrid，不先做小 VLM 自由 JSON。

```text
Grounding DINO + SAM2 保留为感知层
+ SigLIP / DINOv2 / ConvNeXt 视觉编码器
+ 结构化输出头
+ action 模板引擎
+ 本地/GPT 路由器
```

预测目标：

- 每区域 object_type
- color_state
- action
- strength
- confidence
- executor
- plan ranking

验收：

```text
学生 action 与教师一致率 ≥ 85%
executor 路由一致率 ≥ 90%
学生方案人工偏好不劣于教师方案的 80%
推理成本显著低于教师栈
```

---

## 13. 与 BeautySDK / Smart Color v2 的关系

本方案是 Smart Color v2 的语义区域叠加层。

推荐链路：

```text
现有全局 Smart Color
-> 语义区域微调
-> QA
```

约束：

- 不改 v4.3.9 `SmartColorGrading` rules fallback
- 不改现有全局模型训练链路
- 本地执行复用 `PARAM_PROTOCOL.md` 参数族
- 区域 mask 混合作为渲染器扩展能力独立开发
- 规划模型与 Smart Color 全局模型是两个模型、两个交付物，可独立回滚
- 需要同步补充到 `beauty_sdk/docs/smart_color_v2/PLAN.md`

---

## 14. 工程里程碑

| 里程碑 | 交付物 | 验收 |
|---|---|---|
| M1 数据清理与验证集 | clean manifest、100 张验证集、抽样图墙、基础色彩指标 | 四类场景齐全，图片加载正常 |
| M2 感知与 mask | object prompt 清单、检测框、SAM2 mask、mask 可视化、区域指标 | mask 可用率 ≥ 80%，误检有日志 |
| M3 action 与方案生成器 | action 词表 V1.0、双模板映射、plan JSON、路由规则 | 低置信度 no-edit；蓝天 action 只在语义合法时出现 |
| M4 双执行器 | 本地对比图、Python mask 混合验证、GPT Image 2 30-40 张冒烟 | Stage 0 四条验收线达标 |
| M5 QA 闭环 | QA JSON、自动降级、人工评审表、失败类型统计 | GPT 不合格可自动降级，本地失败可转人工 |
| M6 伪标签数据集 | 2,000-5,000 张 teacher labels / masks / plans / preferences | 标签格式稳定，失败样本可追踪 |
| M7 学生规划模型 | 训练代码、模型、评估报告、服务端推理接口 | Stage 2 验收线达标 |

---

## 15. 成本控制

策略：

- 教师 VLM 先跑压缩图
- 检测/分割尽量本地跑
- 所有中间结果缓存
- 本地渲染器全量出预览
- 只把路由命中或人工选中的方案送 GPT Image 2
- Stage 0 GPT 控制在 30-40 张
- Stage 1 GPT 控制在路由命中子集，预计 15-25%
- 商业 VLM 只用于金标和难例
- 学生模型上线后，教师栈只用于抽检和新场景

---

## 16. 风险与应对

| 风险 | 应对 |
|---|---|
| GPT Image 2 改人物或背景 | 强锁定提示词 + 高保真输入 + 人脸/OCR/结构 QA + 自动降级 |
| VLM 幻觉物体 | VLM + 检测框 + mask 三方确认 |
| 蓝天调假 | 天气/日落/夜景/舞台语义门控，低置信度 keep |
| 皮肤变塑料 | 皮肤保护规则 + 高频纹理 QA |
| 衣服/logo/文字被改 | 保护 mask + OCR 一致率 + ΔE 限制 |
| 局部调色边缘光晕 | mask feathering + edge QA，不合格升级 GPT 或人工 |
| 渲染器暂不支持区域混合 | Stage 0 Python 离线验证，C++ 扩展并行 |
| 数据集偏活动/舞台 | Stage 1 按场景补齐天空、户外、绿植、暗光等分布 |
| action 词表过大失控 | Stage 0 后只修订一次，之后冻结 V1.0 |

---

## 17. 立即执行清单

1. 从当前 20,562 张 JPG 中生成 clean manifest，排除 `._*`
2. 挑出 100 张 Stage 0 验证集，按 30/30/20/20 分布
3. 确认 action 词表 V1.0 草案，补齐每个 action 的本地参数模板和 GPT 片段
4. 搭建感知管线：Grounding DINO + SAM2 + 已有人脸/皮肤资产
5. 计算全图和 mask 内色彩指标，生成 region JSON
6. 生成 3-5 个 plan JSON，默认本地路由
7. 用 Python 先验证 mask 混合局部调色
8. 选 10 张做 GPT Image 2 冒烟，其中包含高光裁切、暗部死黑、混合光、人物脸部保护
9. 建立 QA JSON 和人工评审表
10. 如果 Stage 0 四条量化线达标，进入 2,000-5,000 张伪标签阶段

---

## 18. V3 成功标准

Stage 0 成功不是“模型训练完成”，而是证明这条链路商业上值得继续：

```text
系统能稳定识别关键物体和区域
能判断哪些区域该调、哪些必须保护
能用冻结 action 生成可执行调色方案
本地渲染器能覆盖大部分常规图
GPT Image 2 能在宽容度/清晰度疑难图上提供明显增益
QA 能及时拦截人物、文字、结构变化
人工评审认为至少 70% 图片有一个可用方案
```

达到这些标准后，再训练学生规划模型才有意义。
