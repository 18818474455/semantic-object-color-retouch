# 语义物体调色专家 · 开发方案 V2（服务端版）

> 本文档是 `semantic-object-color-retouch-dev-plan.md` 的优化版。
> 主要变更：明确服务端部署；GPT Image 2 升级为正式双执行器之一（实测强提示词锁定下人物变动可控，核心价值是突破原图宽容度与清晰度上限）；本地调色联动现有 BeautySDK C++ 渲染器；action 词表先冻结再标注；补充量化验收线与 Smart Color v2 的层级关系。

---

## 0. 关键决策（先读）

| 决策项 | 结论 |
|---|---|
| 部署形态 | **服务端批处理**（影楼/活动交付场景），不做端侧 |
| 执行器 | 双执行器：本地 C++ 渲染器（颜色/影调）+ GPT Image 2（宽容度/清晰度/疑难图） |
| GPT Image 2 定位 | 正式执行器，不只是探索工具。前提：强提示词锁定 + 逐图 QA |
| 本地调色引擎 | 复用 BeautySDK `pe_process_image` / ColorCore v2 参数体系，新增区域 mask 混合，不用 OpenCV 另造一套 |
| 蒸馏目标 | 先蒸馏「语义调色规划模型」（image → plan JSON），不训练像素生成 |
| action 词表 | 冻结的封闭枚举，每个 action 绑定渲染器参数模板，VLM 只做选择题 |
| 与 Smart Color v2 关系 | 叠加层：先全局 Smart Color，再区域微调。不动 v4.3.9 rules fallback 和全局模型链路 |
| 兜底原则 | 置信度不足时默认 no-edit，宁可不动也不动错 |

---

## 1. 产品目标

构建一个语义调色专家：不只判断"整张偏黄/欠曝"，而是理解画面中的物体和区域，对每个重要区域分别决策——**识别物体 + 判断该不该调 + 给出局部/全局调色方案**。

示例：

```text
输入照片：
- 天空灰暗发闷
- 人物肤色偏黄
- 草地墨绿发暗
- 白衣服偏暖

输出方案：
- Plan A：safe_natural 自然校正
- Plan B：commercial_clean 商业干净色
- Plan C：latitude_recovery 宽容度找回（走 GPT Image 2）

执行：
- 天空转自然日光蓝，保留云层、建筑边缘、地平线
- 肤色去黄绿污染，保身份、保纹理
- 衣服/logo 颜色保真
```

核心判断逻辑：**不是看到天空就变蓝**。夜景、日落、舞台背景、阴天纪实照强行蓝天会假。模型先判断场景语义，再决定调色策略；拿不准就不动。

最终链路：

```text
照片
-> 物体/区域识别（Grounding DINO + VLM 交叉验证）
-> 分割 mask（SAM2 + BeautySDK 现有人脸/皮肤分割）
-> 每区域色彩指标（客观量化）
-> 语义调色诊断（冻结词表内选择）
-> 3-5 个调色方案
-> 路由：本地 C++ 渲染器 / GPT Image 2
-> 自动质检 QA
-> 人工偏好反馈
-> 蒸馏学生规划模型
```

---

## 2. 双执行器架构（V2 核心变更）

### 2.1 分工原则

两个执行器解决的是两类本质不同的问题：

| | 本地参数化调色（Mode A） | GPT Image 2 语义编辑（Mode B） |
|---|---|---|
| 引擎 | BeautySDK C++ `pe_process_image` / ColorCore v2 + 区域 mask 混合 | gpt-image-2-all（现有 APIYi 集成，服务端代理） |
| 能力边界 | 只能重分布原图已有信息：白平衡、影调、HSL、区域色偏 | 可以**再生成**原图没有的信息：死白高光找回、死黑暗部细节、清晰度提升、混合光疑难 |
| 确定性 | 完全确定，同参数同结果 | 生成式，逐图有方差，必须 QA |
| 成本 | 零边际成本，可全量批处理 | 按张计费，只发选中的图 |
| 人物风险 | 无（像素级只调色） | 实测强提示词锁定下变动少，但仍需人脸相似度 QA 兜底 |

### 2.2 路由规则

规划器为每张图输出 `executor` 字段，路由依据：

```text
走本地渲染器（默认）：
- 诊断只涉及颜色/影调/区域色偏
- 原图宽容度足够（无大面积死白/死黑）
- 批量交付、成本敏感

走 GPT Image 2：
- clip_high_pct 或 clip_low_pct 超阈值（高光/暗部信息已丢失，参数化救不回）
- 原图清晰度不足、需要画质提升
- 舞台/LED/混合光等本地参数化效果假的场景
- 本地预览 QA 打分低于阈值的图（自动升级重试）

两段式（推荐用于难图）：
- 先本地做基础校正 -> 再送 GPT Image 2 做宽容度/清晰度提升
```

### 2.3 GPT Image 2 强提示词锁定模板

实测结论：配合强锁定提示词，人物变动可控。所有 GPT Image 2 请求必须使用统一锁定框架，规划器只填充「目标操作」段：

```text
[锁定段 - 固定]
This is a photo retouching task, NOT image generation.
Strictly preserve: person identity, facial features, face shape, skin texture,
pores, expression, age, hair, body shape, pose, clothing design and brand colors,
logos, text and signs, background structure, object shapes, composition, and framing.
Do not add, remove, move, or replace any object. Do not beautify or slim faces/bodies.

[目标操作段 - 规划器生成，从冻结词表映射]
{action_prompt_fragment}

[画质段 - 按路由原因选用]
Recover highlight and shadow detail naturally, as if the photo was shot with
higher dynamic range. Improve overall clarity and micro-contrast without
halos, artificial HDR look, or over-sharpening.
```

配套约束：

- 输入用高保真原图（不压缩到影响细节的程度）
- mask 作为引导传入（不当作像素级精确遮罩依赖）
- 每张输出必须过第 7 节 QA（人脸 embedding 相似度、保护区色差、OCR 校验），不合格自动降级回本地结果

---

## 3. 感知层（教师栈）

```text
VLM：Qwen3-VL / Qwen2.5-VL / InternVL —— 全局场景理解与语义推理
检测：Grounding DINO / 1.5 —— 文本提示找天空、皮肤、草地、衣服、建筑、LED 屏等
分割：SAM2 —— box/point 转精确 mask
复用：BeautySDK 现有人脸检测 + 皮肤分割资产 —— 人脸/皮肤 mask 优先用现成的
色彩指标：OpenCV / Pillow / scikit-image —— mask 内客观量化
```

交叉验证防幻觉：VLM 报告的物体必须有检测框 + mask 确认才进入区域清单；检测到但 VLM 判断语义不符的（如"舞台背景幕布被检测为天空"）打回。

区域清单输出格式沿用 V1（region_id / object_type / role / bbox / mask_path / confidence / editability / protect_level），不再赘述。

---

## 4. 冻结 action 词表（V2 核心变更）

**先冻结词表，再开始标注。** VLM 在枚举内做选择题，不自由发挥。每个 action 预绑定两份模板：本地渲染器参数模板 + GPT Image 2 提示词片段。

### 4.1 词表 V1.0（草案，验证集跑完后修订一次并锁定）

```text
sky:        keep | slight_clean | natural_daylight_blue | deep_blue | keep_overcast_mood | keep_sunset | keep_night
skin:       keep | remove_yellow_green_cast | remove_red_orange_cast | clean_natural | warm_healthy
grass_tree: keep | fresh_natural_green | reduce_yellow_cast | darker_cinematic_green
water:      keep | cleaner_blue_cyan
clothing:   keep(默认保护) | neutralize_white
led_stage:  keep(默认保护) | reduce_face_color_pollution
food:       keep | warm_appetizing
global:     none | white_balance_correct | exposure_lift | latitude_recovery | clarity_boost
```

规则要点（沿用 V1 语义规则库）：

- sky：日间灰蓝天才允许转蓝；日落/夜景/舞台/阴天纪实默认 keep_*
- skin：最高保护级，只做去污染，禁塑料皮
- clothing / logo / text：默认保护，用户明确要求才动
- led_stage：不盲目中性化，舞台色是刻意的；只单独处理人脸色污染
- global.latitude_recovery / clarity_boost：只能路由到 GPT Image 2

### 4.2 action → 执行模板映射示例

```json
{
  "action": "natural_daylight_blue",
  "local_params": {
    "engine": "pe_process_image_v2",
    "region_blend": "sky_mask, feather=8px",
    "hsl_blue": {"hue": +6, "sat": +18, "lum": -4},
    "temp_shift": -300
  },
  "gpt_prompt_fragment": "Only adjust the sky region. Make the sky a natural daylight blue with realistic brightness and soft saturation. Preserve cloud shape, building edges, trees, horizon, and the original lighting direction."
}
```

---

## 5. 每区域色彩诊断

沿用 V1 指标集（avg LAB、色相/饱和度分布、亮度分位、高光/暗部裁切比、色温代理、绿品代理、colorfulness、局部对比），补充两个路由指标：

```json
{
  "region_id": "r001",
  "color_metrics": { "...": "同 V1" },
  "routing_metrics": {
    "clip_high_pct": 6.2,
    "clip_low_pct": 0.4,
    "sharpness_proxy": 0.31
  },
  "diagnosis": {
    "state": "dull_gray_sky",
    "severity": 0.64,
    "action": "natural_daylight_blue",
    "confidence": 0.88
  }
}
```

`confidence < 0.6` 的诊断一律降级为 keep（no-edit 兜底）。

---

## 6. 方案生成器

每图 3-5 个方案，V2 方案集：

1. `safe_natural` —— 白平衡+肤色自然校正，低风险，批量交付默认。执行器：本地
2. `commercial_clean` —— 更亮更干净更讨喜。执行器：本地
3. `object_enhancement` —— 目标物体强化（蓝天/绿草/白衣）。执行器：本地，QA 不过升级 GPT
4. `latitude_recovery` —— 宽容度/清晰度找回，本地做不到的画质提升。执行器：**GPT Image 2**（V2 新增，对应实测优势）
5. `restore_original_intent` —— 舞台/夜景/日落只去污染保氛围。执行器：本地为主

方案 JSON 在 V1 基础上增加字段：

```json
{
  "plan_id": "p4",
  "name": "latitude_recovery",
  "executor": "gpt_image_2",
  "routing_reason": "clip_high_pct=6.2 exceeds 3.0 threshold",
  "two_stage": true,
  "local_pre_correction": {"white_balance": "auto", "exposure": +0.3},
  "gpt_image_prompt": "[锁定段] + [目标操作段] + [画质段]"
}
```

---

## 7. 质检系统 QA

所有输出必须过 QA；GPT Image 2 输出执行**加强版**：

| 检查项 | 本地输出 | GPT 输出 | 阈值 |
|---|---|---|---|
| 目标区域色彩变化达标 | ✓ | ✓ | 目标 mask 内 ΔE 落在 action 预期区间 |
| 保护区色差 | ✓ | ✓ | 保护 mask 内 ΔE ≤ 3 |
| 人脸身份保持 | — | ✓ | 人脸 embedding 相似度 ≥ 0.92 |
| 皮肤纹理 | — | ✓ | 高频能量比 ≥ 0.85（防塑料皮） |
| 文字/logo 完整 | — | ✓ | OCR 前后一致率 ≥ 0.95 |
| 结构一致 | — | ✓ | 无新增/消失/位移物体（VLM 前后对比 + 结构相似度） |
| 边缘伪影 | ✓ | ✓ | 天空/树/建筑边缘无光晕断裂 |

GPT 输出 QA 不合格 → 自动降级回本地结果；本地结果也不合格 → 标记人工。

阈值在 100 张验证集上标定后写死进配置。

---

## 8. 蒸馏策略

不训练像素生成模型。先训练：

```text
image -> semantic_color_plan.json（含 executor 路由决策）
```

### Stage 0：零训练基线（100 张验证集）

分布：30 户外/天空、30 人物/活动、20 舞台/LED/混合光、20 暗光/过曝/困难图。

执行清单：

1. 跑通感知：检测 + mask + 区域指标；**人工检查 mask 可用率**（最容易翻车的环节：树枝天空边缘、舞台混合光）
2. VLM 在冻结词表内诊断选 action，统计教师间一致率
3. 执行：本地渲染器全量出 2-3 方案对比图；GPT Image 2 跑其中 30-40 张（含全部 latitude_recovery 路由图 + 强锁定提示词验证）
4. 人工评审：选最好方案 + **标注"不该动却动了"案例**（语义门控的直接训练信号）+ 记录 GPT 输出的人物变动案例

**Stage 0 验收线（量化）：**

- mask 可用率 ≥ 80%
- 语义门控错误率（该保留却改了）≤ 5%
- 至少一个可用方案的图片占比 ≥ 70%
- GPT Image 2 输出人脸相似度 QA 通过率 ≥ 90%

达标 → 修订并锁定 action 词表 V1.0 → 进入 Stage 1。

### Stage 1：伪标签数据集（2,000-5,000 张）

分布：1,000 随机 + 500 人物密集 + 300 户外/天空/绿植 + 200 暗光/舞台/LED。

目录结构沿用 V1（images/ masks/ metrics/ teacher_labels/ plans/ edited_candidates/ qa/ preferences/ manifest.jsonl）。

### Stage 2：学生规划模型（Hybrid，服务端推理）

```text
Grounding DINO + SAM2（照旧出 mask）
+ 学生规划头：视觉编码器（SigLIP / DINOv2）+ 结构化输出头
  预测：scene_type / 区域诊断 / action 枚举 / strength / executor 路由 / 方案排序
+ 模板引擎：action -> 渲染器参数 / GPT 提示词（确定性映射，不用 LLM）
```

不选小 VLM LoRA 输出自由 JSON——格式稳定性和枚举约束是坑；结构化头便宜、可评估、服务端推理成本低。

Stage 2 验收：

- 学生 action 选择与教师标签一致率 ≥ 85%（分物体类别统计）
- executor 路由一致率 ≥ 90%
- 学生方案经模板引擎执行后，人工偏好不劣于教师方案的 80%
- 推理成本显著低于教师栈

---

## 9. 与 BeautySDK Smart Color v2 的关系

- 定位：**叠加层**。链路为 `全局 Smart Color（现有模型/规则） -> 语义区域微调（本方案）`
- 不动 v4.3.9 `SmartColorGrading` rules fallback，不动全局模型训练链路
- 本地执行复用 `PARAM_PROTOCOL.md` 参数族，区域 mask 混合作为渲染器扩展能力提案，走独立分支开发
- 本方案的规划模型与 Smart Color 全局模型是两个模型、两个交付物，各自独立回滚
- 此层级关系需同步补进 `beauty_sdk/docs/smart_color_v2/PLAN.md`

---

## 10. 成本控制

- 教师 VLM 先跑压缩图；检测/分割本地跑；所有中间结果缓存
- 本地渲染器出全量预览，**只把路由命中 + 人工选中的方案送 GPT Image 2**
- GPT Image 2 预算按张计，Stage 0 控制在 ~40 张、Stage 1 控制在路由命中子集（预估 15-25%）
- 昂贵教师（商业 VLM）只用于金标集和难例
- 学生模型上线后，教师栈只在新场景/抽检时运行

---

## 11. 风险清单（V2 更新）

1. **GPT Image 2 改动非目标物体** —— 强锁定提示词（已实测有效）+ 高保真输入 + 加强版 QA + 不合格自动降级本地结果
2. **VLM 幻觉物体** —— 检测框 + mask 双确认才进区域清单
3. **蓝天调假** —— 场景语义门控（天气/日落/夜景/舞台/倒影规则）+ 低置信度默认 no-edit
4. **皮肤过度处理** —— 皮肤保护规则 + 人脸相似度/纹理 QA
5. **品牌色/logo 被改** —— 保护 mask + OCR/色差 reject
6. **区域参数化调色边缘光晕** —— mask feathering + 边缘伪影 QA + 假了就升级 GPT
7. **数据集场景偏差** —— 场景分布打标，刻意补齐缺失场景
8. **渲染器区域混合能力尚不存在** —— Stage 0 可先用离线 Python 复现渲染器参数 + mask 混合验证效果，C++ 扩展与 Stage 1 并行开发

---

## 12. 里程碑

| 里程碑 | 交付物 | 验收 |
|---|---|---|
| M1 数据清理与基线 | 干净 manifest（排除 `._*`）、100 张验证集选图、色彩指标提取 | 图片加载正常，四类场景齐全 |
| M2 物体+mask 管线 | 物体 prompt 清单、检测+分割、mask 可视化拼图、区域指标 | mask 可用率 ≥ 80%，误检有日志 |
| M3 词表与方案生成器 | action 词表 V1.0、双模板映射（渲染器参数 + GPT 提示词）、方案 JSON | 蓝天 action 只在语义合法时出现；保护约束出现在每个 GPT 提示词；方案间差异有意义 |
| M4 双执行器与 QA 闭环 | 本地全量对比图、GPT Image 2 强锁定实跑 30-40 张、QA JSON、人工评审表 | Stage 0 四条验收线全部达标 |
| M5 蒸馏学生 | 2,000-5,000 伪标签、训练/验证/测试划分、学生规划模型、评估报告 | Stage 2 四条验收线全部达标 |

---

## 13. 立即下一步

1. 挑选并冻结 100 张验证集（按 30/30/20/20 分布）
2. 起草 action 词表 V1.0 + 每个 action 的双模板（渲染器参数 + GPT 提示词片段）
3. 搭建感知管线（Grounding DINO + SAM2 + 色彩指标），跑出第一批 mask 拼图人工检查
4. 用已验证的强锁定提示词框架，在 10 张图上做 GPT Image 2 冒烟（含 2-3 张高光裁切图验证 latitude_recovery）
5. 出第一版人工评审表（方案选择 + 门控错误标注两列必填）
