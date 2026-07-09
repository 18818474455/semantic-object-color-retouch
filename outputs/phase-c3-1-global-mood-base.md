# Phase C3-1：全局氛围基底

日期：2026-07-10
状态：已实现并验证，进入 C3-2（区域受信任度残差）

## 做了什么

新增 `stage0_pipeline/scripts_m2/coherence_controller.py`，实现 `pipeline="coherence"` 的第一层：**只有**全局氛围基底，不含任何区域残差（区域残差是 C3-2 的工作）。故意先单独验证这一层，避免把它的效果和区域逻辑的效果混在一起判断。

### 核心设计

- `compute_style_profile()` 新增 `profile["global"]`：整张参考图（不分区域）的 Lab 均值。旧的缓存 profile JSON 没有这个字段，`compute_global_mood()` 会把它当作"跳过全局基底"而不是报错。
- `compute_global_mood(profile, tgt_lab, compat, global_base_strength)`：
  - 用目标整图（不分区域）的 Lab 均值跟参考图整体均值求差
  - `ΔL` 截断到 ±10，`Δab` 幅值截断到 9（对照方案文档的 8~12 / 8~10 上限）
  - 乘以 `global_base_strength`（light=0.15 / medium=0.30 / strong=0.45，直接作为 `STRENGTH_PRESETS` 里新的一个键，滑杆插值天然兼容）
  - 再乘以 `compat["explainable_tgt_frac"]`：内容匹配越弱，基底力度越小
  - 复用已有的 `compat["suitable"]` 硬门槛：不适合的参考图直接返回全零位移
- `apply_global_base()`：把这个位移原样加到 Lab 的每个像素，皮肤区域位移减半（皮肤已有自己的 hue-lock 保护，不需要跟着整图偏移）。**是加法位移，不是均值/方差重缩放**——不会重演 `scripts/color_transfer.py` 那种全局串色。

### 接口改动

`STRENGTH_PRESETS` 每档新增 `global_base` 数值；`render_from_analysis(pipeline="coherence")` 现在调用 `_render_coherence_from_analysis()`，产出真实结果（C3-0 阶段这里还是 `NotImplementedError`）。webdemo `/api/render` 新增 `?pipeline=` 参数，可以直接在浏览器里切换 legacy/coherence 对比。

## 验证

### 定量：20 图回归全部跑通

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/regression_20.py --pipeline coherence
```

20 组全部无崩溃，gate 判断（suitable/jaccard/explainable）跟 legacy 完全一致（因为 `content_match_score` 跟渲染管线无关），说明新管线没有意外改变谁被跳过。

### 定性：直接復现"背景跟前景脱节"的原始 bug case

`person_event_DSC04819_r2`（参考图=商场吊顶婚礼合影，目标图=户外亭子+人群合影，`building` 标签同时框住两种完全不同的实体，`compat={jaccard:0.444, explainable_tgt_frac:0.522, suitable:true}`）：

- **legacy**：亭子屋顶被强行拉成不自然的灰白色，跟原图暖褐色木质屋顶完全脱节——这正是用户报告的"背景跟前面人物严重脱节"的原始 bug。
- **coherence（仅全局基底）**：亭子屋顶几乎保持原色，只有极轻微的整体氛围偏移（medium 档 `ΔL=-1.57, Δa=-0.03, Δb=-0.59`，肉眼几乎不可见），完全没有脱节感。

### 定性：过冲光晕瑕疵 case（outdoor_sky bucket）

`outdoor_sky_DSC04085(1)`：

- **legacy**（当前 C3-0 冻结版，已经修过三次局部 bug）：天空仍然是不自然的高饱和霓虹蓝，树冠边缘、右侧建筑仍有明显青色光晕。
- **coherence**：天空是柔和自然的浅蓝色，树冠边缘和建筑都没有青色光晕，整体看起来像正常的色彩微调而不是"加了滤镜"。

`person_event` 桶的三组样本（DSC01533 被内容匹配门槛跳过、058A1568、DSC04819）视觉上都自然，没有引入新的问题。

## 解读

即使只有这一层（没有任何区域残差），视觉自然度已经明显超过三次局部补丁后的 legacy 版本——证明方案文档 §一的诊断是对的：**核心问题不是某个区域的具体参数没调好，而是"每个区域各自追一个独立的完整统计目标"这个结构本身**。全局基底用一个被严格限幅、被内容匹配度打折的整图加法位移替代了这个结构，天然不会产生"这块被拉爆而别的地方没动"的脱节感。

代价：目前的"效果"比 legacy 弱很多（milder，接近"感觉到了"而不是"加了个强滤镜"）——这是预期的，C3-2 会在这个安全基底上叠加受信任度控制的区域残差，把天空这类"确实可靠"的区域的"抓眼感"找回来，同时保留这次验证到的"不可靠区域不強行拉爆"的安全性。

## 下一步（C3-2）

1. 把现有 `_grade_class_from_stats` / `_grade_neutral_additive` 的输出，从"直接替换 Lab 值"改写成"相对于全局基底的残差"：`residual = graded_from_stats(base_lab) - base_lab`。
2. 统一信任度：`trust = scene_confidence * pair_confidence * homogeneity_confidence * pixel_confidence`，其中 `pair_confidence`/`pixel_confidence` 复用已有的 `_class_pair_confidence`/`_class_outlier_confidence`，但要控制**整个残差**而不是只控制 `cs>1` 的过冲部分——这是方案文档 §一发现的关键代码缺口，本轮还没有动它，是 C3-2 的核心任务。
3. 每个区域设最大 ΔE 预算，多区域重叠时重新归一化残差权重。
4. 在 `person_event_DSC04819_r2`（材质错配）和 `outdoor_sky_DSC04085(1)`（光晕）两个 case 上验证：区域残差重新给天空"抓眼感"的同时，亭子屋顶依然不脱节。
