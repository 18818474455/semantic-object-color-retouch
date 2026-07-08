# Chroma 引擎对齐说明

本文件记录本项目(语义物体调色)与 **Chroma 调色引擎源码**(`/Users/mac/Desktop/chroma调色模型源码_20260705`)的对接关系。**开始 M4(接引擎)前必读。**

## Chroma 已经有什么(不要重复造)

Chroma 不是空白引擎,它已经具备:

1. **完整 PS/Lightroom 对齐滑块** —— `chroma_process_params_t` / `BasicAdjustParams`(`core/include/chroma/adjust_basic.h`):
   `exposure, highlights, shadows, whites, blacks, contrast, saturation, vibrance,
   temperature, tint, texture, clarity, dehaze, vignette, splitToning*, globalHue,
   midtones, sharpen, denoise`,加 8 段 HSL Color Mixer、tone curve、style LUT。全部 `[-1,1]`。
   渲染入口:`chroma_process_srgb_image_f32` / `apply_basic_adjustment_image`。

2. **已有的 VLM 智能调色链路** —— `docs/AUTO_VLM_GLM45V.md`:
   `图像 → GLM-4.5V → 输出上面这些滑块(JSON) → 本地 5 条安全钳制规则 → 渲染`。
   安全规则含 **护肤(skin_tone_protection)/护高光/护暗部**,是确定性的。

3. **区域统计** —— `chroma_auto_color_context_t`:已经算
   `sky_pixel_ratio / skin_pixel_ratio / foliage_pixel_ratio / highlight_clipped_ratio /
   shadow_clipped_ratio / gray_world_rgb / average_oklab / average_oklch_*`。
   见 `docs/COLOR_DETECTION.md`。注意:这是**比例级**统计,不是像素 mask。

## 本项目的增量(真正要新做的)

| 能力 | Chroma 现状 | 本项目增量 |
|---|---|---|
| 调色维度 | **全局单方案** | **区域化(按 mask)+ 多方案(3-5)** |
| 语义判断 | VLM 出全局 scene + 滑块 | **逐物体门控**:该不该调、哪些必须保护 |
| 区域 mask | 只有比例统计 | Grounding DINO + SAM2 出**像素 mask** |
| 宽容度/清晰度上限 | 受原图限制 | **GPT Image 2** 补死白/死黑/清晰度 |
| mask 混合执行 | **引擎没有**("Local/masked edits are later operators") | Python 预览原型 → 未来 Chroma 区域算子 |

一句话:我们是 Chroma AutoColor 的**区域化 + 多方案 + 生成式补强**升级层,不是另起炉灶。

## 参数契约对齐

- 权威映射表:`config/chroma_param_map.json`(action → chroma 真实滑块)。
- **全局 action**(exposure/temperature/tint/saturation/…)可**今天就**通过 `chroma_process_srgb_image_f32` 跑真引擎。
- **区域 action** 列出的是"在 mask 内要施加的滑块",引擎暂无区域算子,Stage 0 用 Python 预览近似,同时这就是未来 Chroma 区域算子的规格。
- `actions.v1.json` 里的 `local_params`(`toward_hue/sat_add/temp/tint/…`)只服务 **Python 预览渲染器**,不是引擎契约。引擎契约以 `chroma_param_map.json` 为准。

## 落地路线修正

1. **M3+**:规划器输出的每个 region_action 附带 `chroma_param_map` 查得的滑块集。
2. **M4**:
   - 全局方案直接调 `chroma_process_srgb_image_f32` 验证真机效果(替换 Python 全局近似)。
   - 复用 `chroma_compute_auto_color_context_f32` 拿 sky/skin/foliage 比例做**门控输入**(比我现在的启发式场景判断更可靠)。
   - 区域算子:向 Chroma 提"masked base-adjust"能力(在 `apply_basic_adjustment_image` 上加 mask 加权),把 Python 原型移植过去。
3. **可选**:本项目的多方案/区域诊断可以作为 Chroma AutoColor 的 `few-shot`/升级版 advisor 回流。

## 复用 vs 新建 决策

- 复用:滑块定义、渲染管线、安全钳制规则、区域比例统计、VLM 参数 JSON schema 思路。
- 新建:像素级 mask(SAM2)、逐物体语义门控、多方案生成、GPT Image 2 补强、mask 混合执行、蒸馏学生规划模型。
