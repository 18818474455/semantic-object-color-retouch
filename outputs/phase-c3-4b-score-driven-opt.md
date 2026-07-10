# Phase C3-4b：按人工评分反馈优化 coherence（前景提亮 / 边界衰减）

日期：2026-07-10
状态：代码已改、13 组 focus 样本重渲中；等待用户用新图复评。

## 评分反馈（13/30 组，用户打分）

- `coherence_v1` 相对 `legacy_v0` **13:0 全胜**，严重问题 0 例
- 但 `delivery_willingness` 几乎全是 1~2 分（达标率 0%）——"比旧版好"不等于"能交付"
- 唯一文字备注（`outdoor_sky_DSC040851`）：**边界还是明显，前景的人物没有提亮，亮度不统一**

## 根因

1. **前景不提亮**：`apply_global_base` 对皮肤区把 L 和 a/b 一起减半；人物主体又常落在 `neutral`/`clothing`，区域残差 cap 偏低 → 背景（sky/building）被全局基底抬亮，前景几乎不动。
2. **边界仍明显**：引导滤波改善了蒙版贴边，但区域残差在边界像素上仍可能跳变，造成光晕感。
3. **组数太多**：30 组对人工评审过重；先收敛到已打分的 13 组 focus 集。

## 改动

### `coherence_controller.py`
- 皮肤阻尼改为 **只压 a/b，不压 L**（`SKIN_AB_DAMPING`）
- 新增 `compute_foreground_luma_lift` / `apply_foreground_luma_lift`：按参考图前后景亮度差，给前景（skin/clothing/neutral）加有界 L 提亮
- 新增 `boundary_residual_damp`：在蒙版边界处衰减区域残差（最多压到 30%）

### `color_reference_transfer.py` STRENGTH_PRESETS
- `global_base` medium: 0.30 → 0.42
- 新增 `fg_luma_lift`（medium 0.38）
- `region_neutral` medium: 0.35 → 0.55（前景未识别区域也能跟一点氛围）
- `region_default` 略降（1.20 → 1.10），避免背景过冲

### 评审流程
- 新增 `manifest_focus_v1.jsonl`（13 组，来自用户已打分样本）
- `review_app.py` 默认读 focus 清单；`/summary` 只统计当前清单
- 旧评分备份到 `scores_before_opt_v1.json`，focus 的 `review.json` 已清空待复评
- 新增 `try_pair.py`：用户给新仿色案例时，一条命令出四联对比图

## 怎么看新效果

```bash
# 打分页（13 组）
cd stage0_pipeline/eval/fg_bg_coord_v1
../../../.venv-m2/bin/python review_app.py
# http://127.0.0.1:5058

# 用户带来的新案例
.venv-m2/bin/python stage0_pipeline/scripts_m2/try_pair.py \
  --ref /path/to/ref.jpg --tgt /path/to/tgt.jpg --out /tmp/try_pair_out
```

## 验收（复评后）

仍按方案文档 §四，但样本范围改为 focus 13 组：
- 严重问题 0 例
- `delivery_willingness >= 4` ≥ 80%
- coherence 相对 legacy 胜率 ≥ 70%
