# FG-BG-Coord-v1

前景/背景协调性专项人工评审集。C3-0 先冻结已有 20 组回归样本；C3-1 前补齐至少 10 组真实问题样本，使总数达到最低 30 组。

## 当前覆盖

- 已登记：20 组（均来自 `regression_20.py`，外置盘挂载后可运行）
- 待补：至少 10 组
- 优先补充：密集人群+商场吊顶/钢架顶棚、人物+玻璃幕墙、人物+夜景暖光、白/浅色衣服、参考图与目标图弱匹配
- 特别问题样本：商场钢架顶棚+密集人群原始测试图目前不在项目清单中，必须补录其参考图和目标图原始路径，不能用相似图片冒充
- **2026-07-10**：`eval_harmony.py` 已实现并跑通现有20组，`outputs/<id>/metrics.json` 已生成（legacy_v0/coherence_v1两套自动指标+flag）；`review.json` 已生成空模板但**人工打分尚未开始**。汇总对比见 `outputs/phase-c3-4-eval-harmony.md`。

## 每组输出

```text
outputs/<id>/
  reference.jpg
  target.jpg
  legacy_v0.jpg
  coherence_v1.jpg
  metrics.json
  review.json
```

原图不复制进 Git；`manifest.jsonl` 只记录路径与风险标签。输出图片保持在本地并由 `.gitignore` 排除，评分和指标 JSON 可提交。

## 人工评分

每项 1~5 分：

- `foreground_change_visible`
- `background_strength_natural`
- `fg_bg_same_tone`
- `skin_natural`
- `halo_free`
- `local_dirty_color_free`
- `delivery_willingness`

另设严重问题布尔字段：

- `severe_fg_bg_disconnect`
- `severe_halo`
- `severe_skin_error`

`preferred` 取 `legacy_v0`、`coherence_v1` 或 `tie`。

## 验收

- 严重前后景割裂、严重光晕、严重肤色异常均为 0 例
- `delivery_willingness >= 4` 的比例不低于 80%
- `coherence_v1` 相对 `legacy_v0` 的人工胜率不低于 70%
- 自动指标仅作异常门禁，不能代替人工验收

## C3-0 状态

`legacy_v0` 的源码、C2 数据和 ridge head 由
`stage0_pipeline/baselines/c3-0/legacy_v0/manifest.json` 绑定到 Git commit
`85edb68` 及 SHA-256。运行基线：

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/regression_20.py --pipeline legacy
```

`coherence` 现已实现（C3-1~C3-3：全局氛围基底+区域受信任度残差+边缘感知融合）。

## 如何跑自动指标

```bash
# 对整份清单跑一遍（重新分割+渲染+写metrics.json/review.json模板）
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py \
  --manifest stage0_pipeline/eval/fg_bg_coord_v1/manifest.jsonl

# 只汇总已有 metrics.json，不重新渲染
.venv-m2/bin/python stage0_pipeline/scripts_m2/eval_harmony.py \
  --summarize --out-root stage0_pipeline/eval/fg_bg_coord_v1/outputs
```

自动指标只是异常门禁/参考证据，**不能代替下面的人工评分**——`review.json` 里的字段要由人对着 `outputs/<id>/` 下的4张图（reference/target/legacy_v0/coherence_v1）实际打分后回填。
