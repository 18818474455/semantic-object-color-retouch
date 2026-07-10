# Phase C3-2：区域受信任度残差

日期：2026-07-10
状态：已实现并验证，进入 C3-3（残差级边缘融合 + 材质混杂度判断细化）

## 做了什么

把 `coherence` 管线的区域处理从"独立成品再混合"改写成方案文档 §二/三 规定的残差公式：

```python
base_lab = apply_global_base(target_lab, global_mood)      # C3-1
regional_target = grade_region(base_lab, reference_stats)  # 注意：对 base_lab 分级，不是对原图
regional_delta = regional_target - base_lab
output = base_lab + trust * clamp(regional_delta)
```

### 核心改动：信任度控制整个残差，不是只控制过冲部分

legacy 的根因 bug（方案文档 §一）：

```python
cs = 1.0 + (cs_base - 1.0) * confidence   # confidence=0 时 cs 仍然是 1.0（100%统计匹配）
```

C3-2 替换为：

```python
trust = scene_confidence * pair_confidence * homogeneity_confidence * pixel_confidence
region_strength = region_cap * trust        # trust=0 时 region_strength=0（完全不迁移，停在全局基底）
output = base_lab + region_strength * clamp(regional_delta)
```

四个信任度因子：

| 因子 | 来源 | 说明 |
|---|---|---|
| `scene_confidence` | `compat["explainable_tgt_frac"]` | 复用 C3-1 已经在用的整图内容匹配度，跨全局/区域两层保持一致 |
| `pair_confidence` | `_class_pair_confidence()`（已有，C3-0之前的补丁） | 参考类别均值 vs 目标类别均值的绝对Lab差距；现在**对 `base_lab` 重新计算**而不是对原图，因为全局基底可能已经缩小了这个差距 |
| `homogeneity_confidence` | **新增** `_region_homogeneity_confidence()` | 区域自身内部Lab方差有多大——跟参考图对不对得上无关，纯粹是"这个区域本身是不是同一种材质"的信号（如"building"同时框住亮吊顶+暗钢架，内部方差本身就很大） |
| `pixel_confidence` | `_class_outlier_confidence()`（已有） | 逐像素：这个像素在自己所属区域的统计分布里是否是典型像素（边界/离群像素趋近于0） |

### 新增的强度旋钮

`STRENGTH_PRESETS` 每档新增 `region_default`/`region_skin`/`region_neutral`（残差强度上限，被 `trust` 相乘后才是实际生效强度）：

| 档位 | region_default | region_skin | region_neutral |
|---|---|---|---|
| light | 1.05 | 0.60 | 0.20 |
| medium | 1.20 | 0.75 | 0.35 |
| strong | 1.25 | 0.85 | 0.45 |

对照方案文档 §三·阶段三"天空过冲上限1.15~1.25，不再普遍用1.6~2.0"——这些数值远低于 legacy 的 `default=1.6~2.0`，因为现在的上限只在 `trust≈1`（区域可靠、材质均匀、像素典型、场景匹配都很高）时才真正生效；`trust` 稍低就会把有效强度压到 1.0 以下（不再是"至少100%匹配"），`trust→0` 时压到 0（完全停在全局基底，不做任何区域迁移）。

同时新增 `MAX_REGION_DELTA_E=30`：任何单区域的原始残差（未乘 trust/weight 前）先按 Lab 距离裁剪，防止极端统计量（比如超小mask、极端std比值）注入离谱的残差。

这些新键跟 `global_base` 一样直接塞进已有的 `STRENGTH_PRESETS` 字典（不是单独一张表），webdemo 滑杆插值 (`_interp_preset`) 自动线性混合这些新键，不用额外改插值逻辑；`IDENTITY_PRESET` 同步补了归零值。

## 验证

### 定量：20图回归

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/regression_20.py --pipeline coherence
```

20组全部跑通，gate判断（suitable/jaccard/explainable）跟 legacy/C3-1 完全一致。

### 定性：抽查各bucket对比图

- `outdoor_sky` / `outdoor_sky_r2`：天空自然，无青色光晕，比C3-1单独全局基底版本略有"抓眼感"回归（sky这种均匀高置信区域的残差被适度放行），仍然远比legacy自然。
- `person_event_r2`（含 `DSC04819_r2` 材质错配case）：亭子屋顶依然保持原色（`pair_confidence` 对这个case仍然很低，残差被压到接近0），跟C3-1结论一致——不会因为加回了残差机制就重新触发原始bug。
- `stage_led_mixed*`/`difficult*`：全部被内容匹配硬门槛正确跳过（`suitable=false`），无变化，符合预期。

未发现新的视觉异常。

## 已知限制 / 留给C3-3的工作

1. **边缘融合仍是对成品做羽化**（复用 `analyze_target` 里已有的 feathered `weights`），还没有按方案文档 §三·阶段五"对残差图做Guided Filter边缘感知融合、用原图亮度引导"。当前只是"残差公式"层面的升级，边界处理还是旧的 feather-mask 混合。
2. **多区域重叠时的残差权重归一化**未做（`weights[c]` 已经是sum-to-1的软分配，暂时够用，但没有专门处理"两个高trust区域同时覆盖同一像素"的情况）。
3. **`homogeneity_confidence` 只用了Lab方差**，还没有引入方案文档 §三·阶段四提到的纹理强度/边缘密度/颜色熵等特征——先用最简单的方差代理验证残差公式本身有效，符合"先验证结构，再加特征"的顺序。
4. `eval_harmony.py` 自动指标、`FG-BG-Coord-v1` 补齐到30组、30组人工A/B评审都还没做——C3-4 的工作。

## 下一步（C3-3）

1. 把 `analyze_target` 里 `feathered`/`weights` 的输出，改成对**残差图**（而不是最终Lab图）做边缘感知融合：用 Guided Filter，以原图亮度做引导，让残差本身在边界处平滑衰减，而不是先算出两个完整候选图再用固定羽化半径混合。
2. 给 `homogeneity_confidence` 补充纹理强度/边缘密度特征（第一版只用了Lab方差）。
3. 用商场钢架顶棚+密集人群、白衣服鬼影这两个历史case专项复测（`outputs/phase-teacher-*.md` 里记录的原始bug图），确认C3-3的边缘融合没有重新引入过冲光晕。
