# 过冲光晕瑕疵：根因排查 + 修复 + C2 数据重跑

**日期**：2026-07-10
**触发**：用户用 Web Demo 肉眼看真实回归图（`outdoor_sky` 桶）时发现天空/树冠边界过渡很不自然，怀疑是分割精度问题，建议对比 BSHM 人像抠图（`damo/cv_unet_image-matting`）。

## 先排除的方向：不是分割精度问题

BSHM 是专门做**人像/背景分离**的模型（发丝级精细度是针对"人"这一类调出来的）。出问题的边界是树冠跟天空，画面里没有人，BSHM 直接用不上——这是个域不匹配（人像专用模型 vs 树木/天空场景），不是"分割模型不够好"。

导出并可视化了 `build_classes()` 的每类 mask 后发现：SAM 给的 "sky" mask 轮廓粗糙但位置基本对，树冠区域整体落进了 "neutral"（未识别兜底类），**不是误检、也不是精细度不够漏了发丝级细节**——真正的问题在下游的调色公式上。

## 真正的根因：Lab-affine 分级公式的"过冲"设计

`color_reference_transfer.py` 的 `STRENGTH_PRESETS` 里，`medium`/`strong` 档位对大多数类别（"default"）用的强度系数 `cs` 是 **1.6 / 2.0**——`cs>1` 意味着比"完全匹配参考图统计值"还要更夸张（`graded = tgt_lab*(1-cs) + graded*cs`，`cs>1` 是外推超过 100%），这是故意调的"让纯色天空看起来更抓眼"的效果。这个过冲在**纯色、均匀区域**上没问题，但在两种情况下会露馅：

1. **羽化边界**：`feather_mask` 把两个 mask 的硬边界软化成一条渐变带，这条带上的像素是"class A 过冲结果"和"class B 过冲结果"按权重线性混合——两个过冲后的目标本来就差异很大，混合出来就是一条不自然的白色/青色光晕（树冠/天空、人像轮廓/LED 背景两个完全不同场景里都复现了同样的现象）。
2. **材质混杂的大面积区域**（如建筑立面）：不是单一颜色色块，过冲系数直接把整片区域的色相拉爆。

**实测对照锁定根因**：把 `default` 系数从 1.6 强制设为 1.0（其他 vibrance/contrast/sharpen 不变），树冠边界光晕基本消失，天空自然了，画面整体的"punchy"感没丢——确认过冲就是主因。

## 修复：按局部方差自适应抑制过冲（不改预设数值本身）

没有直接把 `STRENGTH_PRESETS` 里的数值改小（那样等于重新定义"medium"/"strong"，且这些数值是"Phase B / Phase D 20 图回归"验证过的），而是新增了 `_class_outlier_confidence()`：

- 对每个匹配上的类别，算出"这个像素的 Lab 值跟该类别自己范围内像素的均值/标准差差多远"（z-score）。
- `cs>1` 的部分只在像素**明显偏离自己所属类别的统计特征**时被压回接近 1.0（即取消过冲，只做普通的统计匹配，不做外推）；像素越"典型"（越像这个类别该有的样子），保留的过冲越多。
- `cs<=1` 的类别（如 neutral、light 档的多数类别）完全不受影响，因为它们本来就没有过冲。

代码上是把 `analyze_target()` 里每个匹配类别的 `_grade_class_from_stats()` 结果旁边多存一张 confidence map，`render_from_analysis()` 里 `cs = 1.0 + (cs_base - 1.0) * confidence`（`cs_base<=1` 时直接用原值）。这两个函数是 2026-07-09 为 Web Demo 拆出来的（见 `webdemo/README.md`），拆分本身已经用新旧代码逐像素比对验证过完全等价，这次修复只改了 `render_from_analysis()` 内部的 `cs` 计算。

## 验证

### 20 图回归套件（`scripts_m2/regression_20.py` 对应的 FULL_CASES + EXPANDED_CASES）

| 检查项 | 结果 |
|---|---|
| 内容匹配门槛（jaccard/explainable/suitable）决策 | **完全不变**，逐条核对 20 张全部一致——修复只动 `render_from_analysis`，不动分割/门槛逻辑 |
| 门槛判定为 SKIPPED 的图片（10/20） | 像素级 **0 差异**（`mean_abs_diff=0.0000`），这些图没有 cs>1 的类别被匹配上，修复完全不触发，符合预期 |
| 门槛判定为 OK、实际做了调色的图片（10/20） | 像素有变化但幅度小（mean_abs_diff 0.001~0.018，是全图平均；p99 最高到 0.22，说明变化确实集中在边界/离群像素，不是整体重新调色） |
| 树冠/天空边界光晕（`outdoor_sky_DSC04085(1)`） | 明显改善，刺眼的白色光晕变淡 |
| 人像轮廓/LED 背景边界光晕（`outdoor_sky_DAP03170(1)_r2`，完全不同的场景） | 同样明显改善——说明修复对"两个差异很大的类别在羽化边界相遇"这一整类问题都有效，不是单个样本调出来的 |

### 已知仍未解决、且这次没打算修的问题

`outdoor_sky_DSC04085(1)` 图右侧的建筑立面仍有明显青色瑕疵——那是整片区域的类别统计量本身被过冲拉偏，不是"局部离群像素"问题，`_class_outlier_confidence` 这个基于"像素是否像本类别"的修复对这类问题没有作用。需要另一种修复思路（比如降低这类材质混杂类别的 `cs_base` 上限，或做更细的实例级内容匹配），留作后续单独任务，不阻塞这次的边界光晕修复。

### C2 训练数据重跑（因为伪标签的"老师"就是这套 `medium` 档位）

修复影响了 `color_reference_transfer.medium` 的实际输出，而 C2 bootstrap 数据集的伪标签就是拿这个档位当老师生成的，所以重跑了全部三步：

| 步骤 | 结果 |
|---|---|
| C2.1 导出（`export_bootstrap_dataset.py`） | 97/97 样本导出成功，`suitable=47`，跟修复前完全一致（门槛决策不变） |
| C2.2 拟合（`fit_region_params.py`） | 97/97 拟合成功，208 class-rows（跟修复前一致），一条 near-flat 兜底提示（`058A1494_s0`，`MIN_STD` 机制正常工作） |
| C2.3 训练（`train_per_class_head.py`，ridge，seed=0 80/20 切分不变） | **held-out MAE = 4.02**（修复前 4.20），predict-mean 基线 6.08（修复前 6.31）——降幅比例 33.9%，跟修复前的 33.4% 几乎一样 |

结论：**C2 的核心信号没有被这次修复破坏，反而略微更干净**（MAE 小幅下降），跟"去掉离群像素的噪声过冲应该让伪标签更一致"的预期方向一致。C2.4 Smart Color v2 嫁接可以放心基于修复后的这批数据继续做。

## 涉及文件

- `stage0_pipeline/scripts_m2/color_reference_transfer.py`：新增 `_class_outlier_confidence()`，`render_from_analysis()` 里用它按像素抑制 `cs>1` 的过冲部分
- `stage0_pipeline/dataset/c2/`：`manifest.jsonl`/`meta/`/`edited/`/`profiles/`/`params/`/`param_targets.jsonl`/`head_ridge_v0.json` 全部用修复后的 teacher 重新生成
- `stage0_pipeline/outputs/color_reference_transfer/regression_20_baseline/` vs `regression_20_fixed/`：修复前后的 20 图渲染结果，供人工复查差异
