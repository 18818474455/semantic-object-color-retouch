# "前景图层没反应"问题：neutral 类改用加法 mood-cast + C2 数据重跑

**日期**：2026-07-10（紧接 `phase-teacher-overshoot-halo-fix.md` 同一天的第二个发现）
**触发**：用户在 Web Demo 上传一张密集人群的商超照片，反馈"只模仿了背景，前面那一层没有模仿"，问能不能像 PS 图层一样多模仿几层。

## 先厘清：这套 pipeline 本来就是多图层架构

`color_reference_transfer.py` 按 sky/grass/tree/water/clothing/led/building/floor/flag/skin 等独立类别分层处理，每层各自算 Lab 统计、各自出目标色，再按软蒙版权重叠加——这已经是 PS 图层的思路，不是笼统调一遍全图。用户反馈的现象不是"架构只有一层"，是这张具体照片里前景没有被识别成任何一个类别。

## 根因：密集人群没有被检测器识别，掉进了保守的 neutral 兜底层

拉取用户在 Web Demo 里实际上传的参考图/目标图（同一家商超、密集人群+钢架顶棚），跑 `analyze_target()` 复查：

```
ref classes: building, skin, neutral   （没有 clothing）
matched_info: building matched=True, skin matched=True, neutral matched=True
```

`build_classes()` 导出 mask 后可视化确认：整片人群（衣服、身体）全部落进 "neutral"——Grounding DINO 的 "clothing" 查询词在这种人挨得很紧、互相遮挡的密集场景里根本没触发任何框。"neutral" 不是"没处理"，它确实会走 `_grade_class_from_stats` + `cs=0.45`（medium）的规则，只是系数比 sky/building 的 1.6 小很多，肉眼几乎看不出变化。

## 走了一步弯路：给"人群"单独开一个类别，效果反而更差

试着把 "people." 加进 `detect_classes` 的查询词，确实能在目标图上框出 26% 的人群区域。但套用跟 sky/building 一样的"整层往参考图均值/方差靠"的 `_grade_class_from_stats` 渲染后——不管用多大的系数（跟 default 一样强的 1.6，还是跟 skin 一样柔和的 1.3）——整片人群都被"洗"成同一种淡紫/淡蓝色调，白衣服、灰衣服、深色衣服全部被拉向同一个目标色，反而丢了人群本身"每个人穿的都不一样"这个真实感。

**结论**：问题不是系数大小，是"把一片本来颜色就千差万别的内容，硬套一个统一目标均值/标准差"这个数学模型（mean/std 重缩放）本身不适合这类内容。sky/building 能这么处理是因为它们本来颜色就相对均匀。

## 修复：neutral 改用"加法 mood-cast"而不是"重缩放"

新增 `_grade_neutral_additive()`，专门给 "neutral"（以及任何这类"识别不出来的异质内容"）用：

- 不再对整个区域做 `(pixel - mean)*(ref_std/tgt_std) + ref_mean` 的重缩放（重缩放会强迫所有像素收敛到同一个目标统计量，抹平原有的色彩多样性）；
- 改成对 L/a/b 三个通道各算一个**固定的加法偏移量**（`ref_mean - tgt_mean`），整片区域每个像素都加上同一个偏移——这样每个像素跟邻居之间原有的色差（也就是"每个人穿的不一样"）完全保留，只是整片区域的"重心"朝参考图的整体色调挪了一点，效果类似给整张照片轻轻套一层白平衡/氛围滤镜，而不是重新调色。
- `render_from_analysis()` 里现有的 `cs`（neutral 在 medium 是 0.45）不变，控制这个偏移量实际生效多少；neutral 占比过大时的 taper 逻辑也照常保留在上面。

## 验证

**人群测试图**（同一对 ref/tgt）：medium/strong 档渲染后，人群衣服的颜色多样性完全保留，同时整体氛围有柔和的偏移，不再是"背景变了、前景没反应"。

**20 图回归**：

| 检查项 | 结果 |
|---|---|
| 内容匹配门槛决策 | 完全不变，20 张全部一致 |
| 之前那次过冲光晕修复验证过的老 bug 案例（`person_event_058A1518`，近乎纯色天空） | 用新的 neutral 加法法渲染依然正常，没有重新引入洗色/串色问题 |
| 20 张整体像素差异 | 全部有小幅变化（mean_abs_diff 0.0006~0.0135），符合预期——neutral 在大多数图里都存在，但改动幅度本身很温和 |

### C2 训练数据重跑

neutral 的分级方式变了，C2 bootstrap 伪标签里凡是带 neutral 的 class-row 都会变。重跑全部三步：

| 步骤 | 结果 |
|---|---|
| C2.1 导出 | 97/97，`suitable=47`，跟改动前完全一致 |
| C2.2 拟合 | 97/97，208 class-rows，跟改动前一致 |
| C2.3 训练（ridge，seed=0） | held-out MAE = **4.07**（上一版过冲修复后是 4.02，基本持平，仍远低于 predict-mean 基线 6.07；降幅比例 32.8% vs 上一版 33.9%，噪声级别的波动，信号稳定） |

结论：改动没有破坏 C2 的核心信号，可以放心继续基于这批数据做 C2.4 Smart Color v2 嫁接。

## 涉及文件

- `stage0_pipeline/scripts_m2/color_reference_transfer.py`：新增 `_grade_neutral_additive()`，`analyze_target()` 里 "neutral" 类改用它而不是 `_grade_class_from_stats()`
- `stage0_pipeline/dataset/c2/`：全部用新版 teacher 重新生成
- 独立小任务（未做，记录留后续）：Grounding DINO 在密集人群场景对 "clothing" 的检测召回率不稳定（同一场景两帧里 "people" 查询词命中率从 4% 到 26% 波动很大），如果以后真的要给"人群/服装"单独开一层，需要先解决检测稳定性，而不是先解决调色公式——即使调色公式换成加法也是给"整片 neutral 兜底"用的保守方案，如果分出一个独立的高置信度"人群"类别，还是要重新判断加法/重缩放哪种更合适
