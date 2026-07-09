# Phase C1c 实验结果：Qwen3-VL 天空门控对比

**日期**：2026-07-09
**脚本**：`stage0_pipeline/scripts_c1c/sky_gate_experiment.py` + `scripts_c1c/qwen_vl_client.py`
**模型**：`qwen3-vl-plus`（通过 API易 `https://api.apiyi.com/v1/chat/completions` 代理，OpenAI 兼容格式）
**样本**：C2 数据集（`dataset/c2/meta/*.json`）中所有 `matched_classes` 含 `"sky"` 的行——按构造，这些都是启发式 `_sky_plausible()`（纹理+饱和度双阈值）**已经判定为"合理天空"**的检测结果。n=30（outdoor_sky 20 + person_event 10）。

## 方法

对每个样本：重新跑 `build_classes()` 拿到 sky mask → 裁剪 mask 的 bounding box（+8% padding）→ 把"完整照片 + 裁剪区域"两张图一起发给 Qwen3-VL，要求判断裁剪区域是不是"真的开放天空"，还是"碰巧很亮/很白但其实是别的东西（墙/屏幕/LED/织物背景/过曝高光）"。

## 结果

| 指标 | 数值 |
|---|---|
| 判定成功 | 29/30（含 1 次重试后成功） |
| VLM 判定为 NOT_SKY | **0** |
| 持续失败（5 次尝试全部超时/连接中断） | 1（`person_event_058A1518`，见下方重点分析） |

**Qwen3-VL 对 29 个成功判定的样本 100% 认同启发式规则的"sky"标签**，包括 `person_event` 桶里 10 张构图更复杂、天空多为过曝灰白色的照片——VLM 的推理理由普遍是"位于树冠/屋檐上方、无纹理、无人造边缘、符合阴天/过曝天空特征"，说明它确实在做视觉推理，不是无脑同意。

## 重点案例复核：`person_event_058A1518`

这是设计稿 `phase-c2-reference-self-distill-design.md` §7 记录的"假天空检测"bug 案例（原始描述：*"one 'sky' class in the person_event bucket is a false-positive detection (no real sky in that event photo)"*，std L/a/b = 0.15/0.10/0.26）。

Qwen3-VL 对这张图的 5 次调用全部因网络问题失败（`Remote end closed connection` / `read timeout`），**没有拿到 API 判定**。转而人工查看了原图：

![058A1518](../stage0_pipeline/outputs/c1c/crops/person_event_058A1518_full.jpg)

这是一张户外团建/主题活动的合影（赫奇帕奇主题拱门装饰），构图里确实有真实天空——只是天气/曝光导致上方区域几乎纯白、毫无纹理色彩信息，**不是 LED 墙或屏幕**。

**结论：原 bug 叙述里"假天空检测"的说法不准确，需要纠正。** 真正的问题是纯数值层面的：这块区域方差极小（近乎纯色），导致 `fit_region_params.py` 里 `scale = std(edited)/std(orig)` 的比值估计器除以近零方差而爆炸（68x scale, -6719 shift），这是**分母退化**问题，不是**语义识别错误**——启发式规则把它标成"天空"是对的（这块确实是过曝天空），只是这块天空信息量太少，不适合用标准差比值去拟合缩放系数。这个数值问题已经用 `MIN_STD` 门槛修复（`fit_region_params.py`），不需要靠 VLM 来解决。

## 对 C1c 动机的影响

C1c 这条轨的原始动机（"启发式天空合理性规则有假阳性，VLM 可以替代/校验它"）建立在对这个 bug 案例的误读上。这次实验用真实数据检验后：

1. **在这 30 个样本里，没有找到任何启发式规则误判的语义级假阳性**——Qwen3-VL 与规则 100% 一致。
2. 真正触发过数值问题的案例，事后核实是"规则判断对了，但下游数值拟合对低方差区域不稳健"，已经用 `MIN_STD` 修好，跟 VLM 门控无关。
3. 项目历史上确实记录过一次"LED墙误判天空"的真语义错误（region_provider_v2.py 顶部注释、`SKY_MAX_TEXTURE_STD`/`SKY_MAX_MEAN_SAT` 两个阈值就是为它加的），但当前 97 样本的 C2 数据集里没有覆盖到这类场景，所以这次实验没能复现/验证 VLM 在那类场景下的价值。

**结论：C1c 不应该继续以"修复假天空检测 bug"为主要理由推进。** 它作为"不依赖 GPT Image 2 API 稳定性的本地/托管 VLM teacher-critic 备选"仍有次要价值，但优先级应该降低——且这次实验顺带发现 API易 代理的 `qwen3-vl-plus` 本身也有类似 GPT Image 2 的可靠性问题（同一张图 5 次请求全部超时/断连，其余 29 张全部一次或二次成功），说明"用托管 VLM API 替代 GPT Image 2 API 以获得更好稳定性"这个附带价值也没有在小样本里得到验证。

## 遗留的正确修正项

- [ ] `stage0_pipeline/scripts_c2/fit_region_params.py` 顶部注释：把"false-positive detection (no real sky in that event photo)"改成准确描述（真实天空、过曝、方差退化，非语义误检）。
- [ ] `outputs/phase-c2-reference-self-distill-design.md` §7/§9：同步纠正。
- [ ] `outputs/semantic-object-color-retouch-dev-plan-v3.md` §4.1、§12.0：C1c 标记为"已做小实验，结论是当前数据集里未发现语义假阳性，优先级下调"。
- [ ] `outputs/semantic-object-color-retouch-dev-plan-v3.1-c2-addendum.md` §2：同步。
- 完整逐样本结果：`stage0_pipeline/outputs/c1c/sky_gate_report.json`；裁剪图与原图：`stage0_pipeline/outputs/c1c/crops/`。
