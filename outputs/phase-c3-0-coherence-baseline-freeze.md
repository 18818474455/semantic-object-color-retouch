# Phase C3-0：仿色一致性升级基线冻结

日期：2026-07-10  
状态：已完成工程冻结；专项评审集已登记20组，待补至少10组真实问题样本

## 为什么先冻结

现有 Teacher v0 已经过三次局部修复，但仍采用“每个语义区域独立生成完整统计匹配结果，再按蒙版混合”的结构。尤其：

```python
cs = 1.0 + (cs_base - 1.0) * confidence
```

当 `confidence=0` 时仍有 `cs=1.0`，意味着低置信类别仍执行100%统计匹配，只是不再额外过冲。因此 C3 不再继续追加单图补丁，而是升级为“全局氛围基底 + 区域受信任度残差”。

## 冻结内容

`stage0_pipeline/baselines/c3-0/legacy_v0/manifest.json` 记录：

- 冻结 Git commit：`85edb68`
- Teacher v0 与20图回归脚本 SHA-256
- C2 manifest：97样本
- C2 parameter targets：208 class-rows
- ridge head 与报告：held-out MAE `3.74317`，均值基线 `5.48641`

大体积 edited/masks 不重复复制；它们由已跟踪的 manifest 和 commit 可重建。

## 管线开关

核心 API 和 CLI 现支持：

```text
pipeline=legacy
pipeline=coherence
```

- `legacy`：默认，调用冻结前的原始渲染逻辑。
- `coherence`：已预留名字，但在 C3-1/C3-2 实现前明确抛出 `NotImplementedError`，避免把 legacy 输出错误标记为 coherence_v1。

回归命令：

```bash
.venv-m2/bin/python stage0_pipeline/scripts_m2/regression_20.py --pipeline legacy
```

## FG-BG-Coord-v1

位置：`stage0_pipeline/eval/fg_bg_coord_v1/`

- 已将旧20图回归全部登记进 `manifest.jsonl`
- 已定义人工评分字段、严重问题字段和验收门槛
- 仍需补至少10组，优先是密集人群+商场吊顶/钢架、玻璃幕墙、夜景暖光人物、白衣服、弱内容匹配
- 用户实际测试的“商场钢架顶棚+密集人群”原始参考图/目标图尚未进入项目清单，必须补录原图，不能用相似样本替代

## 下一步

C3-1：新增 `coherence_controller.py`，先实现只包含低阶安全变化的全局氛围基底；保持区域残差关闭，先在商场密集人群问题图上验证全图是否属于同一套曝光、白平衡与色调，再进入 C3-2。
