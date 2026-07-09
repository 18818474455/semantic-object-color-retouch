# C2.3b 实验结果：ridge baseline 升级为 torch MLP，结论是暂不升级

**日期**：2026-07-09
**脚本**：`stage0_pipeline/scripts_c2/train_per_class_head_mlp.py`
**动机**：`train_per_class_head.py`（C2.3 ridge baseline）的文档里明确写了"数据量到 ~40+ 后应该升级成 torch MLP"，n 已经到 208，按计划做这次升级尝试。

## 改动

1. 类别 one-hot（8 类：neutral/sky/skin/building/floor/clothing/screen/stage backdrop）代替原来 16 维的 hash 伪嵌入——词表已经稳定，one-hot 更干净、没有哈希碰撞噪声。
2. 两层小 MLP（16→8 或 32→16，dropout + weight decay），用跟 ridge baseline **完全相同的 train/held-out 切分**（seed=0, 80/20）保证可比性。
3. 超参搜索用**只在训练集内部做的 5-fold 交叉验证**（grid：隐藏层大小/dropout/weight_decay 五组），选出最优配置后才在真正的 held-out 测试集上评估一次——避免直接在测试集上调参这种偷看数据的做法。

## 结果

| 模型 | 训练集内 5-fold CV MAE | 真实 held-out MAE |
|---|---|---|
| ridge（v0，基线） | 3.26 | **4.20** |
| MLP（v1，CV 选出的最优配置：hidden=(16,), dropout=0.1, wd=0.01） | 3.27 | **6.13 ± 0.39**（5 个随机种子） |
| predict-mean 基线 | — | 6.90 |

两个模型在训练集内部的交叉验证分数几乎一样（3.26 vs 3.27），**但换到固定的 held-out 测试集后，ridge 只退化了 29%（3.26→4.20），MLP 退化了 87%（3.27→6.13）**。这不是"测试集运气差"（同一个测试集，ridge 表现正常），而是 MLP 在这个数据规模下更容易过拟合训练集的具体噪声、迁移到新样本上更脆弱——跟 `train_per_class_head.py` 原始文档里"n 太小时线性模型更合适"的预判一致。

## 结论

**暂不用 MLP 替换 ridge。** ridge (v0) 继续作为生产用的 PerClassHead。

不是"MLP 这条路走不通"，而是"当前 n=208 还不够"——`train_per_class_head_mlp.py` 已经实现好了完整的 CV 选参 + 多 seed 评估流程，等数据量进一步扩大（比如 C1 GPT 残差补充 hard-case，或者更多 Stage 0/新拍摄样本进来）之后可以直接重跑这个脚本复查，不需要重新搭建。

产出物：`dataset/c2/head_mlp_v1.pt`、`dataset/c2/head_mlp_v1_report.json`（含完整 CV grid search 结果）。
