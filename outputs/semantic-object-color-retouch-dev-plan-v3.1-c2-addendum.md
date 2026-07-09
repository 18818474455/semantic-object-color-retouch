# 开发方案 V3.1 增补 · Phase C 双轨与 C2 自蒸馏

> **日期**：2026-07-09  
> **基线**：`semantic-object-color-retouch-dev-plan-v3.md`（841 行执行版）  
> **详细设计**：`phase-c2-reference-self-distill-design.md`

本增补 **不替换 V3**，只在 Phase C / 蒸馏 / 里程碑三处收束策略变更。未提及部分仍以 V3 为准。

---

## 1. 变更摘要

| 项 | V3 原表述 | V3.1 修订 |
|----|-----------|-----------|
| Phase C | 单一 GPT teacher 蒸馏量化 | **C1 ∥ C2 双轨**：C1=GPT 残差量化；C2=Reference 自蒸馏（主路径） |
| M6 伪标签来源 | 主要靠 teacher 栈 + GPT | **C2 bootstrap 为主**（本地 pseudo-target），GPT ≤15% hard-case |
| M7 学生模型 | 结构化规划头 + 模板引擎 | 规划头不变；**执行头改为 RegionalParamHead**（嫁接 Smart Color v2） |
| GPT 阻塞 | 阻塞 M3 | **不再阻塞**；C2 可独立推进到 C2.3 |
| API 提供商 | funai | 已切换 **API易**（`api.apiyi.com`，`gpt-image-2-all`） |

---

## 2. §12 蒸馏路线 · 修订版

在 V3 §12 之前插入本双轨定义：

### 12.0 Phase C 双轨（2026-07-09）

```text
Phase C1 — GPT Teacher 量化（辅助轨）
  目的：标定哪些语义类 / 场景必须走 GPT（latitude、clip、结构级变化）
  输入：target + reference + 本地 medium 结果
  输出：per-class Lab 残差报告（distill_report.json）
  依赖：API易 gpt-image-2-all 双图编辑稳定
  状态：API 已切换；待冒烟验证

Phase C2 — Reference 自蒸馏（主轨）★
  目的：用可训练 per-region head 替代手写 Lab 统计，对接 Smart Color v2
  Teacher v0：color_reference_transfer.py medium 档 pseudo-target
  数据：20 图回归集 → 扩至 200+（无需 GPT）
  详见：phase-c2-reference-self-distill-design.md
```

### Stage 0 / 1 / 2 衔接调整

- **Stage 0**：已完成骨架 + 仿色产品化；C1 冒烟 **降级为可选验收项**（不达标不阻塞 C2）
- **Stage 1（M6）**：manifest 采用 `c2_manifest.jsonl` schema；优先导出 suitable 样本
- **Stage 2（M7）**：训练分两阶段  
  - **M7a 执行头**：C2 RegionalParamHead（仿色 / 区域 grading）  
  - **M7b 规划头**：action / strength / route（原 V3 Stage 2）

---

## 3. §13 Smart Color v2 · 嫁接结论

V3 §13 写「两个模型、两个交付物、可独立回滚」——**V3.1 明确执行方式**：

1. **全局 Smart Color v2**：不改训练链路、不改 v4.3.9 fallback  
2. **RegionalParamHead（C2 新建）**：  
   - 输入：style_vec + tgt_vec + 语义类  
   - 输出：`chroma_param_map.json` 定义的滑块子集  
   - 训练：复用 SCv2 param-target + renderer parity  
3. **Chroma 引擎增量**：masked base-adjust 算子（`CHROMA_ALIGNMENT.md` 已列规格）

---

## 4. §14 里程碑 · 新增行

| 里程碑 | 交付物 | 验收 |
|--------|--------|------|
| **M3.5 C2 Bootstrap** | `scripts_c2/export_bootstrap_dataset.py`、`dataset/c2/manifest.jsonl` | ≥40 suitable 样本，mask/profile/pseudo 三元组完整 |
| **M3.6 C2 Head v1** | `train_per_class_head.py`、eval 报告 | 20 图回归 ΔE 不劣于 rule baseline 的 ≥90% |
| **M3.7 SCv2 嫁接** | Chroma 仓 parity 报告 + 权重导出 | Python/C++ max ΔE < 1.0 |

原 M6/M7 保持不变，但 M6 输入改为 **C2 manifest 为主**。

---

## 5. §17 立即执行清单 · 更新（替换 V3 §17 第 8–10 条）

**已完成**（截至 2026-07-09）：

- [x] git init + GitHub 仓库  
- [x] API 切换 API易 + client 适配  
- [x] C2 设计稿 + V3.1 增补  
- [x] **C2.1 全量导出**：21 样本（外置盘挂载后跑通，9 条 suitable=true）  
- [x] **C2.2 全量拟合**：41 class-rows（顺带修复了假天空检测导致的退化 scale bug，详见设计稿 §7）  
- [x] **C2.3 ridge baseline**：held-out MAE=3.83，明显低于预测均值基线 5.83，说明信号可泛化  

**当前优先级**：

1. **扩样**：把 Stage 0 100 张验证集也并入 C2.1，n_rows 从 41 → ≥100，为升级 torch MLP 做准备  
2. **M3.7**：在 Chroma 仓开 `feature/regional-smart-color-head` 分支（数据门槛已达标，可以启动 C2.4 嫁接）  
3. **C1** API易 双图冒烟（并行，不阻塞 1–2）  
4. **Web Demo**（Polarr 式三控件）基于 `color_reference_transfer.py`  
5. 与 Chroma 仓对齐 **M3.7 masked operator** 排期  

---

## 6. §18 成功标准 · 补充

V3 Stage 0 成功标准保留。新增 **C2 成功标准**（可独立于 GPT 达成）：

```text
Bootstrap manifest ≥40 条 suitable 样本
Learned head 在 20 图回归集上 ≥90% 不劣于 rule baseline
至少 3 个 bucket（outdoor/person/stage）各有一条 learned ≤ rule 的 showcase
RegionalParamHead 输出可映射到 chroma_param_map（可解释、可手动微调）
```

达到 C2 成功标准即可进入 M6 扩样，**无需等待 GPT API 全量稳定**。
