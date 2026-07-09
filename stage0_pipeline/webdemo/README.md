# 仿色 Web Demo

参考图 + 目标图 + 强度滑杆，跑的是 C2 主路径的规则教师
`scripts_m2/color_reference_transfer.py`（真实语义分割 + 逐区域 Lab 统计匹配），不是 GPT。

目的：在决定要不要把这套逻辑嫁接进 Chroma/Smart Color v2（M3.7，更大的跨仓库工程投入）之前，
先用真实照片肉眼验证效果，而不是只看 Lab 数值指标。

## 运行

```bash
cd stage0_pipeline
../.venv-m2/bin/pip install flask   # 只需装一次
../.venv-m2/bin/python webdemo/app.py
# 打开 http://127.0.0.1:5057
```

只监听 `127.0.0.1`（本机），不对外网开放。单进程、内存态 session 缓存，是本地调试用的轻量 demo，
不是生产服务（Flask 开发服务器本身也会打印这条警告）。

## 交互设计 & 性能

- 上传参考图 + 目标图后点"开始分析"：这一步会跑真实的 Grounding DINO + SAM 分割 + 逐类 Lab 统计匹配，
  耗时数十秒（模型推理本身的成本，跟 demo 无关）。
- 分析完成后拖动强度滑杆：这一步复用分析阶段缓存的分割结果和逐类分级结果，只做"按强度混合 + 全局精修"，
  单次渲染 <150ms，滑杆可以流畅拖动。
- 实现上是把 `color_reference_transfer.apply_profile()` 拆成了两半：
  `analyze_target()`（跟强度无关，贵）+ `render_from_analysis()`（跟强度相关，便宜）。
  这个拆分是纯粹的 extract-method 重构，已经用新旧代码逐像素对比验证过（`max_abs_diff=0.0`），
  `apply_profile()` 的公开行为完全不变，CLI 用法不受影响。
- 强度滑杆 0-100 对应四个锚点线性插值：0=无效果（identity）、33=light、66=medium、100=strong，
  中间的取值是相邻两个已验证档位之间的线性插值，不会外推到"strong"之外的未知参数区间。

## 已知发现（不是 demo 的 bug，是 pipeline 本身的问题）

用 `outdoor_sky` 桶的真实回归测试图（`DAP02456(1).JPG` 参考 → `DSC04085(1).JPG` 目标）跑通 demo 后，
在 medium/strong 档位都能看到天空过饱和 + 树冠/建筑轮廓边缘出现青色光晕的瑕疵。用原版 CLI
（不经过 demo/任何重构）在同一对图片上跑 `--strength medium` 复现了完全一样的输出，说明这不是
demo 引入的问题，是这套 pipeline 在这张真实图片上原本就有的视觉瑕疵——此前的验证只看了 Lab 空间的
聚合统计指标（ΔE/MAE），没有人逐像素肉眼看过这张图的渲染结果。值得后续单独排查（大概率是 feather
边缘羽化半径不够、或者天空/建筑两个 mask 相邻处的权重混合在高对比度边缘上不稳定）。
