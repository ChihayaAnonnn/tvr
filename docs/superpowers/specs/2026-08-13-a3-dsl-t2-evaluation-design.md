# A3 最佳检查点 DSL T=2 探索性复评设计

## 目标

在不训练、不重新选择检查点、不搜索超参数的前提下，对已选定的 A3 epoch index 3 检查点补做一次 DSL T=2 的 JSFUSION 评估，判断 T2V R@1 是否超过历史同口径结果 52.6。

## 固定输入

- 代码分支：`feature/contextual-intrinsic-uncertainty`
- 检查点：`ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/pytorch_model.bin.3`
- 检查点 SHA-256：`2e6f662bc69dda9b880bf882195247cdd1394de6ecb6865c48148cdfb0d85a8c`
- 数据集：MSRVTT JSFUSION 1000-video test split
- DSL：开启
- DSL temperature：固定为 2.0
- 检索方向：T2V 与 V2T
- 精度与模型参数：沿用已完成 A3 评估配置

## 实验边界

- 仅执行 eval-only，不加载 optimizer，不产生训练梯度。
- 不扫描 DSL temperature，不依据测试结果调整任何参数。
- 不修改 A3 候选、winner、acceptance 或原有 `final_test.json`。
- 使用独立日志和独立结果文件，明确标记为 post-hoc exploratory evaluation。
- 主比较是同口径的 T2V R@1：DSL T=2 新结果对历史 52.6。
- 同时记录 T2V/V2T R@1、R@5、R@10、MdR、MnR；若现有评估入口自然输出 A1/A3 风险指标，也一并记录，但不把它们解释为新的注册检验。

## 数据流

模型先生成原始 WTI 矩阵。DSL 分别构造 T2V/V2T 方向性检索矩阵并据此计算检索指标；A1/A3 风险头仍读取原始 WTI 特征，但其 failure/rank 指标以 DSL 后的检索结果为参照。A3 不参与 DSL 分数计算，因此本实验测量的是 DSL 对冻结检索结果的改善，不是 A3 对 R@1 的改善。

## 成功与解释标准

- 运行必须正常退出，日志中无 traceback、NaN 或检查点加载异常。
- `T2V R@1 > 52.6`：超过历史结果。
- `T2V R@1 = 52.6`：复现历史最佳，但没有超过。
- `T2V R@1 < 52.6`：检查数据、代码版本和方向性 DSL 口径差异；不据此调试测试温度。

无论结果如何，都保留原始无 DSL 的 A3 注册结果，且把本次结果单独报告为事后补充实验。
