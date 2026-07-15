# P2-006：Dual-Source Pair Evidence Refiner

## 状态与授权边界

**实现验证已完成（冻结规格，仍未获准训练或大规模计算）。**

当前只允许：新增 refiner 模块、接入独立 profile、编写单元测试，以及在小型合成张量上验证 strict-off、mask/shape/有限性、dense–blockwise forward/gradient 等价、确定性和表征梯度。实现验证通过不代表方法成立，也不构成 MSR-VTT 训练授权。

当前禁止：启动长期或短程数据集训练、全量 internal-val 评估、构造或读取 JSFusion 1K test dataloader、根据 test 或训练结果修改结构，以及恢复 P2-001～P2-005 的任何实现。若后续获准训练，必须先在本档案新增并冻结运行命令、seed、checkpoint selection、计算预算和验收阈值；实质机制变化必须新建实验编号。

## 问题与假设

P1 发现短文本明显较弱，且 146 条持续高置信错配中有 119 条的正确视频在三个 checkpoint 的 Top-10 内。这说明固定 token/frame 表征与 WTI hard matching 可能没有充分组织局部跨模态证据，但该单训练轨迹诊断不证明不确定性机制一定有效。

P2-006 的可证伪假设是：在不改变标签和主损失的前提下，用 pair-level 数据歧义代理与表征视角分歧代理控制候选上下文对 token/frame representation 的有限更新，能够比单纯继续训练 WTI 或同等计算量的常数门控上下文更新更有效地改善检索排序。

唯一机制入口是 **pair evidence representation aggregation**。本实验不输出独立风险分数，不在 WTI 后减 uncertainty，不引入概率分布参数、Monte Carlo、KL 或 auxiliary objective。

## 冻结结构

### 输入与 pair-level 边界

输入是现有 CLIP + `seqTransf` 输出并归一化的：

- text token `X ∈ R^[Q,T,D]` 与有效 token mask；
- video frame `Y ∈ R^[C,V,D]` 与有效 frame mask。

每个 `(text_i, video_j)` 独立构造证据，不读取其他候选的 score、rank 或表示。覆盖完整 `Q × C` 候选矩阵，不先用 P0 Top-k 筛选；因此它属于 P2 pair-level matching，而不是 P3 candidate-set interaction/reranking。

### 确定性多视角证据

固定 `K=4`。特征维度按 `k::K` 的索引交错方式分到四个互补子空间，每一维只属于一个视角；不使用 learned projection，也没有 projection dimension 超参数。每个子空间的 L2 normalize、view similarity、softmax 与熵/JS 计算固定使用 FP32；normalize 的 epsilon 固定为 `max(1e-6, finfo(input_dtype).eps)`，避免 FP16 零/极小子空间在 forward 或 backward 产生 NaN/Inf。随后以冻结温度 `τ=0.07` 计算 masked token–frame evidence `z^(k)_ijtv`，并分别得到：

\[
p^{(k)}_{ijtv}=\operatorname{softmax}_{v}(z^{(k)}_{ijtv}/0.07),
\qquad
q^{(k)}_{ijvt}=\operatorname{softmax}_{t}(z^{(k)}_{ijtv}/0.07).
\]

padding 位置必须在 softmax 前被排除，且不能参与熵、分歧、上下文或最终 WTI。

### 双源代理信号及保守命名

记 `H(r)=-Σ_l r_l log r_l`。若某个样本的有效对侧元素数为 `n`，则文本方向精确定义为：

\[
A^T_{ijt}=\frac{1}{K}\sum_{k=1}^{K}\frac{H(p^{(k)}_{ijt})}{\log n_V},
\]

\[
E^T_{ijt}=\frac{H(\bar p_{ijt})-\frac{1}{K}\sum_{k=1}^{K}H(p^{(k)}_{ijt})}
{\log(\min(K,n_V))},
\qquad
\bar p=\frac{1}{K}\sum_k p^{(k)}.
\]

视频方向用 `q^(k)` 和有效 token 数 `n_T` 对称计算。`n=1` 时相应 `A/E` 明确定义为 0；其余结果 clamp 到 `[0,1]`。因此：

- `A` / `data_ambiguity`：四个视角对齐分布的平均归一化熵；
- `E` / `view_disagreement`（仅可别名为 `epistemic_proxy`）：四个视角相对其平均对齐分布的归一化分歧。

二者须有界于 `[0,1]`，并对有效位置求值。`A` 只表示当前证据是否弥散，`E` 只表示确定性子空间是否给出不同解释。当前实现不得宣称它们已被识别为真实 aleatoric/epistemic uncertainty；只有后续语义实验表明 disagreement 随数据/模型证据增加可约减，而 ambiguity 在固有含糊样本中相对持续，才可升级命名。

此外，未来训练前必须预注册验证：`A/E` 不是 WTI score、top1-top2 margin 或 row entropy 的简单复制，且两源没有退化为同一信号。

### 不确定性如何影响表征

以各视角平均对齐分布聚合跨模态上下文：

- `c_ijt = Σ_v mean_k(p^(k)_ijtv) y_jv`，用于 text token；
- `d_ijv = Σ_t mean_k(q^(k)_ijvt) x_it`，用于 video frame。

唯一门控为：

\[
g^T_{ijt}=\operatorname{stopgrad}((1-A^T_{ijt})E^T_{ijt}),
\qquad
g^V_{ijv}=\operatorname{stopgrad}((1-A^V_{ijv})E^V_{ijv}).
\]

固定 `lambda_max=0.1`，更新为：

\[
x'_{ijt}=\operatorname{norm}(x_{it}+0.1g^T_{ijt}(c_{ijt}-x_{it})),
\]

\[
y'_{ijv}=\operatorname{norm}(y_{jv}+0.1g^V_{ijv}(d_{ijv}-y_{jv})).
\]

gate detach 用于阻止模型通过直接操纵 `A/E` 数值放大或关闭残差；上下文、更新后 representation 和最终 WTI 仍保持可微，检索损失必须能回传至 text token、frame 和可训练 encoder。
更新后的 representation 归一化沿用上述 dtype-aware epsilon；在合法单位范数输入上不改变公式，只作为低精度零范数防护。

原 WTI 的 text/video base weights 仍由候选无关的归一化 token/frame 经过现有权重头各计算一次并 softmax。refiner 接收这两个 base weights；pair-conditioned 更新后不重新计算权重，只将 `x'/y'` 的 refined cosine evidence 交给原双向 `max + base-weight` 聚合。聚合结果在模型接线层乘原 `logit_scale`，作为唯一最终 logits。训练目标保持按精确 `video_id` 的双向多正例 InfoNCE，不增加 uncertainty penalty、KL、risk/correctness head、辅助 loss、伪标签或 hard-negative。

## 冻结实现配置

| 字段 | 固定值 |
|---|---|
| profile / 唯一启用入口 | `experiment_profile=pair_evidence_refiner` |
| 视角数 | `K=4` |
| 视角构造 | 确定性互补交错特征子空间；无 learned projection |
| refiner 新参数 / buffer | 0 / 0；现有 encoder 与 WTI 权重头仍按原训练策略更新 |
| alignment temperature | `τ=0.07`，固定常量、禁止 sweep |
| residual 上界 | `lambda_max=0.1` |
| gate | `stopgrad((1-A)E)` |
| query block | 16 |
| candidate block | 32 |
| WTI weights | 由原归一化 token/frame 各计算并 softmax 一次；refine 后不重算 |
| refiner 输出 score | refined pair representations → 原双向 `max + base-weight` 聚合；不含 `logit_scale` |
| 最终 scorer | refiner score → 原 `logit_scale` |
| 主损失 | 原双向多正例 InfoNCE，且只有这一项 |
| stochastic sampling | 无 |

生产路径必须按 query 16、candidate 32 分块，不得全局物化 `[Q,C,T,V,D]`。小张量 dense reference 只作为测试 oracle，不进入常规训练/评估路径。

## Strict-off 契约

`experiment_profile=hygiene` 时必须：

1. 不实例化 refiner，不注册新参数或 buffer；
2. 不计算任何 view、`A/E`、context 或 refined representation；
3. 不消费额外随机数，直接执行现有 WTI；
4. state dict key 集、optimizer 参数组、logits、loss 及输入梯度与 P0 路径一致。

为保护 FP16 的零 padding，active profile 在进入 refiner 前使用与 refiner 相同的 dtype-aware epsilon 做 L2 normalize；该防护只属于 active 路径，不能改写 hygiene 的冻结 P0 normalize 调用。

把 `lambda_max` 设为 0、把 gate 乘 0、构造模块后在 forward 绕过，都不满足 strict-off。`pair_evidence_refiner` profile 则必须在训练和评估使用完全相同的 refined scorer；用 P2 checkpoint 关闭模块只是一项 encoder 消融，不能冒充冻结 P0。

## 实现验证门槛

以下项目全部通过，才算完成“实现验证”；仍不会自动获准训练：

1. 二进制/布尔/整数/浮点 mask 均正确；空有效 token/frame 明确报错；padding 不影响输出；
2. FP32 以及项目支持设备上的低精度 forward/gradient 有限，无 NaN/Inf；
3. dense oracle 与 blockwise 路径的 final logits、`A/E` 聚合诊断、representation update 诊断和输入梯度在预先固定容差内一致；
4. 相同输入重复运行结果一致，模块不消费随机数；
5. 非零 gate 的合成样本上，refined logits 对 text token 和 frame token 均有有限、非零梯度；
6. hygiene strict-off 通过实例化、state dict、调用 spy、数值和梯度测试；
7. rectangular `Q != C`、不同 `T/V`、单一有效 token/frame 以及全负相似度均有覆盖；
8. profile 配置、固定超参数及 scorer 定义被 manifest/测试保护，hygiene 明确拒绝 refiner 参数覆盖。

诊断输出只保留 detach 后的聚合量，例如 `data_ambiguity_mean`、`view_disagreement_mean`、`representation_update_norm` 和 `representation_update_rate`；不得在日志对象中持有完整 pair map 或 autograd graph。

## 未来训练前必须另行冻结的对照

本节只规定未来审查要求，不授权运行：

| 模式 | 要回答的问题 |
|---|---|
| P0 frozen | 冻结参照 |
| `wti_continue` | 收益是否只是从 P0 checkpoint 继续训练 |
| `constant_gate` | 收益是否只是额外跨模态上下文/计算 |
| `ambiguity_only` | data ambiguity 路由是否有独立贡献 |
| `disagreement_only` | view disagreement 路由是否有独立贡献 |
| `dual_source` | 两源是否互补 |
| `shuffled_map` | pair 对应关系是否真的重要 |

所有模式必须从同一 P0 checkpoint warm-start，保持 split、global forward batch 256、4 GPU、每卡 micro 64、accumulation 1、optimizer steps、checkpoint selection 与预处理一致。single seed 只作机制筛选；进入主线前使用至少三个配对 seed。

主结果必须是 T2V/V2T Recall、GT rank、MdR/MnR 和预定义 P1 错配切片。校准、AURC 或错误检测只可作为代理信号语义证据，不能替代检索收益。

## 停止条件

- strict-off、mask、有限性、dense–blockwise 等价或 representation-gradient 任一契约失败：停止，不请求训练；
- `A/E` 退化为 WTI score/margin/entropy 的复制，或二者不可区分：不得声称双源 aleatoric–epistemic 机制；
- `dual_source` 不能超过 `constant_gate` 与 `wti_continue`：最多保留为一般 pair representation refiner，不保留双源 uncertainty 主张；
- 至少三个 seed 不改善预注册检索主指标，或收益只来自额外参数/计算：停止 P2 核心路线；
- 只有校准、错误预测、abstention 或人工审核收益：停止，不能以这些结果挽救检索假设；
- 任何训练前读取 test，或根据 test 修改结构/超参数/checkpoint：该确认评估失效并停止。

## 运行与结果

2026-07-15 已完成冻结范围内的实现验证：

- `tests/test_pair_evidence_refiner.py`：25 项通过，覆盖独立 dense pair/view oracle、blockwise 与 checkpoint 等价、五项聚合诊断、四类输入梯度、矩形 batch、mask/singleton/全负证据、CPU BF16、CUDA FP16、零子空间、零 padding、确定性与 RNG 状态；
- refiner、模型接线、CLI、manifest 与 strict-off 的 5 文件定向回归：268 项通过；
- 项目规定的 `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests`：403 项通过，只有 1 条既有 `timm` deprecation warning；
- 相关 Python 文件 Ruff、`train_msrvtt.sh` / `eval.sh` 的 `bash -n`、`git diff --check` 全部通过。

审查期间发现并修复了两个低精度阻断问题：FP16 零/极小 interleaved 子空间的归一化反向溢出，以及 FP16 零 padding 在 refined normalize 中产生 NaN。最终实现固定使用 FP32 view 计算和 dtype-aware epsilon，并由单侧零子空间、非零 gate、零 padding的 CUDA forward/backward 回归保护。

当前没有运行训练、internal-val 全量评估或 JSFusion test，没有新增训练日志、checkpoint 或检索收益。上述证据只说明实现满足冻结契约，不说明方法有效；在训练规格另行获批前不得补写或执行训练命令。
