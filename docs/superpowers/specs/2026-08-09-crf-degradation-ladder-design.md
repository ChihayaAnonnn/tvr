# CRF 退化阶梯：让 aleatoric 的质量风险分数对齐增量压缩损伤

日期：2026-08-09
状态：设计冻结（rev.3.2，已审阅通过），进入实施计划

## 命名约定（先立，全文遵守）

本 spec 监督和验证的**不是**概率意义上的标准差。三个名字互不通用：

- `u_frame`：帧级压缩质量风险，即模块里的 `local_uncertainty`，形状 (B, T)。
- `u_video`：视频级压缩质量风险，由 `u_frame` 加权聚合而来。
- `σ² / variance`：概率方差（`propagated_variance`、`global_variance`、
  `embedding.variance`）。本 spec **不监督也不验证**它，留给未来的检索建模。

模块里的 `min_variance` / `max_variance` 是沿用旧实现的配置名。在本 spec 里
它们只是 `u` 的数值边界（softplus 下界与硬 clamp 上界），**不带概率含义**。

即使 `u` 排 CRF 排得完美，也**不能**据此声称概率方差得到了校准。

## 背景

`1e11659` 给 aleatoric 模块接上了像素级退化的 clean-vs-degraded ranking loss，
`u` 不再是常数（smoke：`u=0.1059±0.0097`，`d_u=+0.0482`，ranking acc 1.00）。
但那三种退化——高斯噪声、降采样重插值、随机帧置黑——是我们凭空造的，
测试时没有对应物，所以 smoke 只证明了「模块能学到我们加的退化」，
没有回答「`u` 是不是在测真实画质损伤」。

本设计换掉退化源，把 `u` 的监督建立在一个**测试端可量**的物理量上。

## 一个必须先记下的事实

MSRVTT 上不存在干净的天然画质轴。实测（各 200 个随机样本）：

- 原始件 `videos/all`：全部 320×240，码率 p5/p50/p95 = 159/327/717 kbps（4.5×），
  归一化到 bits/pixel 是 73/164/342（4.7×）。
- 训练用的 `msrvtt_224_12fps`：全部 298×224，码率 p5/p50/p95 = 60/153/395 kbps（6.6×）。

分辨率和帧率都固定时，码率主要反映运动量和细节复杂度，不是源画质——
静止的说话头天然低码率，而它并不「差」。所以**原视频码率不能当 ground truth**。

结论：域内和跨域都用**受控阶梯**（同一视频内比，内容复杂度自动被抵消）；
天然码率相关性只当探索性指标，不进判据。

## 目标与非目标

目标：证明 `u` 能识别**固定 x264 配方下的增量压缩损伤**，能泛化到训练时没见过的
CRF 档位，**显著优于零参数画质基线**，并在跨域数据集上零样本保持同视频内的排序。
四项都有预登记门槛（§3）。

这**不等于**「`u` 测真实画质」。真实画质还包含失焦、运动模糊、曝光、传感器噪声、
不同编码器的伪影，本 spec 一个都没覆盖。结论段不许越过这条线。

非目标（本 spec 一律不做）：

- temperature>0 的检索臂。`u` 能否提 R@1 是下一个 spec 的事。
- 概率方差的校准。
- 码率回归、dataloader 实时编码、文本侧退化、epistemic 模块。
- 改 baseline 配方或 hygiene 协议。
- MSVD 上的任何训练或微调，它只解码评估。

## 1. CRF 阶梯产物

源文件用训练管道现在就在读的 `msrvtt_224_12fps/*.mp4`，不用 `videos/all`。
两路共享的是**同一个已经预处理编码过的源文件**；退化路唯一新增的操作是一次
固定配方的 x264 转码。干净路不做任何额外转码。

- 编码器 `av`（pip 安装，自带 libx264）。保持 298×224 / 12fps，只动 CRF。
  preset 固定 `veryfast`。**所有退化档位使用相同的 GOP 配方**
  （keyint / min-keyint / scenecut 显式写死；不假设输出 GOP 与源文件一致）。
  pixel format、color range/matrix、time base、B-frame 也全部显式写死，
  不吃 libx264 默认值——默认值会随版本漂。音频统一丢弃。
- 训练档位 CRF ∈ {30, 38, 46}，连同干净版共 4 级。
- 验证档位取训练时**没见过**的 {34, 42}（内插）、{50}（外推），
  以及 baseline challenge 用的相邻档 {36, 44}（见 §3.3）。
- 产物：`data/generated/msrvtt_crf/crf{NN}/videoXXXX.mp4`。
- 训练阶梯只造 8500 个训练视频；验证阶梯只造留出的 500 个 val 视频，用另一组 CRF。
- 一次性脚本 `scripts/build_msrvtt_crf_ladder.py`：原子写入（临时文件 + rename）、
  断点续跑、坏文件重建，边跑边落盘并报 ETA。

**编码前 preflight**（`pip download av` 成功不证明运行时有 libx264）：
取一个视频，建 libx264 encoder → 编 ~16 帧 → 重新解码 → 校验 fps、分辨率、帧数。
不过就直接失败，不进批量。

**manifest**（每个输出文件一行）：源文件 hash、输出文件 hash、CRF、preset、
PyAV/FFmpeg/x264 版本、pixel format、fps、time base、frame count、
keyint/min-keyint/scenecut/B-frame 设置、实测大小与码率、成功标志。

高 CRF 的文件更小，各档合计约 4–5 G，磁盘余量 5.8 T，不构成约束。

## 2. 数据管道与损失

### 2.1 两路必须只差压缩

dataloader 每个训练样本返回：干净帧张量、退化帧张量、档位 index `r`。要求：

- **帧号只采样一次**，两路复用同一组 index；
- resize / crop / 归一化 / mask 完全相同，随机增强只掷一次骰子，两路共用结果；
- 构建阶段校验每个退化文件的 frame count / fps / duration 与源一致；
- 退化文件缺失或损坏 → 直接报错，**不允许静默回退到 clean**（否则 label 反向）。

**identity 对照（必做）**：把同一个 clean 文件同时走 clean 路径和 degraded 数据路径，
要求逐样本 `|Δu_video| < 0.01 × base_margin`（= 5e-4）。这条能抓住增强不同步、
解码器行为差异、以及第二次前向本身带来的假差异。

**两次前向必须一致**：`torch.no_grad()` 只关梯度，不关 dropout，也不阻止
BatchNorm 更新 running stats。实施第一步是断言「同一输入连做两次视觉前向」满足
`torch.allclose(a, b, rtol=1e-4, atol=1e-5)`（在训练实际使用的 dtype 下测，
fp16 的累加顺序抖动要一并看住）。不满足就先定位随机层。
**两路必须处于同一个模式**——不允许 clean 走 train 而 degraded 单独切 eval，
那本身就是一个混淆变量；要冻结就两路一起冻结。

### 2.2 配对时共享 clean 的 relevance 权重

如果两路各用自己的 relevance 权重聚合：

```
u_clean = sum(w_clean * u_clean_frame)
u_r     = sum(w_r     * u_r_frame)
```

即使权重 detach，`u_r − u_clean` 里仍混进了「压缩改变了哪些帧重要」这一部分，
不是风险头本身的输出差。**配对监督一律共享 clean 的权重**：

```
w_pair  = relevance_weights(clean).detach()
u_clean = sum(w_pair * u_clean_frame, dim=1)
u_r     = sum(w_pair * u_r_frame,     dim=1)
u_lo, u_hi 同理
```

这样同一组语义重要性权重比较同一批帧，干净地隔离压缩风险。
单路推理（检索、诊断打分）仍用输入自身的 relevance 权重，不受此约束。

### 2.3 档位采样

第一档从 `{30, 38, 46}` 均匀采样。每个 **local batch 固定取
`n_order = max(1, round(B_local × p))` 个**样本额外抽第二档（`p = 0.20~0.25`），
第二档从剩下的两个档位里均匀采样；其余样本只解一档。
固定个数而不是逐样本掷骰，是为了避免某个 rank 一个 order pair 都没有；
`max(1, ·)` 兜住小 batch 被四舍五入到 0 的情况。
同时要求 **各 rank 的 `B_local` 相同**（`drop_last=True`），
否则末尾不整除的 batch 上「每个 rank 都有 order pair」这句话不成立。
平均解码开销约 2.2–2.25×，而训练目标真正对应完整 ordinal 阶梯。
这个比例是超参，只在 CRF-dev 上调（见 §3.1）。

### 2.4 损失（冻结形式）

```
margin_r        = base_margin * r,                base_margin = 0.05
margin_hi_lo    = base_margin * (r_hi - r_lo)

L_clean_frame = masked_mean(relu(margin_r     + u_clean_frame - u_r_frame))
L_clean_video =        mean(relu(margin_r     + u_clean_video - u_r_video))
L_order_frame = masked_mean(relu(margin_hi_lo + u_lo_frame    - u_hi_frame))
L_order_video =        mean(relu(margin_hi_lo + u_lo_video    - u_hi_video))

L_rank = lambda_cf * L_clean_frame + lambda_cv * L_clean_video
       + lambda_of * L_order_frame + lambda_ov * L_order_video
```

`masked_mean` 在 `video_mask` 的有效帧上取平均（先对帧求和再除有效帧数，
再对 batch 求平均），保证补齐帧不进分母。
四个 `lambda` 默认**全为 1.0**，各开一个 flag 便于消融。
`L_rank` 整体再乘现有的 `--aleatoric_ranking_weight`（默认 1.0），
即现有 flag 的语义不变，只是它乘的对象从单项变成了这四项之和。

**order 两项只在有第二档的样本上归约**：
`L_order = sum(pair_mask * per_sample) / pair_mask.sum().clamp_min(1)`。
不这么做的话，采样比例 `p` 会同时控制计算量和 order 的有效权重，
`p` 从 0.20 调到 0.25 会顺带把 order 梯度放大 25%，与 `lambda_of/lambda_ov`
的职责重叠。现在 `p` 只负责覆盖率和吞吐，两个 lambda 只负责强度。

`base_margin × r` 是**启发式的单调间隔**，不是「CRF 损伤线性增加」的断言——
CRF 到实际视觉损伤是内容相关且非线性的。写进 limitation。

**为什么必须有 order 项**：clean-relative 的下界不蕴含档位间单调，
`u30=0.5, u38=0.2, u46=0.3` 同时满足三个 margin 却完全乱序，
而主判据考的正是 `(34, 42)` 这种档位间排序。

### 2.5 prior loss 的处置

`variance_scale_prior_loss` 会给 `uncertainty_head` 梯度（见
`test_prior_only_does_not_train_relevance_head`），它的最优解是把 `u` 拉向常数，
和 ranking 的目标直接对冲。

**本 spec 所有臂 `--aleatoric_prior_weight` 固定为 0**，preflight 里断言这一点。
函数本身保留在 `uncertainty_modules/losses.py`（测试还在用），只是不接进训练。

### 2.5b 冻结 global_uncertainty_head

prior 关掉之后，`global_uncertainty_head` 在本 spec 里没有任何梯度来源。
`main_task_retrieval.py:754-756` 是 `find_unused_parameters=False` **加
`static_graph=True`**，后者支持每步固定不变的未用参数集，所以它不会崩
（step-013 的 smoke 就是 `prior_weight=0` 跑满 310 步的实证）。

但留着它 `requires_grad=True` 是一份永远不更新的 AdamW 状态，也让
「本 spec 只学 `u`」这句话靠约定而不是靠构造成立。所以本 spec 显式
`global_uncertainty_head.requires_grad_(False)` 并把它排除在 optimizer 之外。
概率方差本来就是非目标，冻结比开 `find_unused_parameters=True` 干净，
且 `state_dict` 结构不变，checkpoint 兼容。

### 2.6 其余接线

模型侧结构不变：仍在 `forward()` 里、allgather 之前、`no_grad` 的第二次视觉前向、
两路特征 detach。改的只是退化张量的来源——从 GPU 现算变成从盘上读。

`degrade_video_pixels` 及其 5 个 flag 一并删除。留着就是两个退化源共存，
是混淆变量；需要回溯有 git（`1e11659`）。

**标准检索 eval 不做成对前向**，路径完全不变。CRF 诊断器是独立的评估入口，
它把每个 CRF 版本当**普通单路输入分别推理**，再在外面组对比较。

**`u` 数值上也不进检索前向。** `aleatoric.py:123` 是短路分支：
`pooling_temperature == 0` 时 `aggregation_logits = relevance_logits`，
不做减法。`detach()` 只隔离梯度、不隔离数值，所以这里必须是短路而不是乘 0 系数
（乘 0 还有 `0 * -inf = nan` 的问题）。`pooling_temperature` 就是那个独立系数，
本 spec 全臂固定 0，不新增 flag。配套断言见 §3.5 第 5 条。

现在 3.6 s/step 由 GPU 侧梯度重算主导，12 个 worker 大概率吃得下 2.2× 解码，
但这是本版最大的吞吐风险，实施时单独量一次。

## 3. 验证协议与判据

判据在跑之前写死。

### 3.1 数据面：CRF-dev / CRF-audit

hygiene 留出的 500 个 val 视频，按视频固定切成 **CRF-dev 250 / CRF-audit 250**。
所有超参迭代（双退化比例、四个 lambda、诊断口径）只看 CRF-dev；
CRF-audit 只在最终 checkpoint 上跑**一次**，出的数就是报告数。
不这么切，500 个会在几轮迭代后悄悄变成开发集。

**checkpoint 选择规则**：在 CRF-dev 上 `u_video` 的配对排序准确率最高者；
并列取更早的 epoch。检索 R@1 不参与本 spec 的 checkpoint 选择。

test 的 1000 个在本 spec 里一次都不碰。
（val 参与过 R@1 的 checkpoint 选择，与 CRF 排序无关。）

### 3.2 主判据（域内、非循环）

按训练时没见过的 CRF {34, 42} 重编。每个视频三个有序对
`(clean, 34)`、`(clean, 42)`、`(34, 42)`，逐对判 `u` 是否更大。
**bootstrap 单位是视频，不是 pair**——三个 pair 来自同一个视频，不独立。

- 配对排序准确率（视频内三对求均值，再对视频求均值）在 CRF-audit 上 **≥ 0.90**；
  帧级和视频级各报一份，门槛只压在 `u_video` 上。
- 外推点 CRF 50：`u(50) > u(42)` 的视频占比 **≥ 0.80**。同时报告
  `median(u50 − u42)` 与饱和率；若中位增量 **< 0.2 ×** `median(u42 − u34)`
  或饱和率 **> 0.10**，这一格判为「不可读」而非「通过」，
  结论里就只说内插成立、外推测不出来。

**饱和的定义**：`u` 距 `min_variance` 或 `max_variance` 的距离小于
`(max_variance − min_variance)` 的 1%。

### 3.3 淘汰门：必须显著赢过零参数画质基线

逐帧 Laplacian 方差，定义写死：转 **luma**（BT.601）→ 每帧 3×3 Laplacian 卷积 →
取该帧像素方差 → 视频级取帧间**中位数** → 分数取 `-log(var + 1e-8)`
（值越大越「不确定」，与 `u` 同向）。用完全相同的解码帧和相同的 pair 集合评。

**判据（淘汰门，非参考）**：按视频 bootstrap 求
`Δacc = acc_u − acc_Laplacian` 的 95% CI，要求 **下界 > 0**。
不比点估计。

`{clean, 34, 42}` 这组档位间隔大，两边都可能贴近 100% 而出现天花板效应，
读不出差异。所以另建 **baseline challenge set**：相邻未见档对
`(34, 36)` 和 `(42, 44)`，同样在 CRF-audit 上评，淘汰门以这一组为准；
`{clean, 34, 42}` 那组的 `Δacc` 同报，当参考。

措辞收窄：Laplacian 排得一样好**不能**证明 `u` 内部机制是模糊探测器，
只能说明——**`u` 在留出视频的压缩排序任务上没有显著超过廉价无参数基线，
因此尚无证据表明这个学习头有额外价值**。

可另算 PSNR / SSIM / LPIPS 作为受控配对的全参考诊断（它们需要参考帧，
不是测试时可用的基线），只用来解释现象，不进判据。

### 3.4 跨域（正式目标，有门槛）

MSVD 取测试集的一个子集（≥ 300 个视频），源文件用 MSVD **原始件**，
**保持各视频自己的分辨率和 fps 不变**，只复用同一套 x264 编码参数造 {34, 42} 阶梯。
不做统一预处理——把它们压成 298×224/12fps 等于先抹掉跨域差异再说泛化。
准确率仍由 `(clean, 34)`、`(clean, 42)`、`(34, 42)` 三对组成，口径与 §3.2 一致。
仍满足「MSVD 不训练不微调」。

**门槛**：准确率 ≥ 0.80，且按视频 bootstrap 的 95% CI **下界 > 0.5**（即显著优于随机）。
不过门不关闭整条路，但结论必须缩回「域内成立、跨域未验证」，
并且下一个 spec 不得假设跨域泛化。

天然 `Spearman(u, −bits/pixel)` 降为**探索性指标**，符号预登记为正，
明确记它被分辨率、内容复杂度、年代、编码器混淆。

### 3.5 预登记的诊断表

全部在同一个 checkpoint 上一次跑完：

1. retrieval loss 反传后，`uncertainty_head` 与 `global_uncertainty_head`
   的每个参数 `.grad is None` 或严格为 0；
2. `L_rank` 反传后，`relevance_head` 与检索主干的每个参数 `.grad is None` 或严格为 0；
3. `global_uncertainty_head` 全部参数 `requires_grad=False`，且完整 `L_total`
   反传后所有 `requires_grad=True` 的参数都拿到了梯度（这条同时兜住 DDP）；
4. 同输入两次视觉前向满足 §2.1 的 `allclose`；
5. **检索不变性**：随机扰动 `uncertainty_head` 的参数，
   `pooling_temperature=0` 下检索 logits 必须逐位不变；
6. identity pair 的 `|Δu_video| < 5e-4`；
7. `u(clean) / u30 / u38 / u46` 的分布与两两序关系；
8. 帧级与视频级 ranking accuracy 分别报；
9. 饱和率（按 §3.2 的定义）；
10. 本臂 hygiene R@1 落在噪声地板内（|Δ| < 1.1 pt）。第 5 条通过时这是纯粹的
    冗余确认；第 5 条不过则它是接线出错的信号。

bootstrap CI **只加在样本统计量上**：视频级 accuracy、`Δacc`、Spearman。
梯度断言（1、2）、确定性断言（3、4）、吞吐都是通过/不通过或单值，不配 CI。

### 3.6 证伪条件

任一命中即证伪：

- CRF-audit 上 `u_video` 配对排序准确率 < 0.90；
- baseline challenge set 上 `Δacc` 的 95% CI 下界 ≤ 0。

命中时能推出的**只有**：在当前视觉特征、当前风险头和当前训练配方下，
没有得到「压缩损伤可分」或「优于 Laplacian」的证据。
不能推广成「压缩失真在帧级特征上不可分」。
无论如何这条路关掉，不再往 temperature>0 走。

跨域门槛（§3.4）不属于证伪条件，不过只缩窄结论。

## 依赖

- `av`（pip，自带 libx264）。当前环境无 ffmpeg 二进制、无 `av`、无 `decord`，
  已确认 `pip download av` 可用；运行时能力由 §1 的 preflight 兜底。
- `dataloaders/dataloader_msvd_retrieval.py` 已存在，跨域判据需要给它加
  CRF 阶梯读取，但不需要新写 dataloader。
