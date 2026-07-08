# UACL + Query Hard Negative Implementation Plan

更新日期：2026-07-04

## 当前决策（2026-07-04）

**Hard negative 主线终止。**

相关实现保留为消融与诊断工具，但后续训练主线不再继续：

- 不再跑 `--use_hard_negative_packing` repeat。
- 不再跑 `--use_explicit_hard_negative_loss` repeat。
- 不再做 `w_hard_negative` 权重搜索。
- 不再继续构建新的 hard map 来试图挽救该方向。

当前可写入论文/汇报的结论是：**显式 hard negative 与 batch packing 在 MSRVTT 上能产生训练信号，但不能稳定转化为 T2V R@1 提升。**

| 路线 | 关键结果 | 决策 |
|------|----------|------|
| raw HN packing | Best T2V R@1=48.1，低于 B1-only 49.3 | 失败 |
| clean HN packing | 单次 49.7，但 repeat/诊断不稳定，后续最高 48.6 | 不作为主线 |
| 旧 clean-map explicit HN | Best T2V R@1=49.4，fixed/regressed=32/31 | 净收益不足 |
| model-mined explicit HN | Best T2V R@1=48.6，fixed/regressed=31/38 | 失败 |

根因判断：

1. HN loss 并非没有生效：`ret_gap` 扩大、GT rank mean/`rank<=10` 局部改善，说明训练信号存在。
2. 训练信号没有转化为 Top-1：fixed/regressed 诊断显示修复与退化接近甚至偏负。
3. MSRVTT 的文本-视频检索存在大量语义近邻和多正例式歧义；许多“难负例”在语义上接近合理正例，强行推远会制造边界退化。
4. 从代码搜索迁移来的 query hard negative 假设更适合一对一、可精确定义负例的任务；视频文本检索里同主题不同视频之间常不是干净负例。
5. 当前检索主指标由 WTI/CrossEn top-1 排序决定，HN 更像改善候选中段排序，而不是稳定翻转 top-1。

当前下一步：**验证 UACL-style 模态内对齐的稳定性**。HN 结果作为负消融保留，不再作为优化方向。

## 已实现但不再作为主线的 HN 代码（保留）

- 已在 `feat/uacl-explicit-hn-intra` 分支接入论文/原版代码口径的 **显式 hard-negative InfoNCE loss**，默认关闭：
  - CLI：`--use_explicit_hard_negative_loss`
  - 权重：`--w_hard_negative`
  - 数据：复用 `--hard_negative_path`，默认 clean map。
- MSRVTT 训练 dataloader 已支持在样本后额外返回 hard-negative video：
  - 无属性：`text/mask/segment/video/video_mask/sample_index/hard_video/hard_video_mask/hard_valid`
  - 有属性：在属性三元组后追加同样的 `sample_index/hard_*` 字段。
- 模型侧已支持额外编码 hard-negative video，并将 hard-negative logits 作为额外列并入 query-to-video CrossEntropy 分母：
  - `logits = concat([sim(q_i, v_j), sim(q_i, v_hard_j)], dim=1)`
  - label 仍指向原始正样本列 `i`。
  - `hard_valid=0` 的 hard-negative 列会被 mask，不参与分母竞争。
- 2026-06-30 已将早期 softplus margin-style 约束清理为 InfoNCE 分母扩展；单测覆盖 invalid hard-negative mask 与拼接分母行为。
- 已接入 **UACL-style 模态内对齐**，默认关闭：
  - CLI：`--use_uacl_intra_alignment`
  - 权重：`--w_uacl_intra`、`--w_uacl_kl`
  - 温度：`--uacl_temperature`
  - 文本侧复用 `probabilistic_text()` 的 Gaussian samples，视频侧复用 SAP `mu_video/logsigma_video` 采样。
- 旧的 `--use_hard_negative_packing` 保留为 legacy/diagnostic 路线，没有删除，避免影响 2026-06-19/22 的已完成对照实验。
- HN 相关脚本与开关保留，但不再建议用于新训练主线。

## 历史执行状态（2026-06-22，已被 2026-07-04 决策取代）

- B1 离线构建 raw hard-negative 映射已完成：`cache_dir/hard_negatives/msrvtt_train_hardneg.json`，共 180000 条。
- B2 hard-negative batch packing 已接入训练链路，开关为 `--use_hard_negative_packing`，当前 `main_task_retrieval.py` 的 `--hard_negative_path` 默认值已切到 clean 映射。
- raw HN packing 首轮实验已完成：`logs/20260619/hn_pack_wmil0_repeat1_4gpu_b64_train_msrvtt.log`，Best T2V R@1 = **48.1**，低于 B1-only v2 的 49.3，不能作为正向结果。
- raw map 审计发现明显假负例/近重复：exact caption pairs=4059，高风险 pairs=17637，hard caption/video 最大复用=504/547。
- 已生成 clean map：`cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json`，保留 163592/180000 条，exact caption pairs=0，hard caption/video 最大复用降至 39/93；审计报告见 `cache_dir/hard_negatives/msrvtt_train_hardneg_audit.md`。
- 当时下一步曾计划跑 **clean-HN packing + `w_mil=0`**；后续已完成 clean/map、explicit、model-mined 与 fixed/regressed 诊断，最终 HN 主线终止。

历史命令（仅作追溯，不再推荐启动）：

```bash
RUN_DATE=20260622 RUN_TIME=hn_pack_clean_wmil0_repeat1_4gpu_b64 CUDA_VISIBLE_DEVICES=1,2,3,4 EXPERIMENT_DESC="Clean HN packing, 4GPU global batch=64, w_mil=0" bash run_train_msrvtt_bg.sh --use_hard_negative_packing --hard_negative_path cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json --w_mil 0 --w_evidential 0 --w_neg_reg 0 --warmup_steps 500 --batch_size 64 --gradient_accumulation_steps 1
```

## 0. 目标

参考詹佳庆硕士论文第三章与 AAAI 2025 论文中的两个算法，将其低风险迁移到当前 UATVR 的 MSRVTT 训练链路中：

1. 基于查询的难负样本挖掘策略：利用文本查询之间的语义相似度与 BM25 字面相似度，为每条 caption 找到难负视频。
2. 不确定性感知数据增强算法：将文本/视频表示建模为 Gaussian，通过重参数化采样生成同语义增强视图，并加入模态内一致性约束。

当前优先级：难负样本方向已终止；下一步只验证不确定性感知增强 / UACL-style 模态内对齐是否能作为稳定辅助项。

## 1. 当前代码事实

- 主训练入口：`main_task_retrieval.py`
- 当前主模型：`modules/modeling_mulit.py`
- 当前主检索分数：`retrieve_logits = wti_logits`
- SAP 输出的 `mu_video/logsigma_video` 当前主要进入 MIL 采样与诊断，不是最终 eval 的主分数。
- 文本侧已有 Gaussian 采样路径：`probabilistic_text()`
- 视频侧已有 Gaussian 采样路径：`sample_gaussian_tensors(mu_video, logsigma_video, n_video)`
- dataloader 即使返回 attributes，当前训练循环也会丢弃属性分支输入；本计划不依赖 query branch/attributes。

## 2. 实施总路线

### Phase A：实验卫生与基线冻结

目的：避免把新算法效果和历史实验噪声混在一起。

- 等当前三个实验完成后，记录三组结果：
  - current B1only_v2 repeat
  - baseline pure sim `169ba95`
  - Exp1 repro `21348d1`
- 在开始实现前创建独立分支或 worktree。
- 先固定一个干净对照配置：
  - `--w_evidential 0`
  - `--w_neg_reg 0`
  - 先测试 `--w_mil 0` 与默认 `--w_mil 0.01` 的差异
- 顺手修实验基础设施：
  - `train_msrvtt.sh` 参数化 `MASTER_PORT`
  - 日志明确打印 shell batch、post-accum batch、per-GPU micro batch、effective batch

验收标准：

- 能稳定复现实验启动，不再出现端口冲突。
- 明确当前 B1only_v2 在 `w_mil=0/0.01` 下的差异。

## 3. Phase B：基于查询的难负样本挖掘

### B1. 离线构建 hard negative 映射

新增脚本：

- `scripts/build_msrvtt_hard_negatives.py`

输入：

- `--train_csv`
- `--data_path`
- `--output`
- `--dense_top_k`
- `--target_rank`
- `--max_words`

输出：

- 当前训练使用 clean 映射：`cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json`
- 原始未清洗映射仅作追溯：`cache_dir/hard_negatives/msrvtt_train_hardneg.json`

输出格式建议：

```json
{
  "caption_key_or_index": {
    "video_id": "video1234",
    "hard_caption_index": 5678,
    "hard_video_id": "video9876",
    "dense_rank": 34,
    "bm25_rank": 50,
    "dense_score": 0.72,
    "bm25_score": 8.31
  }
}
```

核心逻辑：

1. 读取训练 caption 与 video_id。
2. 为每条 caption 计算 dense embedding。
   - 第一版可用 CLIP text encoder 的冻结表示。
   - 若实现成本高，可先用 TF-IDF/BM25-only 做 smoke test，但正式实验应加入 dense cosine。
3. 对每条 caption 找 dense Top-K 候选。
4. 排除同一个 `video_id` 的候选，避免 MSRVTT 多 caption 同视频造成假负例。
5. 在候选集中用 BM25 重排。
6. 选择 `target_rank` 或前 10% 位置的候选作为 hard negative。
7. 保存映射和诊断统计。

诊断统计：

- hard negative 覆盖率
- 被排除的同 video 候选比例
- dense score 均值/分位数
- BM25 score 均值/分位数
- 每个 video 被选作 hard negative 的次数分布

### B2. 第一版接入：batch 内 hard negative packing

目标：不改模型 forward，只改变 batch 组成，让当前 CrossEn 分母自然包含 hard negative。

可选实现：

- 新增 `HardNegativeDistributedSampler`
- 或在 `MSRVTT_TrainDataLoader` 内提供 hard-negative-aware index order

原则：

- 一个 anchor caption 尽量和其 hard negative video 对应的样本进入同一 batch。
- DDP 下每个 rank 的 batch 仍保持大小一致。
- 若 hard negative 样本不可用，回退到普通随机采样。

优点：

- 不增加视频编码次数。
- 不改 WTI 和 loss。
- 归因最干净：只验证更高质量负样本是否提升主检索。

风险：

- DDP sampler 实现容易影响 shuffle 与 drop_last。
- 难负样本覆盖率可能受 batch size 限制。

验收标准：

- 单测确认同 batch 中 hard negative 命中率高于随机 batch。
- 训练首个 epoch 不出现 sampler 死循环或 batch size 变化。
- 日志打印 hard negative hit rate。

### B3. 第二版接入：显式 hard negative loss

如果 B2 提升不明显，再实现显式 hard negative 分母。

改动：

- dataloader 额外返回 hard negative video。
- 模型额外编码 hard negative video。
- 新增 loss：

```text
L_hn = -log exp(sim(q_i, v_i)/tau)
       / (exp(sim(q_i, v_i)/tau) + sum_inbatch_neg + sum_hard_neg)
```

参数：

- `--hard_negative_path`
- `--use_hard_negative`
- `--w_hard_negative`
- `--hard_negative_mode {batch_pack, explicit_loss}`

风险：

- 显存和训练时间增加明显。
- 如果 hard negative 假负例较多，可能损伤 R@1。

验收标准：

- small batch 前向通过。
- hard negative logits shape 与主 logits 对齐。
- loss finite，无 NaN。

## 4. Phase C：不确定性感知数据增强

### C1. 先复用现有 Gaussian，不新增多层池化

论文中使用四路多层 hidden pooling，但当前 CLIP `encode_text(return_hidden=True)` 只返回最终 token hidden，不返回所有层 hidden。第一版不建议改 CLIP Transformer 输出结构。

第一版实现：

- 文本侧复用 `probabilistic_text()` 的 `embedding` 和 `logsigma`
- 视频侧复用 SAP 的 `mu_video/logsigma_video`
- 从每个样本分布采样两个视图：
  - `z_text_a, z_text_b`
  - `z_video_a, z_video_b`
- 加模态内一致性对比：
  - `L_intra_text`
  - `L_intra_video`
- 加轻量 KL/variance 正则：
  - 第一版仅约束 `logsigma`，避免把 L2-normalized `mu` 强行拉向零先验

参数：

- `--use_uacl_aug`
- `--w_uacl_intra`
- `--w_uacl_kl`
- `--uacl_num_samples`
- `--uacl_temperature`

建议初始值：

- `w_uacl_intra=0.01`
- `w_uacl_kl=1e-4` 或 `1e-3`
- `uacl_num_samples=2`
- `uacl_temperature=0.03` 或沿用当前 logit scale 前的 cosine temperature

验收标准：

- loss finite
- `logsigma_text/video` 不长期贴住 clamp 下界
- `L_intra_text/video` 有正常下降趋势

### C2. 再考虑多粒度池化

如果 C1 有正向结果，再考虑实现论文的多粒度池化。

可能方案：

- 修改 `modules/module_clip.py` 的 Transformer，让其可选返回每层 hidden state。
- 文本侧构造 `[last_mean; first_last_mean; last2_mean; eot]`。
- 视频侧可构造 `[frame_mean; cls_mean; sap_mu; spatial_mean]` 的等价多粒度特征，而不是机械照搬文本层级。

该阶段工程风险较高，不作为第一批实现。

## 5. 实验矩阵

第一轮只跑 MSRVTT，固定 seed 与当前 B1only_v2 对齐。

| 实验 | 改动 | 目的 |
|------|------|------|
| HN0 | 当前 B1only_v2，`w_mil=0` | 建立干净主检索基线 |
| HN1 | HN batch packing，`w_mil=0` | 已验证不稳定，主线终止 |
| HN2 | HN batch packing，`w_mil=0.01` | 不再继续 |
| HN3 | explicit HN loss | 已验证净收益不足，主线终止 |
| UACL1 | `w_mil=0` + UACL intra/KL | 验证论文式不确定性增强是否优于 MIL |
| UACL2 | HN batch packing + UACL | 不再作为优先实验；HN 不参与主线组合 |

优先看指标：

- T2V R@1
- V2T R@1
- ret gap
- `pos-neg gap`
- hard negative hit rate
- `logsigma_v_mean`
- `var_text_mean`
- `L_intra_text/video`

## 6. 推荐执行顺序

1. 等当前三组实验完成，更新 `docs/project/STATUS.md`。
2. 修 `MASTER_PORT` 与 batch 日志。
3. 跑 `w_mil=0` 干净基线。
4. Hard negative 相关步骤已完成并终止，结果作为负消融。
5. 下一步只做 UACL intra/KL 的稳定性 repeat。
6. 只有在 UACL 有稳定收益后，才进入多粒度池化 C2。

## 7. 暂不做的事

- 暂不恢复 query branch/attributes 融合。
- 暂不把论文中的多层 hidden pooling 直接塞进 CLIP。
- 不再把 hard negative 与其他机制组合成新主线。
- 暂不在 MSVD 上同步实现，等 MSRVTT 有明确正收益后再迁移。

## 8. 预期结论形态

理想情况：

- Hard negative 已形成负结果：训练信号存在，但不能稳定提升 R@1。
- UACL intra/KL 若进一步提升，说明当前概率分支可以从 MIL 式跨模态采样转向论文式同模态一致性增强。

若结果不理想：

- HN 无提升：已完成诊断；核心问题是 hard negatives 与验证 Top-1 错误不够对齐，且 MSRVTT 存在语义近邻/多正例式歧义。
- UACL 无提升：说明当前概率方差未被可靠校准，应先简化概率支路或改主检索耦合方式。
