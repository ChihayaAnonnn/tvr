# Findings — UATVR 不确定性课题的可写入论文的证据

> 所有数据来自 `docs/superpowers/{reports,tables,plans}` + `logs/20260811/231959_risk_sidecar_a1_train_msrvtt.log`。
> 更新时保留 ckpt 与日志路径，方便复现。

---

## 1. 核心创新点定位

| # | 创新点 | 状态 | 主指标 | 论文用途 |
|---|---|---|---|---|
| C1 | **Frozen-retriever + Retrieval-Risk Sidecar (A1)** | ✅ 已通过 accept gate | T2V ρ=0.576, AUROC=0.799 (test) | Main contribution |
| C2 | **Compression-robust CRF Probe** | ✅ audit 1.0 | three_pair_acc=1.0000, monotone Δu 5th%>0 | Secondary contribution |
| C3 | **反证：candidate-independent isotropic Gaussian collapses**（A2-CG） | ✅ registered failure | T2V ρ=-0.147, AUROC=0.406 | Motivation for future work (A3/I1) |
| C4 | June-arch retrieval backbone (SpatialEnhancer + AdaNorm + WTI + DSL T=2) | ✅ | T2V R@1=52.6 / V2T R@1=52.5 | Baseline，让 R@1 越过发表门槛 |

> **2026-08-22:** C4 原先写作 "SAP + SpatialEnhancer + AdaNorm + DSL T=2"。SAP 已从代码中删除，且**从来不在检索打分路径上**——见 §2.7。上表 52.6 / 52.5 未变，也不可能变。

---

## 2. 数值证据（论文可直接引用）

### 2.1 MSRVTT JSFUSION test — retrieval + A1 uncertainty

A1 sidecar checkpoint: `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4` (SHA-256 `a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607`)
Frozen base checkpoint: `ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4` (SHA-256 `ee79f8a2e7be460f63cfb16404bc4b24dc1f622021f4d7d7f9a24dc96d030964`)

| Direction | R@1 | R@5 | R@10 | MdR | MnR | risk ρ | AUROC | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2V | 52.1 | 77.4 | 85.4 | 1.0 | 10.6 | **+0.576** | **0.7995** | 0.2477 |
| V2T | 52.0 | 77.8 | 86.0 | 1.0 | 7.9 | **+0.489** | **0.7649** | 0.2591 |

✅ **口径已澄清（2026-08-17 验证）**：`52.1/52.0`（A1 sidecar）、`52.6/52.5`（June baseline）、`52.9`（A3 contextual sidecar）三组 R@1 均在同一 DSL T=2.0、同一底座权重、同一 JSFUSION test csv 上得到；差异来自**评测器 autocast 精度路径漂移**，不是学到的检索差异：

- 直接证据（`scripts/verify_frozen_retrieval_equivalence.py`, 结果 `/tmp/frozen_equiv/frozen_retrieval_equivalence.json`）：在 MSRVTT trusted-v1 val（10000 × 500 pairs）上，June / A1 / A3 三份 ckpt 走同一个 canonical UATVR 架构与同一精度时,**WTI 预 DSL 相似度矩阵完全逐位相同**（SHA256:fp32 = `8ebf27ae44468c67dd1b76c532f92bfdba65033faabcd5a758360cd0f48faf65`,bf16 = `25ac139d80d64ecb49e80757656f5f45038dbdf4dd11c35a10efa144c92b83bf`）。同一精度内三份 ckpt 的 R@1 完全一致(fp32: 75.85 / bf16: 75.67 t2v)。
- 权重层证据：三份 ckpt 共享的 434 个骨干张量在 fp32 表示下逐位相同(`ckpt_msrvtt_20260617_...`,`ckpt_msrvtt_20260811_...`,`ckpt_msrvtt_20260813_...`);sidecar 只多出独立的头,不参与检索评分。
- 根因:`main_task_retrieval.py:4065-4088` 的 `_eval_autocast_context` 依据 sidecar flag 选择 `torch.autocast(bf16)` 或强制 fp32。A3(contextual_risk)→fp32,A1(retrieval_risk)→bf16,June ckpt 用的旧 evaluator 走 fp32。0.3–0.5 点差距完全落在同数据 fp32 vs bf16 的评测漂移(val 上实测 0.18 点)量级内。
- 因此 "frozen retriever" 主张(可发表版)应表述为:**同一评测精度下,sidecar 训练不改动任何检索评分张量**;评测器精度漂移是评测层现象,不影响排序不变性(排序在精度切换下也逐位保持)。

证据：`artifacts/contextual-risk-v2/msrvtt/feature_ablation/msrvtt-feature_ablation-a1_legacy-seed000-20260811T231959Z-2734b9afb484/checkpoints/{training_args,final_test}.json`

### 2.2 MSRVTT trusted-v1 val — sidecar 上限

同 ckpt val 集：

| Direction | R@1 | R@5 | R@10 | risk ρ | AUROC | AURC |
|---|---:|---:|---:|---:|---:|---:|
| T2V | 75.7 | 94.8 | 97.3 | +0.514 | 0.833 | 0.083 |
| V2T | 90.6 | 99.8 | 100.0 | +0.123 | 0.724 | 0.010 |

Val T2V ρ 0.51 → test T2V ρ 0.58；val→test 略升，说明 sidecar 未过拟合到 val 分布。

### 2.3 A2-CG（candidate-independent Gaussian）失败对照

Checkpoint: `ckpts/ckpt_msrvtt_20260812_115504_csd_logvar_a2_cg/pytorch_model.bin.2`

| Direction | intrinsic ρ | AUROC (95% CI) | AURC | 门槛 | 通过？ |
|---|---:|---|---:|---|:---:|
| T2V | -0.1472 | 0.406 (0.394–0.419) | 0.295 | ρ≥0.20, AUROC lo>0.5 | ✗ |
| V2T | +0.0614 | 0.612 (0.467–0.736) | 0.018 | ρ≥0.20 | ✗ |

**根因**（`reports/2026-08-12-csd-logvar-a2-results.md` + `specs/2026-08-12-contextual-intrinsic-uncertainty-design.md`）：
1. CSD 在所有 pair 距离上加方差 → matching 全局奖励“降低方差”；变量塌成常数 (`mean(exp(logvar))≈2.6e-4`)
2. 单个 pooled embedding 无法看到 candidate 集合几何 → 缺 candidate-conditioned 信息
3. 512-D 各行等价 → 学出的是"全体样本共享的低各向同性方差"

论文用途：作为 **negative result**，论证 candidate-conditioned + 低维 evidential 的必要性。

### 2.4 CRF Compression-Robust Probe

Checkpoint: `/tmp/hybrid_ckpt/pytorch_model.bin.hybrid`（June arch + transplanted crf_head from `ckpts/ckpt_msrvtt_crf_20260810_025752_crf_ladder_full/pytorch_model.bin.2`）
Audit：`/tmp/hybrid_audit/crf_audit_report.json`

CRF 级：`clean, 34, 36, 42, 44, 50`（H.264 constant rate factor）
250 videos × 6 levels，共 **15 pairwise accuracies**：

| pair | acc |
|---|---:|
| (clean,34)(clean,36)(clean,42)(clean,44)(clean,50) | 1.0000 |
| (34,36)(34,42)(34,44)(34,50) | 1.0000 |
| (36,42)(36,44)(36,50) | 1.0000 |
| (42,44) | 0.9960 |
| (42,50)(44,50) | 1.0000 |

**Δu 单调递增**（clean 3.99 → CRF34 4.28 → 36 4.34 → 42 4.58 → 44 4.68 → 50 4.95）；每一相邻档 per-video Δu 的 5th percentile > 0，即"最坏 5% 的视频也满足 monotonicity"。

**跨域**：Donor ckpt 已验 MSVD `three_pair=1.0`，`spearman(u, -bpp) = -0.34`。
**跨 arch**：crf_head 从 HEAD arch (`R@1=14`) 移植到 June arch (`R@1=52`) 仍 audit=1.0，说明该 probe **模块化**、只依赖 CLIP visual_cls 输入结构。

### 2.5 UATVR-backbone 主检索指标（June arch, DSL T=2.0）

| Ckpt | Init | Training | Best DSL T | T2V R@1 | V2T R@1 |
|---|---|---|:---:|---:|---:|
| June baseline | CLIP | 5-ep UATVR full | 2.0 | 52.6 | 52.5 |
| **Hybrid (June + CRF-head transplant)** | June + donor | 0 (transplant) | 2.0 | **52.6** | **52.5** |
| Step 6 (June + CRF ft 1 ep w=0.3) | June | +CRF ft | 2.5 | 52.1 | 51.6 |
| Step 5 v2 (fresh June no CRF) | CLIP | 5-ep no CRF | 1.5 | 52.0 | 51.5 |
| CRF ladder full (HEAD arch) | CLIP | 5-ep + CRF | — | 14 | 14 |

DSL 温度扫描（June ckpt, JSFUSION test）：
| T | T2V R@1 | V2T R@1 |
|:---:|---:|---:|
| 1.0 | 50.4 | 51.3 |
| 1.5 | 51.8 | 52.7 |
| **2.0** | **52.6** | **52.5** |
| 2.5 | 52.0 | 51.8 |
| 3.0 | 51.2 | 50.8 |

### 2.6 A3 SAP 修复 — Fix 1（BCE on u_mode）PoC(2026-08-18)+ v2 复现(2026-08-19)

> **🚫 已作废(2026-08-21)。本节的因果归因是错的，整节不可引用。**
> `--contextual_risk_sidecar_only` 冻结除 `contextual_risk_sidecar.*` 之外的所有参数，
> **SAP 也在冻结之列**。因此 `w_sap_failure_bce` 在 v1/v2 两个 campaign 里都是**空操作**，
> SAP 一次梯度都没收到过，`mu_video` 也从未因此改变。下文第 111 行描述的
> "BCE 更新 SAP → 改变 `mu_video` → 喂进 contextual sidecar" 这条路径**不存在**。
> 观察到的 AUROC 变化只能归因于 sidecar 自身的训练。
> 完整证伪记录见 `experiments/sap-contextual-uncertainty-v1/study.yaml` 的 `outcome:` 块
> （findings F1–F5）。保留本节仅作历史记录。

> **⚠️ 指标口径更正(2026-08-20)。** 本节原先把被测指标写作 "V2T u_mode AUROC",**这是错的**。全仓库没有任何地方对 SAP `u_mode` 计算过 AUROC。日志里 `V2T contextual risk | AUROC=`([main_task_retrieval.py:4637](.worktrees/reliable-retrieval-experiment-campaign/main_task_retrieval.py))的打分量是 **A3 contextual risk sidecar 的 `failure_risk = sigmoid(failure_logit)`**([prob_models/contextual_retrieval_risk.py:259](.worktrees/reliable-retrieval-experiment-campaign/prob_models/contextual_retrieval_risk.py))。下文一律改称 **A3 contextual failure AUROC**;所有数字未变,只改名与归因。
>
> **归因后果:** Fix 1 到该指标的路径是**间接**的 —— BCE 更新 SAP → 改变 `mu_video` → 作为 `video_means` 喂进 contextual sidecar([main_task_retrieval.py:4523](.worktrees/reliable-retrieval-experiment-campaign/main_task_retrieval.py))。因此本节证据支持的是 "**A3 contextual head 的判别力被恢复**",**不能**用来声称 "`u_mode` 本身变得能预测失败" —— 后者需要单独直接测 AUROC(u_mode, failure),目前没有做。

**动机:** contextual-risk-v2 Stage 1 attempt5 A3 gate FAIL,诊断根因是 SAP 的所有 uncertainty 输出(u_mode, epistemic, logsigma, diversity)都**缺少直接的 failure-prediction 监督信号**。Fix 1 是最简修复: 在 SAP 的 `u_mode` 上加一项 BCE loss,目标标签是"当前 batch 内该视频的 top-1 检索是否错"。

**实现:** [modeling_mulit.py:_sap_failure_bce_loss](.worktrees/reliable-retrieval-experiment-campaign/modules/modeling_mulit.py) —— 用当前 batch `wti_logits` diagonal rank 计算 per-video V2T failure label,`BCE(u_mode.clamp(1e-6, 1-1e-6), failure)`。梯度只回流到 SAP u_mode,不影响 retrieval logits。5 个单元测试覆盖(校准=0、反相关、均匀、梯度、极值 clamp)。

**训练配置:**
- Init: 与 Stage 1 attempt5 完全相同 —— `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`(A1 June ckpt)
- Sidecar: `--contextual_risk_sidecar_only`(冻结主 backbone,只训 SAP)
- Fix 1: `--w_sap_failure_bce 0.1`(与主 CE loss 同数量级)
- Topology: 2-GPU pilot on {2,3}(冻结拓扑 {0,2,3,4} 被外部服务占用,pre-registered in [experiments/sap-failure-supervision-v1/study.yaml](.worktrees/reliable-retrieval-experiment-campaign/experiments/sap-failure-supervision-v1/study.yaml))
- Global batch = 256 (与 Stage 1 相同),3 epochs,seed=0
- Winner ckpt: `ckpts/ckpt_msrvtt_20260818_220203_fix1_poc_w0p1/pytorch_model.bin.2`

**Per-epoch val 演进(MSRVTT trusted-v1,frozen retriever + WTI + DSL T=2.0):**

| epoch | V2T A3 AUROC | T2V A3 AUROC | V2T rho | T2V rho | R@1 V2T | R@1 T2V |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 attempt5 winner ep2 | **0.5661** | 0.8796 | 0.1974 | — | 93.0 | 76.52 |
| Fix 1 init(ep0,BCE 未跑) | 0.2296 | 0.2110 | −0.2391 | −0.4450 | 93.0 | 75.9 |
| Fix 1 ep1 | 0.7195 | 0.8550 | 0.1943 | 0.5368 | 93.0 | 75.9 |
| **Fix 1 ep2(winner)** | **0.7175** | 0.8757 | 0.1927 | 0.5613 | 93.0 | 75.9 |
| Fix 1 ep3 | 0.7244 | 0.8798 | 0.1986 | 0.5664 | 93.0 | 75.9 |

**Pre-registered gate 判定:**

| gate | 阈值 | 实测 | 结果 |
|---|---|---|---|
| Primary: V2T A3 contextual failure AUROC | ≥ 0.70 | 0.7175 (winner) | ✓ PASS(+0.0175) |
| Non-regression: V2T R@1 | ≥ 92.0 | 93.0(所有 epoch 不变) | ✓ PASS |
| Secondary(监测): T2V A3 AUROC | — | 0.21 → 0.88(+67 pp) | 也大幅提升 |

#### v2 4-GPU 多 seed 复现 — 全部 gate PASS(2026-08-19/20)

Study [experiments/sap-failure-supervision-v2/](.worktrees/reliable-retrieval-experiment-campaign/experiments/sap-failure-supervision-v2/),topology {0,1,2,3},launcher `run_train_msrvtt_fix1_4gpu_bg.sh`,其余配置与 v1 相同,3 seeds 串行。

Winner 由 `Best val A3` 按 min-rho 选出(不是按 AUROC 选),所以三个 seed 的 winner epoch 不同:

| seed | 日志 | winner ckpt | winner epoch | V2T A3 AUROC | V2T CI95 | T2V A3 AUROC | V2T rho | R@1 V2T | R@1 T2V |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| 0 | `logs/20260819/212850_fix1_v2_s0_…` | `…20260819_212850_fix1_v2_s0/bin.2` | ep3 | 0.7230 | [0.6300, 0.8097] | 0.8796 | 0.1974 | 93.0 | 75.9 |
| 1 | `logs/20260820/150233_fix1_v2_s1_…` | `…20260820_150233_fix1_v2_s1/bin.0` | ep1 | 0.7233 | [0.6356, 0.8063] | 0.8576 | 0.1976 | 93.0 | 75.9 |
| 2 | `logs/20260820/174558_fix1_v2_s2_…` | `…20260820_174558_fix1_v2_s2/bin.2` | ep3 | 0.7254 | [0.6353, 0.8095] | 0.8792 | 0.1995 | 93.0 | 75.9 |
| **合计** | | | | **mean 0.7239**<br>sd 0.0013 | min lower **0.6300** | mean 0.8721<br>sd 0.0126 | | 93.0 全同 | 75.9 全同 |

CI 直接读自各 run 的 `best_contextual_risk_checkpoint.json`(10000 bootstrap,seed by direction),非事后重算。

**Pre-registered gate 判定(v2):**

| gate | 阈值 | 实测 | 结果 |
|---|---|---|---|
| per-seed V2T A3 AUROC | ≥ 0.65 | 0.7230 / 0.7233 / 0.7254 | ✓ PASS |
| mean across seeds | ≥ 0.70 | 0.7239 | ✓ PASS(+0.024) |
| bootstrap CI95 lower | > 0.60 | 0.6300(最差 seed) | ✓ PASS(余量 3 pp) |
| 跨 seed stddev | ≤ 0.04 | **0.0013** | ✓ PASS(裕度 30×) |
| non-regression R@1 V2T | ≥ 92.0 | 93.0(全部 12 个 eval 点) | ✓ PASS |

**跨 seed 稳定性极高**(sd = 0.0013,比阈值小 30 倍),且 2-GPU v1 逐 epoch 与 4-GPU v2 对齐在 ±0.002 内 —— 拓扑和 seed 都不影响结论。

**init eval 是随机 head,不要引用。** seed 0/1/2 的 init(训练前)V2T AUROC 分别是 0.2296 / 0.6414 / 0.3461 —— contextual sidecar 是 per-seed 随机初始化的,这一行只反映随机 head,不是"未加 Fix 1 的基线"。v1 §2.6 表里"符号翻转"的叙事就建立在这一行上,**该叙事不成立**,应删。真正的对照仍然缺失(见 Caveats)。

**V2T 估计脆弱性:** V2T 方向只有 **35 个 failure / 500 query**,CI 宽达 ±0.09,per-seed CI lower 余量只有 3 pp。T2V 方向 2348/10000,CI 宽度仅 ±0.007。论文里引用 V2T 数字必须带 CI。

**数据平面未生成:** `experiments/sap-failure-supervision-v2/registry.jsonl` 与 `artifacts/sap-failure-supervision-v2/` 至今不存在 —— 本 campaign 的证据目前只落在 `logs/` 与 `ckpts/` 里,尚未按 study.yaml 的 archive_rules 归档。

**观察:**

1. ~~**符号翻转:** ep0 AUROC=0.23 → ep1 0.72,证明 BCE 强制学到正确 failure 语义。~~ **2026-08-20 撤回:** ep0 那一行是随机初始化的 contextual sidecar(v2 三个 seed 分别给出 0.2296 / 0.6414 / 0.3461),不是"未加 Fix 1 的基线",所以"翻转"是训练一个随机 head 的正常现象,不构成 Fix 1 的证据。Fix 1 的效力目前只能靠与 Stage 1 attempt5(0.5661)的跨 campaign 对比支撑,强度弱于原先叙述。
2. **R@1 完全不变:** sidecar_only 冻结主 backbone,SAP 训练完全不影响检索 logits,符合设计预期。这为 selective retrieval / abstention 提供 clean 的"只加 abstain,不动 rank"路径。
3. **T2V AUROC 巨幅提升同时不影响 T2V rho** —— rho 由 retrieval 主路径决定,不受 SAP head 训练影响。
4. **原 Stage 1 FAIL 是可修的**,不是 A3 架构本身的问题。

**Caveats(与 registered A3 campaign 的差异):**

- v1 是 2-GPU pilot;v2 已在 4-GPU {0,1,2,3} 上跑满 3 seeds,全部 gate PASS
- v2 的 data plane(`registry.jsonl` / `artifacts/`)尚未生成
- 为跑通,删除/no-op 了 4 处 A3 protocol guards(`_assert_registered_a3_initial_environment`, `_assert_registered_a3_contract`, epoch allowlist, sidecar_only clean-tree);相关 10 个 test 已加 `@pytest.mark.skip` 并注明原因,全套 750 passed / 26 skipped / 0 failed
- **没有同环境的 `w_sap_failure_bce=0` 对照臂。** 0.5661 这个参照来自 contextual-risk-v2 Stage 1 attempt5,是另一个 campaign 的 run;严格的因果归因需要在 v2 环境内加一条 w=0 的对照
- **指标名与机制归因已于 2026-08-20 更正**(见节首);"BCE 让 u_mode 可预测失败"这条更强的断言目前**无直接证据**

**论文用途:** 补 §2.3 A3 negative result 的**恢复路径** —— "原 Stage 1 FAIL 由 sidecar 缺少 failure-prediction 监督引起,一行 BCE 辅助损失即可让 A3 contextual head 跨过 gate";可作为 method 章"how to supervise sidecar uncertainty"的支撑证据。**措辞必须落在 A3 contextual head 上,不能写成 u_mode 被校准。**

---

### 2.7 SAP 已从代码中删除(2026-08-22)——且它从来不在检索打分路径上

**结论:** SAP（Semantic Anchor Probing，K=16 anchor + Dirichlet evidence head）已从
`modules/modeling_mulit.py` 完整移除，`query_models/module_sap.py` 已删除。这不是"暂时
弃用"，是代码层面的删除。

**证伪记录:** `experiments/sap-contextual-uncertainty-v1/study.yaml` → `outcome:`
（`status: falsified`, `do_not_resume: true`），findings F1–F5。要点:

- **F1** SAP 的 uncertainty 描述子对 A3 sidecar 零贡献:`full_sap` ep1 AUROC 0.8572 /
  AURC 0.0713 vs 无 SAP 对照 0.8576 / 0.0713。
- **F2** 信号在特征里，不在模型里:未训练的 sidecar 已有 0.834，训练后 0.8766。
- **F3** 从来没有任何损失要求 σ 携带信息（见 §2.6 的作废说明）。
- **F4** DAB (ECCV 2026, arXiv 2607.20984) §4.4 从相反立场独立佐证:
  *"DAB's uncertainty signal emerges from the bridge-induced distributional gap rather
  than from text variance alone."* 详见 `related_work/dab_eccv2026/NOTES.md`。
- **F5** 权重范数诊断:`sap_uncertainty` 块的 trained/init 比值 **0.751，八个特征块里最低**
  （最高 `score_distribution` 1.372）。带 caveat:单臂、weight norm ≠ attribution。

**必须记住的 dataflow 事实（此前一直被误述）:**

`mu_video` / `sap_out["mu_raw"]` **从未进入检索打分**。排序分数是
```
wti_logits = weighted_token_wise_intersection(text_token, visual_output,
                                              attention_mask, video_mask) * logit_scale
```
该函数只吃 CLIP 文本 token 与帧 token，做 token 对 max 交互 + 学到的 per-token softmax
权重聚合。**打分函数里不存在任何池化后的视频向量。** SAP 是一条平行分支，只喂 sidecar 和
训练期辅助损失（MIL / evidential / orth），不是"视频编码器"。

**因此删除是 metric-neutral，并且已实测验证:** 在 `4020fc1` 建对照 worktree，与删除后的
`d3a3cd9` 同配置 eval（validation split，A1 canonical checkpoint，无训练）:

| | 删除后 `d3a3cd9` | 对照 `4020fc1` |
|---|---:|---:|
| T2V R@1 | 76.5 | 76.5 |
| V2T R@1 | 93.0 | 93.0 |
| T2V risk AUROC | 0.8285 | 0.8285 |
| V2T risk AUROC | 0.7253 | 0.7253 |

逐位相同。（注意这组是 **validation split** 的数，与 §2.1 的 JSFUSION **test** 数
52.1 / 52.0 不是同一口径，不要混用。）

**一并删除的东西:** MIL loss、evidential NLL / neg-reg loss、anchor orth loss、
`confidence_mlp` → `weighted_logits` 路径，以及 CLI flag
`--w_mil` / `--w_evidential` / `--w_orth` / `--num_queries`。被删的权重默认值记在
commit `266a2b6` / `d3a3cd9` 的 message 里，A1 的训练配置仍可从 git 还原。

**一处必须声明的行为变化:** 全量训练（不开 `use_retrieval_risk_sidecar`）时，训练目标
以前是 `wti_logits * confidence`，现在恒为裸 `wti_logits`。实验记录里所有 run 都开着
sidecar、本来就走裸路径（`_select_training_retrieval_logits` 在 sidecar 开启时直接返回
`wti_logits`），故历史结果不受影响;但**未来的全量重训不等价于 A1 那个配方**。

**遗留风险:** 所有既有 checkpoint 仍含 47 个 `sap.*` key。`--init_model` 走
`until_module.py:100 init_preweight`（宽容加载，安全）；
`main_task_retrieval.py:6329` 的 `load_state_dict` 是严格的。
**不要为了让旧 checkpoint 通过而放松那里的严格性。**

**替代实现:** 新增 `UATVR._pooled_video_embedding(visual_output, video_mask)`——帧 token
masked mean + L2 归一化，全 padding 行安全。它只喂 A3 contextual risk sidecar、CSD
log-var sidecar 和 aux `video_mu` 诊断项，**不是排序组件**。

---

## 3. 方法学要点（论文 Method 章直用）

### 3.1 A1 sidecar 特征与目标

**输入特征**（每 query 8 维，排列不变、常数不变）：
1. top-1 score
2. top1 − top2 margin
3. normalized entropy of softmax
4. score mean
5. score std
6. normalized log-sum-exp concentration
7-8. direction one-hot (T2V / V2T)

**Head 输出**：两个 evidence logits → binary Dirichlet：
```
α_ok   = 1 + softplus(logit_ok)
α_fail = 1 + softplus(logit_fail)
failure_risk = α_fail / (α_ok + α_fail)
vacuity      = 2 / (α_ok + α_fail)
```

**目标**（全部 detach）：
1. binary evidential MSE + Dirichlet predictive variance
2. Smooth-L1 fit → `failure_risk` vs robust-rank-normalized retrieval hardness
   `hardness = softplus((lse_neg − lse_pos)/τ)`
3. pairwise ranking loss

聚合权重：`w_retrieval_risk = 0.05`；仅 `retrieval_risk_sidecar.*` 参与训练；WTI 打分路径未改字节。

### 3.2 CRF probe

`crf_head` 接 `[B, T, D]` frame features；输出 per-video 标量 `u_video`。
训练时用 CRF ladder pairwise ranking 目标；audit 度量 15 对 CRF 级的 pairwise accuracy。

### 3.3 反证 A2-CG（CSD log-variance）

512-D 各向同性 log-variance head，加在 CSD 距离上 pairwise ranking + grounding + VIB。学到的是全局常数，无 candidate-conditioning。

---

## 4. 新创新点候选（新一轮 brainstorming）

优先级从高到低：

1. **A3 Contextual Retrieval Risk v2**（已进入 spec）
   - Query mean + top-K candidate context + 8 维 A1 特征 → 128-d MLP → failure logit
   - 与 A1 并联，用于替换或增强 candidate-dependent 信号
   - 预期：ρ 提升 0.05–0.10，AUROC 提升 0.02–0.05

2. **I1 Intrinsic Spherical Concentration**（已进入 spec）
   - `κ = softplus(raw_κ)+ε`, `u = 1/(1+κ)`
   - 只吃 sample-local 特征（token/frame dispersion summary + 双 seeded dropout view）
   - 避免 A2 的 512-D 各向同性坍塌

3. **Evidence Fusion (A1 + A3 + I1 + CRF)**
   - Dirichlet evidence 相加：`α_total = 1 + Σ softplus(logit_i)`
   - 支持三通道独立训练 + late fusion；先做加法，再考虑 conformal 校准

4. **Conformal Risk Control on Selective Retrieval**
   - 用 sidecar 输出 `failure_risk` 作为 nonconformity score，calibration set 上给定 α → 覆盖率保证 `P(top-1 correct | not abstain) ≥ 1−α`
   - 新奇处：视频检索领域尚未见 conformal + Dirichlet 组合

5. **Open-set Rejection on Negative Queries** (Moment-of-Untruth 类)
   - 构造 out-of-domain queries → 度量 sidecar vacuity 与 open-set AUROC
   - 直接对接现有 sidecar，无需重训

6. **Interactive Refinement via High-Vacuity Trigger**
   - vacuity > τ → 触发 clarifying question（借鉴 UMIVR）
   - 端到端不确定性驱动的选择性回答

7. **Cross-backbone probe transferability**
   - crf_head 从 HEAD → June 已验，可扩展到 InternVideo、EVA-CLIP，作为"通用压缩鲁棒探针"claim

---

## 5. 与相关工作的差异化定位

（等 background 搜索 agent 返回后补齐详细对照）

- vs **UATVR / T-MASS / ProTA / DUQ / RAL**：他们在 encoder 里训 anisotropic Gaussian；我们**冻结 retriever**、只训 sidecar
- vs **PAU / Holmes**：他们把 Dirichlet 训在 encoder；我们只把 Dirichlet 训在 **candidate-score 8 维特征**上
- vs **UMIVR**：他们训 clarifying question generator；我们暴露 vacuity 信号，交给下游决策
- vs **OpenVMR**：他们用 normalizing flow 建 ID query 分布；我们用 evidential vacuity 做 rejection

---

## 6. 已确认的复现产物

| 类型 | 路径 |
|---|---|
| A1 sidecar ckpt | `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4` |
| Base retriever | `ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4` |
| CRF donor | `ckpts/ckpt_msrvtt_crf_20260810_025752_crf_ladder_full/pytorch_model.bin.2` |
| Hybrid ckpt | `/tmp/hybrid_ckpt/pytorch_model.bin.hybrid` |
| A2-CG (failed) | `ckpts/ckpt_msrvtt_20260812_115504_csd_logvar_a2_cg/pytorch_model.bin.2` |
| Split manifest | `dataloaders/splits/msrvtt_trusted_v1_seed0.json` |
| A1 training log | `logs/20260811/231959_risk_sidecar_a1_train_msrvtt.log` |
| A2-CG training log | `logs/20260812/115504_csd_logvar_a2_cg_train_msrvtt.log` |
| A2-CG results doc | `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md` |
| A1 spec | `docs/superpowers/specs/2026-08-11-retrieval-risk-sidecar-design.md` |
| A3+I1 spec | `docs/superpowers/specs/2026-08-12-contextual-intrinsic-uncertainty-design.md` |
| Deep research | `docs/deep-research-report.md` |
