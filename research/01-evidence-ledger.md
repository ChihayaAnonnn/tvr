# 01 · Evidence Ledger —— 已确立的实证事实

> 缩写：`WT/` = `.worktrees/reliable-retrieval-experiment-campaign/`（`feature/reliable-retrieval-experiment-campaign` 分支的 worktree，
> 其 `ckpts/`、`logs/`、`experiments/` 与主仓库根目录**是不同的物理目录**，找不到 fix1 ckpt 时先看这里）。
>
> 状态标记：
> - **SOLID** —— 数字与归因都可复现、无已知反证。
> - **CAVEATED** —— 数字本身可信，但**归因、口径或统计功效有明确限制**，引用时必须带 caveat。
> - **RETRACTED** —— 已被后续证据推翻，**不可引用**（保留仅为防止重蹈）。

---

## A. Frozen-retriever 契约与 backbone

| ID | 事实 | 数字 | Split / Protocol | Artifact | 状态 |
|---|---|---|---|---|---|
| E-01 | 三份 ckpt（June base / A1 sidecar / A3 contextual）在**同一 canonical 架构 + 同一精度**下，pre-DSL WTI 相似度矩阵**逐位相同** | fp32 SHA-256 `8ebf27ae44468c67dd1b76c532f92bfdba65033faabcd5a758360cd0f48faf65`；bf16 SHA-256 `25ac139d80d64ecb49e80757656f5f45038dbdf4dd11c35a10efa144c92b83bf`；矩阵 shape 10000×500 | MSRVTT trusted-v1 **val**，DSL T=2.0，n_pairs=10000 | `scripts/verify_frozen_retrieval_equivalence.py`；结果 `/tmp/frozen_equiv/frozen_retrieval_equivalence.json`（⚠️ 在 /tmp，易失）；`PAPER.md:53`；`findings.md:35` | **SOLID** |
| E-02 | 同一精度下三份 ckpt 的 val R@1 完全一致 | fp32: T2V 75.85 / V2T 93.00；bf16: T2V 75.67 / V2T 90.559 | trusted-v1 val | `/tmp/frozen_equiv/frozen_retrieval_equivalence.json`（本 session 直接读取校核） | **SOLID** |
| E-03 | 434 个共享 backbone 张量在 fp32 表示下逐位相同 | 434/434 | — | `PAPER.md:53`；`findings.md:36` | **SOLID** |
| E-04 | 52.6 / 52.1 / 52.9 三组 R@1 的差异来自 **evaluator autocast 精度路径漂移**，不是学到的检索差异 | val 上实测精度漂移 0.18 点；test 上 ≤0.5 点 | — | `main_task_retrieval.py:4065-4088`（`_eval_autocast_context`）；`PAPER.md:53`；`findings.md:33-38` | **SOLID** |
| E-05 | `mu_video` / `sap_out["mu_raw"]`（SAP 池化视频向量）**从未进入检索打分**；排序完全由 `weighted_token_wise_intersection`（text token × frame token）产生 | — | 架构事实 | `findings.md:226-234`；`WT/experiments/sap-contextual-uncertainty-v1/study.yaml:390-403`（F3，`retrieve_logits = logit_scale * wti_logits`） | **SOLID** |
| E-06 | 因此删除 SAP 是 metric-neutral，eval-only 复现逐位不变 | T2V R@1 76.5 / V2T R@1 93.0；T2V risk AUROC 0.8285 / V2T 0.7253（删除后 `d3a3cd9` 与对照 `4020fc1` 完全相同） | trusted-v1 **val**，A1 canonical ckpt，无训练 | `findings.md:239-246` | **SOLID** |

---

## B. June baseline 检索指标（backbone sanity）

| ID | 事实 | 数字 | Split / Protocol | Artifact | 状态 |
|---|---|---|---|---|---|
| E-07 | June-arch UATVR baseline，DSL 温度扫描 | T=1.0 → 50.4/51.3；T=1.5 → 51.8/52.7；**T=2.0 → 52.6/52.5**；T=2.5 → 52.0/51.8；T=3.0 → 51.2/50.8（T2V/V2T R@1） | MSRVTT **JSFUSION test** 1000×1000 | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:123-131`；`findings.md:100-107`；`PAPER.md:160`（Fig.5） | **SOLID** |
| E-08 | DSL rerank 本身贡献约 +2.6 T2V R@1（raw 50.0 → DSL T=2 52.6） | +2.6 | JSFUSION test | `docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md:31-35`；`docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md:86` | **SOLID** |
| E-09 | CRF/June 消融矩阵 | June baseline 52.6/52.5 (T=2.0)；Hybrid transplant 52.6/52.5 (T=2.0)；Step6 June+CRF ft 1ep 52.1/51.6 (T=2.5)；Step5v2 fresh no-CRF 52.0/51.5 (T=1.5)；CRF ladder full (HEAD arch) 14/14 | JSFUSION test | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:227-233`；`findings.md:92-98`；`PAPER.md:149-155` | **SOLID** |

---

## C. A1 sidecar（candidate-conditioned Dirichlet，8 维 score 特征）

| ID | 事实 | 数字 | Split / Protocol | Artifact | 状态 |
|---|---|---|---|---|---|
| E-10a | A1 在 **JSFUSION test** 上的 uncertainty 指标（registered artifact） | T2V ρ=0.5759868, AUROC=0.7994542 (CI 0.7728–0.8257), AURC=0.2477, num_failures 479/1000；V2T ρ=0.4892371, AUROC=0.7649404 (CI 0.7365–0.7952), AURC=0.2591, num_failures 474/1000 | JSFUSION test；检索 R@1 T2V 52.1 / V2T 51.976 | `artifacts/contextual-risk-v2/msrvtt/feature_ablation/msrvtt-feature_ablation-a1_legacy-seed000-20260811T231959Z-2734b9afb484/checkpoints/final_test.json`（本 session 直接解析）；`PAPER.md:107-108`；`findings.md:30-31` | **SOLID** |
| E-10b | A1 在 **trusted-v1 val** 上 | T2V R@1 75.7, ρ +0.514, AUROC 0.833, AURC 0.083；V2T R@1 90.6, ρ +0.123, AUROC 0.724, AURC 0.010 | trusted-v1 val | `PAPER.md:114-115`；`findings.md:46-49` | **CAVEATED** —— 这组 R@1（75.7 / 90.6）是 **bf16** 路径的值；fp32 路径同 ckpt 为 75.85 / 93.00（E-02）。V2T ρ=0.123 **未过 0.20 gate**（`PAPER.md:135`）。 |
| E-10c | A1 selective retrieval（25% coverage） | T2V accuracy_at_0.25 = 0.880（vs accuracy_at_1.00 = 0.521，+35.9 pp）；V2T accuracy_at_0.25 = 0.876（vs accuracy_at_1.00 = 0.526，+35.0 pp） | JSFUSION test，risk 模块口径（**分母 1000**） | 同 E-10a 的 `final_test.json`；`PAPER.md:139` | **CAVEATED** —— 见下方 **D-01 分母缺陷**。 |
| E-10d | A1 只训 sidecar 参数 | 263,426 个可训练标量（A2-CG 报告口径）；`PAPER.md:79` 写 "263 K sidecar parameters" | — | `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:11`；`PAPER.md:79` | **CAVEATED** —— 263,426 是 **A2 sidecar** 的参数量；A1 的确切参数量本 session 未独立核实，`[未溯源]`。 |

### ⚠️ A1 test 指标存在三组互不相同的数字（口径冲突）

| 来源 | 条件 | T2V ρ | T2V AUROC | V2T ρ | V2T AUROC |
|---|---|---:|---:|---:|---:|
| `final_test.json` / `PAPER.md:107` | A1 自身 final_test，DSL T=2.0 | 0.5760 | 0.7995 | 0.4892 | 0.7649 |
| `docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md:99-100` | A3 run 的同一次 test pass，**DSL=false** | 0.5887 | 0.8007 | 0.5560 | 0.8057 |
| `docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md:43-44` | A3 winner + 固定 DSL T=2 的 post-hoc pass | 0.5623 | 0.7895 | 0.4913 | 0.7641 |

**解释（部分）：** A1 的 rank/failure 标签依赖 directional DSL 检索矩阵，所以 DSL 开关会改变 A1 指标
（`docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md:48`）。**论文里必须只引第一行，并注明 DSL T=2.0。**

---

## D. 已知开放缺陷（必须在投稿前解决）

| ID | 缺陷 | 证据 | 状态 |
|---|---|---|---|
| **D-01** | **1000 vs 1012 V2T 分母不一致（未解决）。** risk 模块按 1000 个 query 归一化 V2T（`num_queries: 1000`，`accuracy_at_1.00 = 0.526`），而检索指标按 1012 归一化（`cols` 长度 1012，`R1 = 51.976 = 526/1012`）。同分子、不同分母。`PAPER.md:139` 的 "+35.0" 在 coverage 曲线内部自洽，但**与 Table 1 的 V2T R@1 = 52.0 不可比**。 | `artifacts/.../a1_legacy-.../checkpoints/final_test.json`（本 session 直接核对：`retrieval_risk/v2t/num_queries=1000` vs `v2t/cols` len=1012）；`PAPER.md:141` | **OPEN DEFECT** |
| **D-02** | **`PAPER.md:95` 的 base ckpt SHA-256 写错了。** 它写 base ckpt `ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4` 的 SHA 是 `a2c074…4e2607`，但本 session 实测：该 base ckpt 是 `ee79f8a2e7be460f63cfb16404bc4b24dc1f622021f4d7d7f9a24dc96d030964`；`a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607` 是 **A1 sidecar ckpt** 的 SHA。`findings.md:25-26` 是对的。 | 本 session `sha256sum` 实测；`findings.md:25-26`；`docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:9` | **OPEN DEFECT** |
| **D-03** | **`PAPER.md:51` 写 backbone 是 CLIP-ViT-B/32，实际是 ViT-B/16。** 两份 `training_args.json` 均为 `pretrained_clip_name: ViT-B/16`；A3 报告也记 canonical CLIP asset = `ViT-B/16`（SHA-256 `5806e77c…df416f`）。 | 本 session 读取 `ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/training_args.json` 与 `ckpts/ckpt_msrvtt_20260617_b1only_v2_repeat1/training_args.json`；`docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md:10` | **OPEN DEFECT** |
| **D-04** | **52.1 / 52.6 / 52.0 三个 V2T-相关 R@1 的散布**：`PAPER.md:25` 同时把 52.1 与 52.6 当作 full-coverage baseline，`PAPER.md:107-108` Table 1 又给 V2T R@1 = 52.0。三者分别是：T2V coverage 分母 1000 (0.521)、V2T coverage 分母 1000 (0.526)、V2T 检索指标分母 1012 (51.976)。**同一段文字里混用了三种口径。** | `PAPER.md:25,107-108,139-141`；`final_test.json` | **OPEN DEFECT**（是 D-01 的表现） |
| **D-05** | `PAPER.md:51` 称 June baseline "V2T R@1 = 52.5"，而 `PAPER.md:25` 与 `PAPER.md:139` 把 "52.6" 当作 V2T full-coverage baseline。前者是 JSFUSION 检索指标，后者是 risk 曲线 accuracy_at_1.00。**不要混用。** | `PAPER.md:51,25,139` | **OPEN DEFECT** |
| **D-06** | **`findings.md:209` 声称 "`query_models/module_sap.py` 已删除。这不是'暂时弃用'，是代码层面的删除"，与 master 实况矛盾。** 本 session 实测：master 上该文件存在（7798 字节）且被 `modules/modeling_mulit.py:21`、`scripts/evaluate_sap_uncertainty.py:183` import。删除只在未合并分支 `feature/reliable-retrieval-experiment-campaign` 上。`findings.md §2.7` 需补一句分支限定。 **2026-08-24 已消解**：研究者选择合并（ADR-10），master 已 fast-forward 到 `d3a3cd9`，`module_sap.py` 实际删除，全仓库无残留引用；`findings.md:209` 的表述现在与 master 相符。 | 本 session `git ls-tree` / `ls -l` / `grep` 实测；`findings.md:206-211`；2026-08-24 `git merge --ff-only` | ~~OPEN DEFECT~~ → **RESOLVED (2026-08-24)** |

---

## E. A2-CG（candidate-independent 512-D isotropic Gaussian，CSD 目标）—— 注册的负结果

| ID | 事实 | 数字 | Split / Protocol | Artifact | 状态 |
|---|---|---|---|---|---|
| E-20 | A2-CG 在 trusted-v1 val 上**失败每一个 acceptance gate** | T2V intrinsic ρ = **−0.1472380481**, AUROC = **0.4063919497** (CI 0.3939–0.4186), AURC 0.2946；V2T ρ = **+0.0613642730**, AUROC 0.6123045332 (CI 0.4673–0.7361), AURC 0.0177 | trusted-v1 val，seed 0，effective batch 256，4 GPU，FP32 | `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:56-58`；`findings.md:57-60`；`PAPER.md:127-133` | **SOLID** |
| E-21 | A2-CG 的 selective accuracy **非单调**（去掉高不确定 query 后 T2V 准确率反而变差） | T2V acc@25/50/75/100 = 0.6788 / 0.7168 / 0.7312 / 0.7588（**递增**） | trusted-v1 val | `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:57,69` | **SOLID** |
| E-22 | 方差塌成全局常数 | text variance mean 2.602e-4 (std 2.49e-6)；video variance mean 2.209e-4 (std 3.37e-9)；prediction std 2.5e-5（`PAPER.md:132`） | trusted-v1 val | `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:60-66` | **SOLID** |
| E-23 | grounding 未学会 = 0.1669，pairwise ranking 停在 `ln 2 = 0.6931`（完全未学） | Ground fit 0.1669；Ground rank 0.6931（5 个 epoch 全同） | 训练日志 | `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:35-39`；`PAPER.md:87` | **SOLID** |
| E-24 | 选中的 A2 ckpt **完整保留了 A1**（frozen boundary 审计通过，non-A2 mismatch 列表为空） | ckpt SHA-256 `e539108aa8296eaf1e3c71d968117b5f153cd8d7c444282c8b39e9998c8635be`（本 session 实测一致） | trusted-v1 val | `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:27` | **SOLID** |
| E-25 | A2-CG 在 JSFUSION test 上同样失败（**描述性，协议偏离**） | T2V ρ −0.164629, AUROC 0.427073, AURC 0.5335；V2T ρ 0.028813, AUROC 0.513301 | JSFUSION test | `docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md:75-78` | **CAVEATED** —— JSFUSION 被评了**两次**（continuation 前后），是 one-shot test hygiene 的协议偏离；仅供描述，不影响 val-only 的失败结论。 |

---

## F. A3 contextual risk sidecar —— **两次结论相反的评测，必须并列记住**

| ID | 事实 | 数字 | Split / Protocol | Artifact | 状态 |
|---|---|---|---|---|---|
| E-30 | **A3 v1（2026-08-12/13）在 JSFUSION test 上 PASS**，且在 AUROC/AURC/coverage 上全面优于 A1 | A3: T2V ρ 0.5591 / AUROC 0.816244 / AURC 0.2499；V2T ρ 0.5431 / AUROC 0.828955 / AURC 0.2711。对 A1 的 Δ：AUROC +0.0156 (T2V) / +0.0232 (V2T)；AURC −0.0130 / −0.0173 | JSFUSION test，**DSL=false**（raw score path），one-shot | `docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md:90-116`；winner ckpt SHA-256 `2e6f662bc69dda9b880bf882195247cdd1394de6ecb6865c48148cdfb0d85a8c` | **CAVEATED** —— 该 test 未开 DSL；val 上 V2T ρ 仅 0.2007（勉强过 0.20 gate，靠 0.000069 的差距选中 epoch 3）。**已被 E-31 在 trusted-v1 val 上推翻。** |
| E-31 | **A3 Stage 1（2026-08-18）gate FAIL** —— 与 zero-parameter baseline 对比后不成立 | `a3_improves_strongest = false`：mean AUROC Δ = **−0.0287**，mean AURC Δ = **+0.0019**；`directional_rho_gte_0_20 = false`：V2T ρ = **0.19735** < 0.20（T2V ρ = 0.5661）。`next_authorized_stage = null` | MSRVTT trusted-v1 val，500 videos × 20 captions = 10000 pairs；failures T2V 2348/10000, V2T 35/500 | `WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md:11-30`；`todo.md:14-17`；winner ckpt SHA-256 `e503946f0c7d861a2ea6e3e49eecb30e8f51ecb67a83e994fd1d8a26938d565f` | **SOLID** |
| E-32 | **零参数 entropy scorer 击败 A3**（Stage 1 的核心发现） | entropy: T2V AUROC 0.8717 / ρ 0.5591；V2T AUROC **0.7883** / ρ **0.2549**；mean AUROC **0.8300** / mean AURC **0.0441**。A3: mean AUROC 0.8013 / mean AURC 0.0460。one_minus_top1: mean AUROC 0.8200。score_variance（方差 scorer）: mean AUROC **0.4411**（低于随机） | trusted-v1 val | `WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md:26-34` | **SOLID** —— **这是 north-star thesis 最强的自有证据之一。** |
| E-33 | A3 v1 与 Stage 1 结论相反的原因（官方解释） | trusted split 的 V2T 只有 500 query / 35 failure，高方差 regime；JSFUSION test 有 531 failure / 1000 query | — | `WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md:71-78`；`docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md:119` | **CAVEATED** |

---

## G. SAP（Semantic Anchor Probing）—— 已证伪并从代码中删除

| ID | 事实 | 数字 | Split / Protocol | Artifact | 状态 |
|---|---|---|---|---|---|
| E-40 | SAP 的 4 个 per-video uncertainty 信号训练后**全部塌成常数** | `logsigma_video` −1.5 ± 0（触底）；`epistemic_cont` 0.9995 ± 5e-5；`diversity` 0.9996 ± 5e-5；`modal_entropy` 1.0000 ± 0；`u_mode` 0.5794 ± 2e-5 | JSFUSION test 1000 videos | `docs/superpowers/reports/2026-08-11_sap_collapse_investigation_and_fix.md:23-32` | **SOLID** |
| E-41 | CRF 训练**主动抹掉**了 u_video 与检索难度的相关性 | ρ(u, max rank)：bin.0 +0.065 → bin.4 +0.015；同期 u_video std 0.160 → 0.197 | JSFUSION test | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:7-13`；`docs/superpowers/reports/2026-08-11_sap_collapse_investigation_and_fix.md:36-42` | **SOLID** |
| E-42 | **SAP 的 5 个 raw uncertainty 描述子对检索失败的直接预测力 = 随机** | V2T direct AUROC：`u_mode` **0.5441**；`modal_entropy` 0.5397；`epistemic_scale` 0.5138；`aleatoric_scale` **0.4734**；`log_evidence` **0.4561**（后两者弱反预测）。gate 阈值 ≥0.55，**全部 fail** | trusted-v1 val，无训练 head 介入（`_sap_uncertainty_direct_auroc()`） | `WT/experiments/sap-contextual-uncertainty-v1/study.yaml:316-329` | **SOLID** |
| E-43 | 把 `sap_uncertainty` 特征块接进 A3 sidecar，**收益为零** | control（3 seed）T2V AUROC 0.8576–0.8796；`full_sap` s0 落在带内（ep1 0.8572 / ep2 0.8766）。最紧配对：full_sap ep1 (0.8572, AURC 0.0713) vs control seed1 (0.8576, AURC 0.0713) → ΔAUROC 0.0004，AURC 4 位小数相同。V2T 同理（control 0.7230–0.7254，full_sap 0.7201–0.7267）。多出的 1920 个参数（186113→188033）零收益 | trusted-v1 val | `WT/experiments/sap-contextual-uncertainty-v1/study.yaml:352-378`（F1） | **CAVEATED** —— 只跑了 6 个臂中的 1 个（`full_sap` seed 0，2/3 epoch），随后被 SIGKILL(rc=137)；`sap_only` 隔离臂**从未运行**。但 primary gate（E-42）独立失败，结论已足够。 |
| E-44 | **信号在特征里，不在模型里** | 未训练（随机初始化）的 sidecar 已给出 T2V contextual AUROC 0.1660 / ρ −0.4959 → 符号翻转后等效 **0.834**；训练两个 epoch 后 0.8766。学习只带来约 +0.04 AUROC | trusted-v1 val | `WT/experiments/sap-contextual-uncertainty-v1/study.yaml:379-389`（F2） | **SOLID** |
| E-45 | 权重范数诊断：trained/init per-block RMS 比值 | `score_distribution` 1.372 > `score_gaps` 1.200 > `topk_scores` 1.020 > `query_semantics` 0.955 > `top1_semantics` 0.882 > `direction` 0.836 > `context_semantics` 0.774 > **`sap_uncertainty` 0.751（最低）**；初始化本身是平的（RMS 0.0274–0.0280） | 单臂，`pytorch_model.bin.1` of run `20260821_164109_sapfeat_full_sap_s0` | `WT/experiments/sap-contextual-uncertainty-v1/study.yaml:415-438`（F5） | **CAVEATED** —— study.yaml 自己标注：**weight norm ≠ feature attribution**，单臂，无 leave-one-block-out 对照。仅作旁证。 |
| E-46 | **SAP 代码已删除。** 2026-08-22 在分支 `feature/reliable-retrieval-experiment-campaign` 上完成；**2026-08-24 经 fast-forward 合并进 master**（ADR-10），主线上已无 SAP，合并后测试 747 passed / 26 skipped | 删除范围：`query_models/module_sap.py` 整文件、`UATVR.sap` / `sap_out` 分支、`confidence_mlp` / `anchor_attn` / `anchor_ent` / `weighted_logits`、MIL-NCE / evidential NLL / negative-evidence / anchor-orthogonality 四个辅助损失及 `--w_mil --w_evidential --w_neg_reg --w_orth` CLI 开关。文本侧概率路径（`pie_net_text` / `prob_text` / text `logsigma`）与 `modules/modeling.py` 未动 | commits `3477da8`→`3ca9240`→`b9801a9`→`4020fc1`→`266a2b6`→`d3a3cd9`，分支 `feature/reliable-retrieval-experiment-campaign` | `findings.md:206-267`；本 session 在 `WT/` 校核 `git log --oneline`；`WT/docs/superpowers/plans/2026-08-21-sap-removal.md` | **SOLID（但见冲突说明）** |
| E-47 | 旧 ckpt 仍含 47 个 `sap.*` key | 参考 ckpt `ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4`：state-dict 共 **482** key，其中 `sap.*` **47** 个，**8,943,361** 参数 | — | `WT/docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md` | **SOLID** |
| E-48 | 全量训练路径的行为变化 | 不开 `use_retrieval_risk_sidecar` 时，训练目标从 `wti_logits * confidence` 变为裸 `wti_logits`。历史 run 全部开着 sidecar，**历史结果不受影响**；但未来的全量重训**不等价于 A1 那个配方** | — | `findings.md:254-257` | **SOLID** |

### ⚠️ E-46 的仓库现状冲突（重要，本 session 实测）

删除**只发生在分支** `feature/reliable-retrieval-experiment-campaign`，**master 上未删**：

| 检查 | 结果 |
|---|---|
| `git ls-tree master --name-only query_models/` | **含** `query_models/module_sap.py`（工作区实际存在，7798 字节） |
| `git ls-tree feature/reliable-retrieval-experiment-campaign --name-only query_models/` | **不含**该文件 |
| master 上的引用 | `modules/modeling_mulit.py:21` 与 `scripts/evaluate_sap_uncertainty.py:183` 仍 `from query_models.module_sap import SemanticAnchorProbing`；`modules/modeling_mulit.py:361` 仍有 `confidence_mlp`，`:1020` 仍有 `sap_out` |
| `git rev-list --count master..feature/reliable-retrieval-experiment-campaign` | **37**（未合并 commit 数） |
| `git branch -a --contains d3a3cd9` | 只有该本地分支；**不在任何 remote 上** |
| `git for-each-ref refs/heads/` 的 upstream | 两个本地分支**都没有 upstream**（master 另比 `origin/main` 领先 140 commit） |

→ **不要在 master 上假设 SAP 已删；也不要在该分支上假设它还在。删除工作无远端备份。**

---

## H. SAP Fix 1（`w_sap_failure_bce`）—— 数字为真，归因已被推翻

| ID | 事实 | 数字 | Split / Protocol | Artifact | 状态 |
|---|---|---|---|---|---|
| E-50 | Fix 1 v1 PoC（2-GPU {2,3}, w=0.1, 3 epoch, seed 0） | 被测指标 **A3 contextual failure AUROC (V2T)**：Stage 1 attempt5 winner 0.5661 → Fix1 ep2(winner) 0.7175 → ep3 0.7244。R@1 V2T 全程 93.0，R@1 T2V 全程 75.9 | trusted-v1 val，frozen retriever + WTI + DSL T=2.0 | `findings.md:138-144`；`todo.md:49-55`；ckpt `WT/ckpts/ckpt_msrvtt_20260818_220203_fix1_poc_w0p1/pytorch_model.bin.2` | **CAVEATED**（见下） |
| E-51 | Fix 1 v2 三 seed 复现（4-GPU {0,1,2,3}） | V2T A3 AUROC：seed0 0.7230 [0.6300, 0.8097]、seed1 0.7233 [0.6356, 0.8063]、seed2 0.7254 [0.6353, 0.8095]；**mean 0.7239，跨 seed sd 0.0013**（阈值 0.04）。T2V A3 AUROC mean 0.8721 (sd 0.0126)。R@1 V2T 93.0 / T2V 75.9 全同。CI 为 10000-bootstrap，直读各 run 的 `best_contextual_risk_checkpoint.json` | trusted-v1 val | `findings.md:160-165`；ckpts `WT/ckpts/ckpt_msrvtt_2026081{9,20}_*_fix1_v2_s{0,1,2}/` | **CAVEATED**（见下） |
| **E-52** | **归因已被推翻：`w_sap_failure_bce` 在 `contextual_risk_sidecar_only` 模式下是 NO-OP。** `configure_contextual_risk_sidecar_training()` 把所有参数 `requires_grad=False`，只重开 `contextual_risk_sidecar.*`，**SAP 也在冻结之列**。因此 v1/v2 两个 campaign 里 SAP **一次梯度都没收到过**，`mu_video` 从未因此改变。实证确认：三个 v2 winner 的 eval-only replay 返回**逐位相同**的 SAP direct AUROC 与逐位相同的 A1 failure AUROC (0.7294) | — | trusted-v1 val | `WT/experiments/sap-contextual-uncertainty-v1/study.yaml:240-258`（known_limits a/b）与 `:85-90`（equivalence check）；`findings.md:112-118` | **SOLID（这条推翻本身是硬结论）** |
| E-53 | 因此 E-50/E-51 实际测的是 **A3 sidecar 学 score geometry** —— 一个标准的 selective-prediction baseline，**不是** SAP/Dirichlet uncertainty 设计 | — | — | `WT/experiments/sap-contextual-uncertainty-v1/study.yaml:1-20`（study 开头的 WHY） | **SOLID** |
| **E-54** | **"符号翻转" 叙事已 RETRACTED。** 原叙事："ep0 AUROC 0.23 → ep1 0.72，证明 BCE 学到正确 failure 语义"。实际 ep0 那一行是**随机初始化的 contextual sidecar**：v2 三 seed 的 init 分别为 0.2296 / 0.6414 / 0.3461 | — | trusted-v1 val | `findings.md:181,189`；`todo.md:57` | **RETRACTED —— 不可引用** |
| E-55 | 指标名更正（2026-08-20）：原写作 "SAP u_mode AUROC"，实为 **A3 contextual sidecar 的 `failure_risk = sigmoid(failure_logit)` 的 AUROC**。全仓库没有任何地方对 `u_mode` 计算过 AUROC（直到 sap-contextual-uncertainty-v1 才补上，即 E-42） | — | — | `main_task_retrieval.py:4637`（`V2T contextual risk \| AUROC=`）；`prob_models/contextual_retrieval_risk.py:259`；`findings.md:120-122`；`todo.md:58` | **SOLID** |
| E-56 | Fix 1 campaign 缺同环境对照臂 | v2 内**没有 `w_sap_failure_bce=0` 的对照臂**；0.5661 这个参照来自另一个 campaign（contextual-risk-v2 Stage 1 attempt5） | — | `findings.md:199`；`todo.md:56` | **SOLID（缺陷本身）** —— 但因 E-52，这个对照现在已无意义（两臂在数学上等价）。 |
| E-57 | V2T 估计脆弱性 | V2T 只有 **35 failure / 500 query**，AUROC CI 宽达 ±0.09，per-seed CI lower 余量仅 3 pp；T2V 为 2348/10000，CI 宽度仅 ±0.007。**多 seed 不缩小这个采样 CI**（同一批 500 query） | trusted-v1 val | `findings.md:183`；`WT/experiments/sap-contextual-uncertainty-v1/study.yaml`（power note，stop_gates 前的注释块） | **SOLID** |
| E-58 | Fix 1 v1/v2 的数据平面**从未生成** | `experiments/sap-failure-supervision-v2/registry.jsonl` 与 `artifacts/sap-failure-supervision-v2/` 不存在（本 session 核实：`WT/artifacts/` 目录整体不存在）；证据只落在 `WT/logs/` 与 `WT/ckpts/` | — | `findings.md:185`；`todo.md:56` | **SOLID（缺陷本身）** |

---

## I. CRF compression-robust probe

| ID | 事实 | 数字 | Split / Protocol | Artifact | 状态 |
|---|---|---|---|---|---|
| E-60 | CRF ladder 15/15 pairwise ordering 全部通过 | 14 对 = 1.0000，(42,44) = **0.9960**；`three_pair_accuracy = 1.0000` | 250 videos × 6 CRF levels {clean,34,36,42,44,50} | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:199-205`；`findings.md:77-83`；`/tmp/hybrid_audit/crf_audit_report.json`（⚠️ /tmp，易失） | **SOLID** |
| E-61 | mean `u_video` 沿 CRF 单调递增 | clean 3.99 → 34: 4.28 → 36: 4.34 → 42: 4.58 → 44: 4.68 → 50: 4.95 | 同上 | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:207`；`findings.md:85` | **SOLID** |
| E-62 | per-video Δu 每一相邻档的 **5th percentile > 0** | Δ(34−clean) mean +0.29 std 0.07 5th **+0.16**；Δ(36−34) mean +0.06 std 0.02 5th **+0.02**；Δ(50−44) mean +0.27 std 0.09 5th **+0.14** | 同上 | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:209-212` | **SOLID** |
| E-63 | **跨 arch 移植成功**：`crf_head` 从 HEAD arch（R@1=14）verbatim 移植到 June arch（R@1=52.6），audit 分数不变 | three_pair 1.0000，falsified=False，R@1 52.6/52.5 保持 | JSFUSION test + CRF ladder | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:148-169,227-233`；`PAPER.md:149-156` | **SOLID** |
| E-64 | 无 CRF 训练的对照会 fail | Step5v2（fresh June, no CRF）three_pair = **0.2253**，falsified = **True** | 同上 | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:232`；`PAPER.md:154` | **SOLID** |
| E-65 | 跨数据集（MSVD） | donor ckpt 已验 `three_pair = 1.0`，`cross_domain_pass = true`，`spearman(u, −bpp) = −0.34` | MSVD | `docs/superpowers/tables/2026-08-11_r1_and_rho.md:214-216`；`findings.md:87` | **CAVEATED** —— hybrid ckpt 的 MSVD 结论是 **by transitivity**（crf_head 权重相同 + 同一 CLIP 输入），**未直接重跑**；原因是 rawvideo `.avi` decoder 有 `fps=0` bug。 |

---

## J. 相关工作中可复用的外部数字（DAB, ECCV 2026）

| ID | 事实 | 数字 | Artifact | 状态 |
|---|---|---|---|---|
| E-70 | DAB 报告的 T2V 结果（MSR-VTT 1K-A, ViT-B/32） | 56.2 / 85.4 / 92.4，MnR 4.1，RSum 234.0（vs DITS 51.9/75.7/84.6 MnR 11.6；UATVR ICCV'23 47.5/73.9/83.5 MnR 12.9） | `related_work/dab_eccv2026/NOTES.md:53-54` | **SOLID（外部报告值，未复现）** |
| E-71 | DAB 的增益形状异常：R@1 +4.3 但 R@5 +9.7 / R@10 +7.8 / MnR −64% | 同上 | `related_work/dab_eccv2026/NOTES.md:65-69` | **SOLID（观察）** |
| E-72 | DAB 消融：两个组件**单独都把 R@1 拉到 baseline 以下** | Baseline (X-Pool) 46.9 → w/ Prob only **44.3** → w/ Bridge only **43.2** → DAB 52.6 | `related_work/dab_eccv2026/NOTES.md:73-78` | **SOLID（外部报告值）** |
| E-73 | DAB §4.4 的 selective prediction：用 **top1−top2 KL margin** | R@1 56.2 → 85.6 @25% coverage → 94.0 @10%；AURC 0.429 → 0.256 | `related_work/dab_eccv2026/NOTES.md:112-114` | **SOLID（外部报告值）** |
| E-74 | DAB 自承："uncertainty signal emerges from the bridge-induced distributional gap rather than from text variance alone" | — | `related_work/dab_eccv2026/NOTES.md:115-117` | **SOLID（原文引用）** |
| E-75 | DAB 无代码发布 | abs 页与正文均无任何 URL（`grep -oE "https?://" dab.txt` → 空） | `related_work/dab_eccv2026/NOTES.md:6-7` | **SOLID** |

---

## K. OQ-1 Tier −1 增益形状普查（2026-08-28，全部为外部报告值）

> 完整表与混淆变量备注见 `research/oq1/tier-minus1-gain-shapes.md`。
> `A = (ΔR@5+ΔR@10)/2 − ΔR@1`；均为 MSRVTT-1K-A T2V，各方法相对**自己的** baseline。

| ID | 事实 | 数字 | Artifact | 状态 |
|---|---|---|---|---|
| E-76 | UATVR 概率增量（+DUA†+KL† vs TI+DSA，B/16 消融）是 **R@1-forward** | +1.2/+0.8/+0.6，MnR 12.5→12.4；A=−0.5 | UATVR arXiv 2301.06309 Table 1；`research/oq1/tier-minus1-gain-shapes.md` | **SOLID（外部报告值，已对官方 PDF 复核 ✓ E-89；原 WebFetch 转录，投稿前对 PDF 复核）** |
| E-77 | T-MASS vs X-Pool 是 R@1-forward | B/32：+3.3/+2.5/+2.9，MnR −17%，A=−0.6；B/16：+4.5/+3.4/+3.0，A=−1.3 | T-MASS arXiv 2403.17998 Table 6(a)/Table 1 | 同上（已对官方 PDF 复核 ✓ E-89） |
| E-78 | ProTA 概率增量（vs 确定性 DPA）是 R@1-forward | +1.1/+0.4/+0.2（消融表无 MnR）；A=−0.8 | ProTA arXiv 2404.12216v2 Table 1 | 同上（已对官方 PDF 复核 ✓ E-89） |
| E-79 | DiffusionRet（+Gen vs Dis）是 R@1-forward | +2.2/+1.7/−0.2，MnR −7%；A=−1.45 | DiffusionRet arXiv 2303.09867 Table 3d | 同上（已对官方 PDF 复核 ✓ E-89） |
| E-80 | PAU（含其乘性不确定性重排）是 R@1-forward | +2.9/−0.2/+1.0，MnR −3%；A=−2.5 | PAU arXiv 2309.17093 Table 6/Table 1 | 同上（已对官方 PDF 复核 ✓ E-89） |
| E-81 | DUQ（含其乘性重排 Eq.16）是 R@1-forward | T2V +4.3/+2.8/+3.9，MnR −25%，A=−0.95；V2T +6.0/+5.9/+3.5，A=−1.3 | `related_work/duq_ijcai2025/duq.pdf` Table 3（IJCAI'25, DOI 10.24963/ijcai.2025/643） | **SOLID（外部报告值，直接读自 PDF）** |
| E-82 | **DAB 表 5b：KL→W₂ 后 R@1 增益归零、全局重排效应保留** —— 论文自带的 F-A 近似实验 | 52.6→46.8（baseline 46.9），但 R@5 +6.9 / R@10 +6.2 / MnR −54%；A=+6.65 | `related_work/dab_eccv2026/dab.txt:565-567` | **SOLID（外部报告值）** |
| E-83 | **Tier −1 结论：hubness 签名（A>0 + MnR 崩塌）为 DAB 独有**，六个非 DAB 概率方法全部 A<0 | DAB 各配置 A ∈ [+2.1, +7.65]；UATVR/T-MASS/ProTA/DiffusionRet/PAU/DUQ A ∈ [−2.5, −0.25] | `research/oq1/tier-minus1-gain-shapes.md` | **SOLID（汇总观察）→ north-star thesis 家族版收窄** |
| E-89 | **Tier −1 的五篇 WebFetch 转录来源已对官方 PDF 逐数复核：5/5 完全一致，零转录错误** —— E-76…E-80 的转录风险警告解除，数字可直接投稿引用 | UATVR TI 48.4/74.2/83.3/2.0/14.1、+DSA 49.6/75.5/84.9/2.0/12.5、+KL†(UATVR) 50.8/76.3/85.5/1.0/12.4（B/16 确认，另补获 +DUA† 50.1/75.8/84.6 中间行）；T-MASS B/32 50.2/75.3/85.1/1.0/11.9 vs X-Pool 46.9/72.8/82.2/2.0/14.3，B/16 52.7/77.1/85.6/1.0/10.5 vs 48.2/73.7/82.6/2.0/12.7；ProTA DPA 47.0/75.0/84.1 → 48.1/75.4/84.3（消融表确无 MnR 列）；DiffusionRet Dis 46.8/73.5/82.9/13.0 → Both 49.0/75.2/82.7/12.1；PAU Baseline 45.6/72.9/81.5/2.0/14.5 → ours 48.5/72.7/82.5/2.0/14.0 | 官方 PDF 经 `arxiv==4.0.1` 下载至 `related_work/{uatvr_iccv2023,tmass_cvpr2024,prota_2024,diffusionret_iccv2023,pau_neurips2023}/paper.pdf`（各附 `SOURCE.txt` 记 v2/v1/v2/v2/v3 版本号），`pdftotext -layout` 转文本后比对 | **SOLID（一手核对）** |

## L. OQ-1 Tier 0 修正族能力包络（2026-08-28，冻结 fp32 val 矩阵，无 GPU）

| # | 结论 | 数字 | 出处 | 状态 |
|---|---|---|---|---|
| E-84 | 冻结矩阵的 GT 对齐规则（caption i → video ⌊i/20⌋）与 MnR 口径（0-based rank 均值+1）经数值复现验证：用仓库自身函数从 raw 矩阵复现契约 JSON 全部 7 项 post-DSL 指标 | T2V 75.85/94.78/97.26/MdR1；V2T 93.0/99.8/100.0 逐位一致（V2T 走 `independent_directions=False`，从 T2V DSL 矩阵取组内 max） | `scripts/oq1_tier0_correction_envelope.py` verify 段；`main_task_retrieval.py:4378-4384` | **SOLID** |
| E-85 | trusted-v1 val 矩阵是**低 hubness、近饱和 regime** | raw 76.50/95.50/97.89，MdR 1，MnR 2.109；S(N5)=0.355，S(N10)=0.528，maxN10=472（均匀期望 200）；对照 DAB regime：1K-A baseline R@1 46.9 / MnR 14.3 | `artifacts/oq1_tier0/tier0_results.json` | **SOLID** |
| E-86 | **DSL T=2.0 在 val 上是负增益**（训练选点用 raw val R@1，June training_args 无 DSL 键，不受影响）——与 E-08（test +2.6）对照，同一修正的增益完全由矩阵 hubness regime 决定 | val：ΔR@1 −0.65，ΔMnR +16.8%，S(N10) 0.53→0.70；温度网格 0.5–20 内 DSL/逆softmax 全部 ≤ raw | 同上 | **SOLID → "DSL=hubness 校正"解读的正面佐证** |
| E-87 | Sinkhorn 是唯一正增益的 label-free 臂，且其 T2V 作用严格等价加性 b_j=T·log v_j（族内成员）；形状 R@1-forward；**偏度与增益不单调**（T=1 增益最大但 S(N10) 0.53→1.29） | T=1.0：+1.72/+0.25/+0.09，MnR −1.4%，A=−1.55；T=2.0：+1.13，A=−1.05 | 同上 | **SOLID** |
| E-88 | **Oracle 包络：加性/仿射族在本矩阵上造不出 DAB 形状**——split-half 可迁移增益 ≤+0.5（γ=0.3 收缩后；γ=1 时 −0.08，输给 label-free Sinkhorn +0.98）；in-sample 加性 +2.30（A=−1.97）/仿射 +3.54（A=−2.75）全为 R@1-forward，偏度上升（学到标签先验而非反 hub）；corr(b_j, N_k) 全部 \|r\|≤0.12 不显著。**因 regime 错配（E-85），此结果不构成 F-A 证伪，产出为 thesis 精化：签名 = 修正 × 高 hubness 几何** | 500 dof / 5000 样本 MLE：in-sample 增益几乎全为过拟合（方法论警示：小验证集上调出的 per-candidate 校正增益应照此打折） | `research/oq1/tier0-correction-envelope.md` §3-4 | **SOLID（分析性结论）** |

---

## M. 评测数值精度与 OQ-5 分母之谜（2026-08-30，Stage 1 实现审查）

| # | 结论 | 数字 | 出处 | 状态 |
|---|---|---|---|---|
| E-89b | **bf16 autocast 评测把相似度矩阵量化到 ~8 位尾数,产生大规模精确分数并列**,`metrics.py::compute_metrics` 按并列重复计数导致指标严重失真 | arm0 seed000 val1k（1000×1000）：矩阵 100% 精确 bf16 值,每行仅 ~198 个唯一分数；238 行对角线值出现多次,共 1022 个额外并列；compute_metrics 报 t2v R@1 22.65 / MnR 115.0,同一 checkpoint fp32 复评 R@1 45.2 / MnR 21.5 | run `msrvtt-baseline-arm0-seed000-20260829T194200Z-09641b60cd15` 归档矩阵 + fp32 诊断复评；修复 commit 912a452 | **SOLID** |
| E-90 | **OQ-5 的 1000→1012 分母解谜：不存在第二个 V2T query 集**,12 个"多出的 query"是 bf16/低精度评测分数并列被重复计数;risk 模块的 num_queries=1000 是正确口径 | 双向复现:同一 checkpoint 的冻结 val 矩阵,fp32 版 **0** 个 gt 并列(每行 500/500 唯一),bf16 版 2455 个额外 gt 并列(966 行受累,每行仅 ~58 个唯一值);june 与 a1 两个 ckpt 数字完全一致 | `artifacts/rescued_from_tmp/frozen_equiv/sim_{june,a1}_{fp32,bf16}.npy`;OQ-5 硬证据(final_test.json cols len 1012, 分子 526 两侧相同) | **SOLID → D-01/D-04 结案方向:以 fp32 + 1000 分母为唯一口径** |

---

## 引用速查：什么可以写进论文

- ✅ 可直接引用：E-01~E-09、E-10a、E-20~E-24、E-31、E-32、E-40~E-42、E-44、E-46~E-48、E-52、E-55、E-60~E-64。
- ⚠️ 必须带 caveat：E-10b（bf16 口径 + V2T ρ 未过 gate）、E-10c（分母 D-01）、E-25（双次 test）、
  E-30（与 E-31 冲突）、E-43/E-45（单臂）、E-50/E-51（归因已推翻）、E-65（transitivity）。
- ❌ 绝不可引用：E-54（符号翻转）；任何声称 "Fix 1 让 `u_mode` 变得能预测失败" 的表述；
  `findings.md §2.6`（整节已被 `findings.md:112-118` 作废）。

## N. Evidence-conditioned retrieval 独立来源pilot（2026-10-10）

本节为独立新study，不复活任何旧study。科学数值均为MSRVTT trusted-v1 V_development pilot、T2V、DSL=false、seed713、64来源组件/64视频gallery/top8；不是官方test benchmark。

| ID | 事实 | 数字 | Artifact | 状态 |
|---|---|---|---|---|
| EC-01 | 用户批准按原始YouTube ID或文件SHA-256闭包隔离，保留validation500 | 排除254训练视频，剩8246；202交叉来源、31重叠时间片段对 | experiments/evidence-conditioned-retrieval-v1/reports/source-audit-2026-10-10.md | SOLID（操作定义；未知跨URL视觉重复未排除） |
| EC-02 | 五matcher全量B1最终正确数 | det14、fixed18、temperature14、standard15、gauss17 /64；gauss−det +4.6875pp，95% CI[−1.5625,+12.5]pp | 同study/reports/B1-development-comparison.json | CAVEATED（单seed小pilot、121/110步不等与监督差异；不能证明不确定性增量） |
| EC-03 | 六生产者转移全部归档后B2不足 | Gaussian Vdev corrected=4、worsened=4组件；门槛8/4 | 同study/reports/B2-transition-gate.json；B2-component-coverage.md | SOLID（本冻结pilot gate事实）；不构成方向全局证伪 |
| EC-04 | 最终软件通路与新增四卡资格 | 690相关回归通过；Q0-R5四CUDA rank13/13，578引用 | 同study/reports/gain-mse-integration-review.json；Q0-R5-final-source-qualification.json | SOLID（软件证据；不是科学有效性） |
| EC-05 | 全历史终态资源核验 | 科学12.108477896449 GPUh；软件0.469454636127 GPUh；含失败/取消 | 同study/reports/final-resource-accounting.json | SOLID（分配GPU wall hours，不等于GPU活跃时间或独占方法延迟） |

C1/C2仍未得到科学支持；B3/B4/扩展按预注册停止，V_locked未用于模型评价/选择，MSRVTT/MSVD test sealed。冻结文档的skip措辞、locked unread措辞及外层调度身份/并发额度强制范围限制见同study/reports/research-evidence-2026-10-10.md。
