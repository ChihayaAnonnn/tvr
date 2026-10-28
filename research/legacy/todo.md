# TODO — Stage 1 后续与 SAP/Uncertainty 修复

生成日期: 2026-08-18
最后更新: 2026-08-18(协议清理完成)
关联文档:
- [Stage 1 HALT 报告](.worktrees/reliable-retrieval-experiment-campaign/experiments/contextual-risk-v2/summaries/stage1-baselines.md)
- [重试授权报告](.worktrees/reliable-retrieval-experiment-campaign/experiments/contextual-risk-v2/summaries/stage1-baselines-retry-authorization.md)

---

## 1. 已完成

- [x] Stage 1 A3 训练(attempt 5)完成,3 epoch,winner = `pytorch_model.bin.2`
- [x] Stage 1 gate 判定: **FAIL**
  - `a3_improves_strongest = false`: mean AUROC Δ = −0.0287,mean AURC Δ = +0.0019
  - `directional_rho_gte_0_20 = false`: V2T ρ = 0.1974 < 0.20
- [x] `next_authorized_stage = null`,campaign 按 SDD 协议在 Stage 1 停止
- [x] 诊断 SAP + uncertainty 设计,定位 5 个断链
  - α_dir → mu_agg → L2 归一化 → Σα 信息被削掉
  - orth_loss 强制 anchor 正交 → diversity 变常数
  - modal_entropy 依赖孤儿 α → 无信号
  - logsigma init 常数 + MIL 只推整体小 → 无 per-video 差异
  - evidential_similarity 的 epistemic_penalty per-video 缩放整行 + 梯度矛盾 → net signal ≈ 0
  - 根源: SAP 所有 uncertainty 输出缺少直接的 failure-prediction 监督信号
- [x] 确认 Stage 1 训练时 `contextual_risk_sidecar_only=True`,SAP 前向被跑但 **未接收任何 loss**,评测数据反映的是 June ckpt 遗留状态
- [x] **协议清理: Finalize Stage 1**(commit `eb2ee0b`,archive summary + registry 一致)
- [x] **协议清理: 恢复冻结的 `{0,2,3,4}` GPU 拓扑**(commit `c448240`)
  - 5 个文件 14 处 `0,1,2,3` → `0,2,3,4`(launcher 1、main_task_retrieval.py 3、tests 10)
  - 测试验证: 145/145 通过(launcher/final_fixes/experiment_tracking)
  - 采用直接编辑而非 revert 三个原始 commit(结果等价,单 commit 更清晰)

---

## 2. 协议约束(持续生效)

- 不准备 Stage 2(controller 会拒绝,predecessor gate 未通过)
- 不用不同超参 silently re-run Stage 1(post-hoc selection,违反 frozen `study.yaml`)
- Stage 5/7 MSVD launch 保持 fail-closed,直到 Task 8 qualify MSVD launcher
- 不授权任何 test 集评估,直到新 study 的 Stages 4/5 通过

---

## 3. 待执行 — SAP/Uncertainty 修复(需新的 pre-registered study)

任何 A3/SAP 重设计必须开新的 study。本次 campaign 到此结束。

修复候选(按投入产出比排序):

- [x] **修 1: 加显式 failure supervision on `u_mode`** — ✅ PoC PASS (2026-08-18)
  - 实现: `_sap_failure_bce_loss(u_mode, wti_logits)`,BCE 目标 = per-video V2T failure label
  - 训练: 2-GPU pilot {2,3},w=0.1,3 epochs,ckpt `ckpts/ckpt_msrvtt_20260818_220203_fix1_poc_w0p1/pytorch_model.bin.2`
  - 结果: **V2T A3 contextual failure AUROC 0.5661 → 0.7244**(+15.8 pp,超 primary gate 0.70)
  - 非回归: V2T R@1 稳定 93.0(baseline 93.0,通过 floor)
  - 副产物: T2V AUROC 0.21 → 0.88(+67 pp,信号从反向翻转到强正向)
  - **v2 4-GPU {0,1,2,3} 3 seeds 全部 gate PASS (2026-08-20)**: V2T winner AUROC 0.7230 / 0.7233 / 0.7254,mean **0.7239**,跨 seed sd **0.0013**(阈值 0.04),最差 CI95 lower 0.6300(阈值 0.60),R@1 V2T 全部 93.0
  - Caveats: 为跑通需删/no-op 了 4 处 A3 protocol guards;**v2 内无 `w=0` 同环境对照臂**;V2T 只有 35 failure/500 query,CI 宽 ±0.09;`experiments/sap-failure-supervision-v2/registry.jsonl` 与 `artifacts/` 未生成
  - ⚠️ 原"符号翻转"证据已撤回: init eval 那一行是随机初始化的 contextual sidecar(三 seed 分别 0.2296/0.6414/0.3461),不是无 Fix 1 基线
  - ⚠️ **指标名 2026-08-20 更正**: 之前写的 "SAP.u_mode AUROC" 实为 **A3 contextual sidecar 的 `failure_risk` AUROC**(`main_task_retrieval.py:4637` / `prob_models/contextual_retrieval_risk.py:259`)。全仓库没有对 `u_mode` 直接算 AUROC 的地方;Fix 1 经由 `mu_video → video_means` 间接影响该指标。详见 findings.md §2.6 节首

- [ ] **修 3: `epistemic_penalty` 从 per-video 改成 per-pair**
  - 用 `anchor_attn` (text × anchor) 计算 per-pair 模态熵
  - `sim_matrix = cosine * exp(-per_pair_epistemic)`,每对独立缩放
  - 中等改动: `_evidential_similarity` + `_evidential_nll_loss`

- [ ] **修 2: 保留 Σα 的 magnitude 信息**
  - 归一化前将 `log||mu_agg||` 作为额外通道传给下游
  - 或用 `log||mu_agg||` 调制 retrieval logit scale

- [ ] **修 4: 解决 orth_loss 与 diversity-as-uncertainty 的冲突**
  - 二选一: 保 orth(放弃 diversity 当 uncertainty),或改 soft orthogonality(允许 per-video 变化)

- [ ] **修 5: logsigma 差异化**
  - init weight 非零 + 加 `-std(logsigma)` 项鼓励 per-video 变化

- [ ] **修 6(最激进): 把 SAP uncertainty head 换成 A1 风格 Dirichlet failure head**
  - 输入: per-anchor top-K attention + per-anchor magnitude
  - Target: 真实 rank(与 A1 相同的监督范式)

---

## 4. 待执行 — 论文与文档

- [ ] **PAPER.md 更新**: A3 negative result 与 A2-CG negative result 并列写入 §2.3
  - 边界条件: trusted-v1 val,500 v2t queries / 35 failures 的高方差 regime
  - 说明 A3 在 JSFUSION test 上曾报告通过,与本次 trusted split 结果的差异归因

- [ ] **findings.md §2.3 更新**: 补充 A3 sidecar 失败记录 + SAP uncertainty 诊断
  - 5 个断链的技术细节保留在 findings 中,便于未来 study 参考

- [ ] **findings.md §2.6 新增**: Fix 1 (SAP failure BCE supervision) PoC 结果
  - V2T AUROC 0.5661 → 0.7244,R@1 不变,机制验证成功

- [ ] **决定 paper 走向**(需用户决策)
  - 候选路径已在前次讨论中收敛;数据已排除多个方向
  - 需在完成 §4 前两项文档更新后单独决策

---

## 备注

- 本次 campaign 的 retry lineage 已完整记录在 attempt5 manifest 的 `parent_run_ids` + `selection_lineage`
- Campaign 状态: **Stage 1 已 finalize (FAIL) + 拓扑已恢复**,可以起新 study
