# 07 · Protocol Constraints —— 不可违反的常设规则

> 这些不是建议。违反其中任何一条，产出的数字**不可用于论文**，且会被 controller / preflight 拒绝。
> 每条都标了出处。若你（新 session）想做的事撞上其中一条，**先回来找研究者要授权**，不要绕开。

---

## PC-1 · 不准备 Stage 2

`experiments/contextual-risk-v2` 的 Stage 1 gate **FAIL**，`next_authorized_stage = null`。
controller 会直接拒绝 Stage 2 的 reserve（predecessor gate 未通过）。

*出处：`todo.md:36`；`WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md:80-96`。*

---

## PC-2 · 不得用不同超参 silently re-run Stage 1

`study.yaml` 是 frozen 的。换超参重跑 = post-hoc selection，直接毁掉整个 campaign 的统计有效性。
Stage 1 已经 finalize（commit `eb2ee0b`），它的 FAIL 是**已注册的终态结论**，不是待改进的中间态。

*出处：`todo.md:37`。旧终态不可覆盖由生命周期/schema 保证；2026-10-01 用户删除第8条，取消逐次人工重试授权。*

---

## PC-3 · MSVD launch 保持 fail-closed

在 Task 8 qualify MSVD launcher 之前，Stage 5/7 的 MSVD 启动路径必须继续 fail-closed。
`run_train_msvd_june_bg.sh` 与 `run_train_msvd_risk_sidecar_bg.sh` 目前都是 **untracked**，
不满足 AGENTS.md 规则 4（"仅从干净源码启动，绑定精确 source SHA"）。

**这条直接阻塞 OQ-4（MSVD/DiDeMo transfer）。** 另有已知 `.avi` decoder `fps=0` bug 待修。

*出处：`todo.md:39`。*

---

## PC-4 · 不授权任何 test 集评估

在新 study 的 **Stages 4/5 通过之前**，MSRVTT / MSVD test 集保持 sealed。
所有选择、消融、阈值、K、校准**只准用 validation**（trusted-v1）。

**状态更新（2026-08-28）**：那个"新 study"现已存在并签署——
`experiments/oq1-kl-variance-hubness-v1/`（status: registered，commit `20a4a1d`）。
其 stage4_multiseed + stage5_mechanism 双门通过是 **MSRVTT** test 解封的唯一授权路径，
且授权 `this_study_only`；**MSVD 不在其 v1 范围内，继续 sealed**。
在双门通过前，本条对一切评估照常生效。

**这条直接阻塞 OQ-3（conformal coverage under CRF shift）** —— 它必须先写成自己的 pre-registered study（oq1 study 的授权不外溢）。

**状态更新（2026-08-30）**：`oq1-kl-variance-hubness-v1` 于 **Stage 1 P0 regime 门
不通过而 HALT (inconclusive)**（arm0 val1k 三 seed S(N₁₀)=1.144/1.011/0.867 均 <1.5；
MnR、R@1 两条件过；`summaries/stage1-arm0-baseline.md`）。其 stage4/5 永不达成 →
**该解封路径已死,MSRVTT/MSVD test 全部继续 sealed**,直到某个新的 pre-registered
study（新目录 + 新 protocol_version）重新建立授权路径。

*出处：`todo.md:40`；AGENTS.md 规则 6。*

---

## PC-5 · 任何 A3 / SAP 的重新设计都需要一个**新的** pre-registered study

不能挂在 `contextual-risk-v2` 下面继续做。`sap-contextual-uncertainty-v1` 已标
`status: falsified` + `do_not_resume: true`，**不得 resume**。

*出处：`todo.md:45`；`WT/experiments/sap-contextual-uncertainty-v1/study.yaml:455`。*

---

## PC-6 · 不得为兼容旧 checkpoint 放宽严格 `load_state_dict`

旧 ckpt 里带 **47 个 `sap.*` 权重键**（总 482 个 state-dict 键，8,943,361 参数）。
两条加载路径：

- `--init_model` → `modules/until_module.py:100 init_preweight`：**宽容**，可以吃下多余的键。**用这条。**
- `main_task_retrieval.py:6326` / `:6357` 的 `load_state_dict`：**严格**。

`prob_models/uncertainty_protocol.py` 的 `validate_selected_checkpoint` 已经在结构上封死了
"pre-removal ckpt 走到严格路径"的可能。**不要为了省事把严格路径改成 `strict=False`** ——
那会让 silent 的权重不匹配变成可能，破坏 frozen-retriever 契约的可验证性。

*出处：`WT/docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md`；`findings.md:259-262`。*

---

## PC-7 · 未偿协议债：Fix 1 v2 期间被删除/no-op 的 4 处 A3 protocol guard

为了让 Fix 1 跑通，以下守卫被删掉或改成 no-op，**至今未恢复**：

1. `_assert_registered_a3_initial_environment`
2. `_assert_registered_a3_contract`
3. epoch allowlist
4. sidecar_only clean-tree 检查

同时有 **10 个相关测试被 `@pytest.mark.skip`**。

⚠️ **测试基线数字有冲突，两说并存：**
- `findings.md:198` / `todo.md:56`：**750 passed / 26 skipped / 0 failed**。
- `WT/docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md`：正确的 pre-removal 基线是
  **823 passed / 26 skipped**，并指出 plans 里引用的 "745 passed" 是**过期数字**。

**行动要求：** 在把 A3 相关代码用于任何**注册**实验之前，必须先恢复这 4 处守卫并取消 skip；
否则该实验不满足 AGENTS.md 规则 4。

*出处：`todo.md:56`；`findings.md:198`。*

---

## PC-8 · AGENTS.md 的 Scheme A 归档规则 1–7（全文强制）

要点复述（**以 `AGENTS.md` 原文为准**）：

1. 用 `reserve-phase` 预留唯一 run，先提交整批 `reserved` registry 记录。
2. 新输出只写 `artifacts/<study>/<dataset>/<stage>/<run-id>/`；**不得直接写 `logs/` 或 `ckpts/`**。
3. worker 运行期间不得修改受跟踪文件；实时状态只进 run-local `status.json` + append-only audit。
4. 仅从**干净源码**启动，绑定精确 source SHA / 命令 / 配置 / 输入哈希 / manifest SHA-256；
   未提交或不完全一致的预留必须拒绝。
5. >1 MiB 的数值数组外置为 NPY/NPZ 引用并校验哈希；JSON 里不得嵌张量 / 相似度矩阵 / 逐查询数组 / bootstrap 样本。
6. 选择、消融、阈值、K、校准**只用 validation**；test 在 gate 授权前保持 sealed。（= PC-4）
7. 同 phase 所有 worker 终态后，追加 terminal registry 记录 + 受跟踪摘要，再提交该批。

2026-10-01 用户删除原第8条，不再要求逐次人工批准重试。恢复按 campaign 策略核验；现有生命周期/schema、历史产物和旧研究 gate 不变。

> ⚠️ **已知违规现状：** Fix 1 v1/v2 与 SAP study 的产物写在 `WT/ckpts/` 与 `WT/logs/` 下，
> `WT/artifacts/` 与 `registry.jsonl` 均不存在 —— 即**违反规则 2 与规则 7**。
> 因此这些数字目前**只能作为探索性证据**，不能进论文的注册结果表。

*出处：`AGENTS.md:3-13`。*

---

## PC-9 · Stage 0 四卡资格测试是所有 GPU 实验的前置条件

> "Stage 0 的四卡资格测试通过并记录前，不得启动任何 GPU 科学实验。"

*出处：`AGENTS.md:16-17`。*

2026-10-11 显式用户修订仅适用于 `full-data-retrieval-v1`：正式容量双卡 Q0
替代该 study 四卡 gate，保持全局 batch64/128；允许共享两卡并保留显存余量。
其他 study 不变；test 仍封存。见对应 study 的 protocol_amendment/resource_policy。

---

## PC-10 · GPU 拓扑：冻结值是 `{0,2,3,4}`，但现实与之冲突

- ADR-06（commit `c448240`，2026-08-18）把拓扑**恢复**为冻结的 `{0,2,3,4}`，改了 5 个文件 14 处，测试 145/145 通过。
- 但 `sap-failure-supervision-v2` 与 `sap-contextual-uncertainty-v1` 两个 study 都声明
  `topology: [0,1,2,3]`，理由是 **GPU 4 被外部 VLLM 服务占用**。

**两说并存，未裁决。** 启动任何 GPU 实验前**先确认 GPU 4 的实际占用状态**，
并在 study.yaml 里显式记录你用的拓扑与理由 —— 不要默认沿用上一个 study 的值。

*出处：`todo.md:29`（commit `c448240`）；两份 study.yaml 的 `topology` 字段。*

---

## PC-11 · frozen-retriever 契约（数值契约，不只是流程）

任何 sidecar 实验中：**只有 sidecar 参数可训练**，检索打分路径必须是原始 WTI
（`weighted_token_wise_intersection(text_token, visual_output, attention_mask, video_mask) * logit_scale`）。
该契约已由 `scripts/verify_frozen_retrieval_equivalence.py` **逐位验证**（E-01…E-06，max_abs_diff = 0.0）。

推论：**任何 sidecar 都不可能改变 ranking / R@1。** 如果你看到 sidecar 让 R@1 变了，
那是精度路径（bf16 vs fp32，`main_task_retrieval.py:4065-4088`）造成的，不是真实增益 —— 不要当成结果报告。

*出处：ADR-01；`PAPER.md:53`；`findings.md:33-38`。*

---

## PC-12 · 数字纪律

- 报任何数字必须同时给 **split（trusted-v1 val / JSFUSION test）+ DSL 设置（T 值或 false）+ 方向（T2V/V2T）**。
  A1 的 rank/failure 标签依赖于**有向的 DSL 矩阵**，换设置数字就会变（ledger 里三组 A1 test 数字互不相同就是这么来的）。
- 数字的**唯一真源是 `final_test.json` 等 artifact**，不是 `PAPER.md`（后者已知有 D-02/D-03 缺陷）。
- 发现两处来源冲突时，**两说并列记录 + 标冲突**，不要挑一个信。
- V2T 方向的 CI 半宽 ≈ ±0.09（35 failure / 500 query），**加 seed 不会缩小**。
  不要用 3 个 seed 的 sd = 0.0013 去论证 "结果稳健"—— 那测的是训练稳定性，不是统计功效。

---

## PC-13 · 分支 `feature/reliable-retrieval-experiment-campaign` 无远端备份

37 个 commit 未合并进 master，两个本地分支均无 upstream，`d3a3cd9` 不在任何 remote 上。
**在做出处置决策（ADR-10）之前，不要 rebase / reset / 删除该分支或其 worktree。**

2026-10-12 scoped user amendment: “用1k test进行测试” authorizes only
`full-data-retrieval-v1` final MSRVTT JSFUSION1000x1000 evaluation of its six
already validation-selected baseline checkpoints. `FINAL_TEST_PROTOCOL.md` and
study.yaml final_test freeze scope/inputs/checkpoints; fresh matching-source
production two-rank Q0 remains required. Baseline/qualification entries remain
non-test. Other studies, MSVD, tuning on test and new-method gates are unchanged.

2026-10-21 scoped user amendment: evaluate the two completed full-data UATVR
seed713 factor treatments on JSFUSION1k using their frozen validation-selected
checkpoints, compare against the archived original seed713 baseline. See
`full-data-retrieval-v1/FACTOR_TEST_PROTOCOL.md` and `study.yaml final_test_factors`.
Failed/incomplete treatments and all other studies remain outside this scope.
