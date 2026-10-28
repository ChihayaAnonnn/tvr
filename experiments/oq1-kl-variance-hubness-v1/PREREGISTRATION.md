# 预注册 · oq1-kl-variance-hubness-v1

> **状态：REGISTERED（研究者已于 2026-08-28 签署）。`study.yaml` 已冻结，protocol_version = 1。**
> 起草日期：2026-08-28。起草人：Claude。起草依据：`research/04-open-questions.md` OQ-1 Tier 1、
> `research/00-north-star.md` F-A/F-B、`research/oq1/tier-minus1-gain-shapes.md`、
> `research/oq1/tier0-correction-envelope.md`、`research/07-protocol-constraints.md` PC-1…PC-13。
> 格式依据：`experiments/contextual-risk-v2/study.yaml`（结构）与
> `experiments/sap-contextual-uncertainty-v1/study.yaml`（header + lineage 约定）。
>
> **本预注册的协议作用**：PC-4 规定"在新 study 的 Stages 4/5 通过之前，MSRVTT / MSVD test
> 保持 sealed"。本文件即那个 new study。Stage 4 与 Stage 5 的 stop-gate 全部通过，
> 是本 campaign 内 MSRVTT test 解封的**唯一**授权路径；该授权**只对本 study 生效**，
> 不外溢到 OQ-2 / OQ-3 / 任何其他 study（它们各自需要自己的预注册）。

---

## 1. 科学问题（一句话）

**在非对称 KL 打分中，只依赖 candidate 的方差项（−½·log σ_v²）是不是一个"学出来的
hubness 校正"？它的有无，是不是 DAB 与其余概率化检索方法增益形状分野的原因？**

这是 north-star thesis 诊断半部分在 Tier −1 收窄之后剩下的、唯一还站着的可证伪 claim。

## 2. 已有证据与本研究的位置

| 层级 | 已完成 | 结论 |
|---|---|---|
| Tier −1 | 7 方法增益形状普查（E-76…E-83, E-89，五篇已对官方 PDF 逐数复核） | 签名为 DAB 独有；六个非 DAB 方法 A<0。"家族普适"版本已证伪 |
| Tier 0 | 冻结 val 相似度矩阵上的修正族能力包络（E-84…E-88） | val 是**低 hubness 近饱和** regime（R@1 76.50, MnR 2.109, S(N₁₀)=0.53）；DSL 在 val 上 **−0.65** 而在 test 上 **+2.6**；加性/仿射 oracle 可迁移增益 ≤ +0.5 |
| **Tier 1（本预注册）** | — | 在**可控 regime 内**做臂级消融，直接测 F-A / F-B |

Tier 0 的关键教训被写进了本设计：**签名 = candidate-only 修正 × 高 hubness 分数几何，
缺一不可**。在低 hubness 的分数几何里，任何 candidate-only 修正都无的放矢，
消掉它当然"增益不变"——那会让 F-A **假阳性触发**。所以本研究的第一道门是 regime 门，
不是效应门。

## 3. 关键设计决策（以及为什么）

### 3.1 为什么不直接用 X-Pool

Tier 1 原计划以 X-Pool 为 backbone，好处是 DAB 表 5a 的 46.9/72.8/82.2/14.3 可以当外部锚。
但该锚点**在 1K-A test 上**——PC-4 下拿不到。为一个够不着的锚引入一份未经本仓协议
（PC-8 归档规则、PC-9 硬件资格、确定性契约）审计的外部代码库，成本高而收益在 Stage 6 之前为零。

**决定：主线 backbone = 本仓自己的 June/WTI 确定性检索器**（我们能插桩、能复现、已在协议内）。
X-Pool 复现降级为 **Stage 7 可选外部效度臂，明确不在 v1 预算内**。
代价（写进 limitations）：arm 1 与 DAB 表 5a 的可比性靠机制同构论证，不靠数字对齐；
Stage 6 只在 test 上产出与 DAB 同口径的形状表，不产出"我们复现了 X-Pool"这一 claim。

### 3.2 为什么要新建 `val1k` 分区（本设计的核心一步）

Tier 0 的 val 是 **500 视频 × 20 caption**，接近饱和；1K-A test 是 **1000 视频 × 1 caption**。
regime 差异主要来自**评测几何**（候选池大小、每视频 caption 数），不是 backbone。
证据：同一 backbone 同一 DSL，val −0.65 / test +2.6。

因此从现有 **8500 个 train 视频**中划出 **1000 个视频 × 每视频 1 条 caption**，
构造与 1K-A **结构同构**的 `msrvtt_trusted_v1_val1k`；train 缩到 7500。
这样 Stage 1–5 全部可以在一个 hubness-y 的 regime 里做完，**test 一次都不用碰**。

约束：新分区必须与 JSFUSION test 的 1000 个 video_id **零交集**（Stage 0 断言，硬失败）；
caption 选取规则确定性冻结（每视频取标注 JSON 中 caption 列表的最小 index）；
分区 manifest + sha256 入 registry。所有臂（含 arm0）在同一 7500 训练集上从头训练。

### 3.3 后处理必须关掉

DSL / QB-Norm 本身就是 candidate-only 加性修正。若在开着 DSL 的情况下比较各臂，
显式修正会**吸收掉**我们想测的那个隐式修正，F-A 会假阴性。
**主指标一律 no-post-processing**（DSL off、QB-Norm off）。DSL-on 只在 Stage 2 作为
对照列出现。所有数字按 PC-12 三元组标注：(split, DSL 设置, 方向)。

### 3.4 方向

**T2V 为主检验方向**（Tier −1 全表是 T2V）。V2T 全程记录但**不进 stop-gate**
（PC-12：V2T CI 半宽 ≈ ±0.09，加 seed 也压不下去）。

## 4. 臂（arms）

打分为高斯非对称 KL cost（query t → candidate v），按项拆开加权：

```
cost(t, v) = λ_A · (−½ log σ_v²)          (A) candidate-only        ← 被测对象
           + λ_B · (+½ log σ_t²)          (B) query-only
           + λ_C · (σ_v² / 2σ_t²)         (C) bilinear
           + λ_D · (‖μ_t − μ_v‖² / 2σ_t²) (D) Mahalanobis
```

| arm | 配置 | 角色 |
|---|---|---|
| `arm0_deterministic` | 无概率头，原 WTI 打分 | 各臂共同 baseline |
| `arm1_full_akl` | λ_A=λ_B=λ_C=λ_D=1 | 应当**重现**签名 |
| `arm2_no_A` | **λ_A=0**，其余=1 | **F-A 主臂** |
| `arm3_D_only` | 只留 (D) | 概率头退化为确定性距离的对照 |
| `arm4_unit_sigma_t` | σ_t ≡ 1（冻结 query 方差），A/C/D 活跃 | 分离"query 方差"与"candidate 方差" |
| `arm5_symmetric_kl` | 对称 KL（log σ 项相消） | 论文内置对照的干净版（对应 DAB 表 5b 的 W₂ 行，E-82） |
| `arm6_elk` | Expected Likelihood Kernel | 无可分离 candidate-only 项 |

除打分/损失外，**所有臂共享同一份训练配置、同一份数据、同一组 seed**；
概率头参数量在 arm1/2/4/5/6 之间保持一致（arm2 只是把 (A) 项的权重设 0，不删参数），
以免"参数量变了"成为替代解释。

## 5. 预注册的定量预测

指标定义（冻结）：
- `A = (ΔR@5 + ΔR@10)/2 − ΔR@1`，Δ 相对 **同 seed 的 arm0**。
- `ΔMnR% = (MnR_arm − MnR_arm0) / MnR_arm0`。
- `N_k(j)` = candidate j 在全部 query 的 top-k 中出现次数；`S(N_k)` = Fisher 矩偏度，k∈{5,10}。
- `b̂_j` = 把 arm 的分数矩阵回归到 arm0 分数矩阵上的 per-candidate 加性残差
  （`S_arm ≈ α·S_arm0 + 1·b̂ᵀ + R`，最小二乘，冻结实现）。

| # | 预测 | 判定阈（val1k, no-DSL, T2V） |
|---|---|---|
| **P0（regime 前置门）** | val1k 上 arm0 处于高 hubness regime | `S(N₁₀) ≥ 1.5` **且** `MnR ≥ 8.0` **且** `R@1 ≤ 60.0` |
| **P1（arm1 重现签名）** | 全 KL 头产生 hubness 形状 | `A(arm1) ≥ +2.0` **且** `ΔMnR% ≤ −25%` |
| **P2（F-A 主检验）** | 去掉 (A) 项签名消失 | `A(arm2) ≤ +0.5` **且** `ΔMnR%(arm2) ≥ −10%`；**且** `A(arm1) − A(arm2) ≥ 2.0`，配对 bootstrap 95% CI 下界 > 0 |
| **P3（对照臂）** | 对称 KL / ELK 无签名 | `A(arm5) ≤ +0.5` **且** `A(arm6) ≤ +0.5` |
| **P4（机制指纹）** | 隐式偏置打的就是 hub | `Spearman ρ(b̂_j, N₁₀^arm0) ≤ −0.30`，95% CI 上界 < 0 |
| **P5（可分解性）** | 增益的主体是那个秩-1 加性分量 | 把 `b̂` 单独加到 arm0 分数上，回收 arm1 的 `(ΔR@5+ΔR@10)` 与 MnR 改善各 **≥ 50%** |
| **P6（F-B 主检验）** | 显式修正后概率增量的形状塌掉 | arm1 相对「arm0 + Stage 2 最优 label-free 修正」的**增量**满足 `A ≤ +0.5` |

## 6. 证伪映射（这是预注册的核心承诺）

| 观测 | 结论 | 后果 |
|---|---|---|
| **P0 不满足** | regime 不可用 | **研究判为 inconclusive**。不触发 F-A，不解封 test，不改 thesis。记录 regime 结果后停。（Tier 0 的教训制度化） |
| P0 ✓ 但 **P1 不满足** | arm1 没重现签名 | **inconclusive**：我们的 KL 头不是 DAB 的那个现象的载体。不触发 F-A。允许一次、且仅一次的实现审查后重跑（新 run-id，lineage 记明），若仍不满足则停 |
| P0 ✓ P1 ✓ **P2 不满足** | 去掉 candidate-only 项后增益基本不变 | **F-A 触发 → thesis 诊断半部分被证伪。** 必须改写 `00-north-star.md`，本 study 以 negative result 结项 |
| P0 ✓ P1 ✓ P2 ✓ **P6 不满足** | 显式 hubness 修正之上概率增量仍显著 | **F-B 触发 → 同上被证伪** |
| P0…P3 ✓ 而 **P4/P5 不满足** | 效应真实但机制不是 per-candidate 加性偏置 | thesis 需**重大修正**（不是证伪）：效应存在，"加性偏置"这个数学刻画错了。转 Tier 2 找正确刻画 |
| **P0…P6 全部 ✓** | 诊断半部分成立 | Stage 6 解封 test 出确认表；constructive half（OQ-3 conformal）获得启动依据（但仍需自己的预注册） |

**预先声明的反向承诺**：F-A 或 F-B 触发时，本 study 照常走完 Stage 5 并**照常归档**，
以 negative result 形式写进 ledger 与 north-star；不得以"实现不对/超参没调好"为由
撤销注册、换配置重跑。任何配置更改 = 新 study 目录 + 新 protocol_version（PC-2, PC-5）。

## 7. 阶段与门

| Stage | 内容 | 门 |
|---|---|---|
| `stage0_qualification` | 4-GPU 资格测试（PC-9）；val1k 分区构造 + 与 test 零交集断言；确定性契约（同 seed 两次训练前 200 step loss 逐位一致）；代码 clean + SHA 绑定 | 全部硬失败即停 |
| `stage1_baseline_regime` | arm0 训练 + val1k 评测 + hubness 几何 | **P0** |
| `stage2_correction_envelope` | 在 arm0 的 val1k 相似度矩阵上重跑 Tier 0 全套（`scripts/oq1_tier0_correction_envelope.py`，DSL / inverted-softmax / QB-Norm / Sinkhorn / 加性 oracle / 仿射 oracle） | 无门（产出 P6 所需的"最优 label-free 修正"）；这一步本身就是 F-B 的 label-free 半 |
| `stage3_arm_ablation` | 六臂 × seed 0 | 无门（探索性，全部记录） |
| `stage4_multiseed` | arm {0,1,2,5,6} × seed {0,1,2}，10000 次配对 bootstrap | **P1, P2, P3** ← PC-4 授权门之一 |
| `stage5_mechanism` | b̂ 抽取、ρ(b̂, N₁₀)、ΔS 分解、F-B 增量检验 | **P4, P5, P6** ← PC-4 授权门之二 |
| `stage6_test_confirmation` | **仅在 4 与 5 全通过后**：MSRVTT 1K-A test，一次性、全臂冻结 | 见 §8 |
| `stage7_external_replication` | X-Pool 复现（可选，**不在 v1 预算内**） | — |

## 8. test 解封条款（PC-4 的落地）

- `test_sealed.default: true`，`datasets: [msrvtt, msvd]`。
- `authorize_only_after: stage4_and_stage5_gates_passed`。
- 解封后 test 评测**每臂只跑一次**。`terminal_in_place_retry: forbidden`；
  失败的 run 保留证据（PC-8 第 7 条），后继只能是新 unique run-id 且 lineage 指明前驱。
- Stage 6 **不做任何选择**：臂、超参、K、阈值、DSL 设置在 Stage 5 结束时全部冻结并写入
  `stage6_frozen_config.json`（含 sha256）；Stage 6 只是执行。
- 本次授权**不覆盖 MSVD**：v1 MSVD 全程不动（同时规避 PC-3——MSVD launcher 仍未入库）。

## 9. 硬件与 PC-10 的解决

PC-10（GPU 拓扑冲突）在本 study 中的处理方式：**不硬编码卡号**。
起草当日实测（2026-08-28）：

| GPU | 已用 / 总 |
|---|---|
| 0 | 74711 / 81920 MiB（外部占用） |
| 1 | 74843 / 81920 MiB（外部占用） |
| 2 | 14 / 81920 MiB（空闲） |
| 3 | 22447 / 81920 MiB（部分占用） |
| 4 | 74531 / 81920 MiB（外部占用） |

即**当下无法起 4 卡**。因此 `hardware_policy` 改为 fail-closed 的运行时探测：
reserve 阶段探测每卡空闲显存，选出 ≥ `min_free_mib` 的前 4 张，**把实际选中的
UUID 与空闲显存写进 manifest**；不足 4 张则拒绝启动并把探测结果作为证据归档。
这样"用了哪几张卡、为什么"永远有据可查，不再依赖一个会过期的常量。

## 10. 归档（PC-8）

沿用 Scheme A：reserve 阶段生成 unique run-id；只写
`artifacts/oq1-kl-variance-hubness-v1/<dataset>/<stage>/<run-id>/`；
运行期不改任何 tracked 文件；clean source + 精确 SHA 绑定；
> 1 MiB 数值数组外置为 NPY 并记 hash（相似度矩阵、N_k 向量、b̂ 向量均适用）；
registry 分 phase 批量写；失败证据保留，`failed`/`cancelled` 不可变。

## 11. 明确不适用的既有约束（避免后续误判）

- **PC-7**（A3 protocol guards / skipped tests 债）：本 study **不使用 A3 sidecar 代码**，
  不适用。若 v2 引入 A3，须先还清 PC-7。
- **PC-11**（frozen-retriever 契约）：本 study **训练新的检索器**，不复用冻结矩阵，
  故不适用；相应地，本 study 的 R@1 变化是**真实检索变化**，不是精度伪影。
  Stage 2 是唯一复用"冻结矩阵之上做后处理"范式的地方，其冻结对象是 arm0 的 val1k 矩阵。
- **PC-1**（contextual-risk-v2 Stage 2 未授权）：不同 study，互不影响。

## 12. 已知局限（现在就写下来，防止事后合理化）

1. **单 backbone**。主线只有 June/WTI 一个 backbone；"DAB 也如此"靠机制同构论证。
   X-Pool 复现留 Stage 7。
2. **val1k 不是 1K-A**。结构同构（1000×1）但视频来自 train pool，难度分布可能不同；
   Stage 6 的 test 表是唯一的同分布确认。
3. **概率头是我们的实现，不是 DAB 的**。无 bridge 模块。claim 限定在"非对称 KL cost 的
   candidate-only 方差项"，不覆盖 DAB 的 bridge 贡献。
4. **b̂ 的最小二乘抽取是一个模型**。P4/P5 检验的是"加性刻画能解释多少"，
   不是"机制一定是加性的"；残差 R 的范数一并报告。
5. **λ 权重不调**。全部固定为 0/1，不做 λ 扫描——扫描等于在 val 上选择，会污染
   Stage 6 的一次性 test。若 arm1 因此偏弱，代价由 P1 门承担（判 inconclusive），
   不允许事后加扫描。

## 12.5 起草期修订记录（签署前，合规）

- **2026-08-28**：P5 阈值 60% → **50%**（研究者指示"先松一手"）。理由：P5 是七个阈值中
  先验依据最薄的一个——"增益主体来自加性分量"里的"主体"没有数据参照点可插值，60% 是
  纯判断值；50% 取"过半"这一有自然语义的最低界。P0/P1/P2/P3/P4/P6 未动（P1–P3 架在
  Tier −1 实测的两族群空隙上，P0 是两个实测 regime 的插值，动它们会失去数据依据）。
  签署后本节封笔，任何再修订 = 新 study 目录 + 新 protocol_version。

## 13. 签署

| 角色 | 姓名 | 日期 | 签署即表示 |
|---|---|---|---|
| 研究者 | 研究者本人（会话内签署，git user: Qihaya Anon） | 2026-08-28 | 接受 §5 阈值（含 §12.5 的 P5=50% 修订）、§6 证伪映射（含 negative result 照常归档）、§8 test 一次性条款 |
| 执行 | Claude | 2026-08-28 | 起草 |

已签署：`study.yaml` 的 `status` 已改为 `registered`（registered_at: 2026-08-28），
以只含本目录的 commit 提交，此后该文件冻结。
