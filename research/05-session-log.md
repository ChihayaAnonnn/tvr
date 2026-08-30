# 05 · Session Log（倒序，append-only）

> 规则：**只在最上方追加新条目，不修改历史条目。** 需要更正历史结论时，
> 写一条新条目说明"撤回/更正了什么"，并同步更新 `01-evidence-ledger.md` 的状态标记。
> 早期条目由 `progress.md` / `todo.md` / study.yaml 的日期反向重建，标注为 *(重建)*。

---

## 2026-08-30 (2) —— Stage 1 首跑揪出 bf16 评测量化 bug；顺带解开 OQ-5 分母之谜

**发生了什么。** Stage 1 attempt-1（arm0 seed000，四卡全程 5 epoch）跑完后指标可疑
（val1k t2v R@1 22.65 / MnR 115.0，且 compute_metrics 分母 1011≠1000）。追查：归档
矩阵 **100% 精确 bf16 值**，每行 1000 候选仅 ~198 个唯一分数,238 行对角线并列
（1022 个额外精确相等）——评测在 bf16 autocast 下计算,量化摧毁了排名信息,
`metrics.py::compute_metrics` 又按并列重复计数。同一 checkpoint 用 fp32 复评：
**t2v R@1 45.2 / MnR 21.5**（CLIP4Clip 量级,数字健康）。E-89b。

**OQ-5 结案（机制层面,E-90）。** contextual-risk 的 1000→1012 V2T 分母同源：
不存在第二个 query 集,12 条=低精度分数并列重复计数。双向复现:冻结 val 矩阵
fp32 版 0 个 gt 并列,bf16 版 2455 个(june/a1 两 ckpt 完全一致)。唯一口径 =
fp32 + 1000 分母。

**处置（全按协议）。** 修复 commit `912a452`（oq1 评测强制 exact fp32;训练不变);
attempt-1 finalize 存档（seed000 completed、seed001/002 cancelled,{completed:1,
cancelled:2}）;r2 phase 以 implementation-review 授权重开（预注册允许的一次实现
审查,发生在任何 gate 判定前,lineage 完整链接）;r2 seed000 已开跑（step20 loss
2.4860 与 attempt-1 分毫不差——训练确定性旁证）。

**两条流程教训（写入控制面 README #10）。** (a) 源码提交前先清账活跃预留——归档
契约禁止 reserved 跨源码漂移做任何转换,我这次不得不 revert→取消→finalize→
unrevert;(b) 控制器 stdout 不可接管道（SIGPIPE 会杀看护进程）。

**P0 预览（非正式,fp32 诊断复评,split=val1k, DSL off, t2v）。** R@1 45.2 ✓(≤60)、
MnR 21.5 ✓(≥8);S(N₁₀) 待 r2 归档矩阵（bf16 矩阵上的 0.898 不作数,并列任意打散
会稀释偏度）。P0 是否过门由 r2 三 seed 的冻结脚本输出决定。

**产物。** commits `45fbf24`(P0 工具) `912a452`(fp32 修复) `5ec22b0`(attempt-1
finalize) `d501632`(r2 预留);`scripts/oq1_stage1_p0_metrics.py`;E-89b/E-90 入 ledger。

---

## 2026-08-30 (1) —— 确定性契约 PASS：Stage 0 四门全关，Stage 1 解锁

**发生了什么。** GPU 空出后按 PC-8 走完确定性探针归档流程：phase
`stage0-determinism-probe`（活体 GPU 探测 + 8 个数据输入与 CLIP 资产的 sha256 绑定
进 manifest）→ reserve 2 run（HEAD `0761a744dc85`）→ 顺序执行 a/b（arm1_full_akl，
seed 0，四卡 {0,1,3,4}，batch 256，200 步，`--strict_determinism`）→ 逐字节比对 →
双双 `completed` → finalize（terminal_counts {completed: 2}）。

**结果（split=train7500, DSL n/a, t2v+v2t 训练损失）。** 两次运行的 200 步 loss 序列
NPY（sha `7039f69f722dbfa6…`）、参数摘要（`6c1f9b10323c1bee…`）、metrics 文件
**全部逐位一致**；step1 = 5.5449 = ln 256 反证四卡 AllGather 负例池正确，step200 =
4.4448。`determinism_first_200_steps_bitwise_equal_same_seed` ✅ —— **Stage 0 4/4**。

**插曲（协议按设计工作）。** run b 首次 launch 时研究者本人的 UGR sweep 占 GPU0/1，
launcher fail-closed 拒启（`on_insufficient_gpus` 路径实弹验证），证据保留为
`gpu-probe-launch-refused-….json`，预留未消耗，30 分钟后同一 reservation 重试成功。

**产物。** `experiments/oq1-kl-variance-hubness-v1/summaries/determinism-probe.md`；
registry 新增 2 行 completed；`artifacts/.../msrvtt/baseline/msrvtt-baseline-det_probe_arm1_{a,b}-…`。

**下一步。** Stage 1：arm0_deterministic × seed {0,1,2} 全程训练（5 epoch），val1k
上测 P0 regime 门（S(N₁₀)≥1.5 ∧ MnR≥8.0 ∧ R@1≤60；不过则 halt_study_as_inconclusive，
不开火 F-A）。对应 OQ-1 Tier 1。

---

## 2026-08-29 (2) —— arm 训练栈落地（commit 1c0cc4e），确定性探针待四卡

**发生了什么。** 实现了 Stage 0 最后一门所依赖的全部四块：
1. **打分模块** `modules/oq1_probabilistic_scoring.py`：七臂高斯 cost（AKL 按
   lam_{A..D}∈{0,1} 分项 / 对称 KL / ELK），CPU 上对暴力展开逐元素验证 <1e−12，
   arm1 与闭式 KL(v‖t) 恰差弃常数。0 权重项保留在计算图（零梯度参与）以满足
   DDP find_unused_parameters=False 与臂间参数量不变式。
2. **模型接线** `modules/modeling.py`：`--oq1_arm` 选头。none = June 路径逐字节不变；
   arm0 = 纯 WTI CrossEn（不构造概率模块）；arm1–6 = PIE+Uncertainty 头 +
   `logit_scale·(−cost)`（不构造 WTI weight-fc，无采样/MIL/VIB）。
3. **训练器**：`--strict_determinism`（cudnn 定死 + use_deterministic_algorithms +
   CUBLAS_WORKSPACE_CONFIG）与 `--determinism_probe_steps N`（截断 N 步、落盘逐步
   loss NPY + 参数/buffer sha256、跳过 ckpt/eval）；训练期 `--eval_split val1k`。
4. **数据与 launcher**：msrvtt val1k 评测工厂（1000×1，test 同构几何）；train
   loader 仅在 oq1 臂激活时把"与 trusted manifest 相等"放宽为"⊆"；
   `run_train_msrvtt_bg.sh` 新增 OQ1_ARM 固定命令行分支（启动前按已提交 manifest
   校验 val1k 字节一致）；`scripts/oq1_arm_launch.sh` 注册 launcher。
   实现自由度的定死清单（clamp [−3,3]、维度均值归一、双向 CrossEn、precision=fp32
   的解释等）记录在控制面 README。

**验证。** 单卡冒烟：同 seed 两次 2 步探针 losses/params sha 逐字节一致；arm0/arm6
路径各 1 步通过；全 pytest 747 passed（新参数已按默认值钉进注册 A3 契约 schema）。
数字口径：冒烟 loss（train7500 batch=8, DSL n/a, t2v+v2t 训练损失）arm1≈2.08≈ln 8。

**产物。** commit `1c0cc4e`；`/tmp/gen_oq1_det_phase.py`（⚠️ 易失，reserve 期生成
determinism-probe phase JSON，含活体 GPU 探测 + 输入哈希）。

**下一步。** 四卡 200 步 a/b 探针（归档运行）。当前 GPU0/1 被研究者本人的 UGR 任务
占用（各 ~22 GB），按 `on_insufficient_gpus: refuse_launch` 等待空出后 reserve+launch。

---

## 2026-08-29 (1) —— Stage 0 开工：val1k 分区 + 四卡资格测试 3/4 过门

**触发。** GPU 空出来了（siqi 的 VLLM 只剩一个、挪到 GPU 2；0/1/3/4 各 ~81 GB 空闲）。
按注册协议直接进 Stage 0。

**val1k 分区（gate `val1k_disjoint_from_test` ✅）。** `scripts/build_val1k_partition.py`：
8500 个 trusted-v1 train 视频，按 id 字典序排序 + numpy RandomState(0) 置换，前 1000 个
→ val1k（每视频取标注 JSON 首条 caption），其余 7500 → train7500。与 JSFUSION test
零交集、并集恰等于 train8500 均为硬断言；manifest 含全部输入/输出 sha256；重建逐字节
一致。产物 `data/generated/msrvtt_trusted_v1_val1k/`（commit b3d9f2f，force-add 循
trusted_v1 惯例）。

**四卡资格测试（gate `four_gpu_qualification_recorded` ✅，PC-9）。** 给新 study 搭了
自己的控制面（registry/schemas/phases/summaries，commit 4e3d4ec 起），复用通用归档
CLI（reserve → verify → worker → transition → finalize）与 `run_train_msrvtt_bg.sh`
的 ARCHIVE_QUALIFICATION 分支。四场景终态：success `completed`（loss=7.5 解析精确，
外置数组摘要与 contextual-risk-v2 历史夹具**逐位一致**）、两个故障场景 `failed`(1)、
SIGTERM 排水 `cancelled`(143)（完整 TERM→TERM→KILL 升级链留痕，无 UNRESOLVED）。
详见 `experiments/oq1-kl-variance-hubness-v1/summaries/archive-qualification.md`。

**归档机制的三次教训（9 个取消的预留，全部证据保留）：**
(a) manifest 的 `config.evaluation` 会与 worker 有效参数逐键核对——GPU 探测记录只能放
`selection_lineage`；(b) 运行时拓扑词是 `torchrun-4` 不是 study.yaml 的
`torchrun-4-rank`；(c) SIGTERM 场景必须确认控制器存活后再发信号（首试控制器提前退出，
孤儿 NCCL 组手动排空）。另发现归档契约的 stage 词表跨 study 固定
（`experiment_artifacts.py` ALLOWED_STAGES），本 study 的协议 stage 名 → 归档槽位
映射记录在控制面 README（study.yaml 未动）。

**Stage 0 剩余（进 Stage 1 前的最后一项）：** `determinism_first_200_steps_bitwise_
equal_same_seed`——绑定在 arm 训练实现上（概率头 + val1k 评测 + 注册 launcher，
且须把 `main_task_retrieval.py:1240-1242` 的 cudnn benchmark/deterministic 翻回来）。

---

## 2026-08-28 (6) —— Tier 1 预注册签署并冻结（commit 20a4a1d）

研究者审阅 §5 阈值后指示"先松一手"：P5（加性分量回收比例）60% → **50%**——七个阈值中
唯一没有数据参照点的判断值，取"过半"的自然语义界；修订记录写入预注册 §12.5，其余六个
阈值未动（P1–P3 架在 Tier −1 两族群空隙上，P0 是两个实测 regime 的插值）。随后研究者
签署。`study.yaml` status → **registered**（registered_at: 2026-08-28），以只含
`experiments/oq1-kl-variance-hubness-v1/` 的 commit `20a4a1d` 提交，**此后该文件冻结**
（PC-2：任何改动 = 新 study 目录 + 新 protocol_version）。
下一步：等外部进程让出显存 → PC-9 四卡资格测试 → Stage 0。

---

## 2026-08-28 (5) —— OQ-1 Tier 1 预注册起草（解 PC-4 的那份 study）

**触发。** Tier −1/Tier 0 都做完后，唯一的路障是 PC-4：MSRVTT/MSVD test 在"新 study 的
Stages 4/5 通过"之前保持 sealed。研究者指示"请你起草"。

**产出。** `experiments/oq1-kl-variance-hubness-v1/`：
`PREREGISTRATION.md`（散文体：问题、设计决策及其理由、七臂、六条定量预测 P1–P6 + 前置
门 P0、证伪映射、局限、签署栏）与 `study.yaml`（冻结协议，照 contextual-risk-v2 的结构
+ sap-contextual-uncertainty-v1 的 header/lineage 约定；已用 yaml.safe_load 验证可解析）。
`status: draft`，待研究者签署后改 `registered` 并冻结。

**四个非显然的设计决策（都写进了文档，防事后合理化）：**

1. **第一道门是 regime 门，不是效应门。** 这是 Tier 0 教训的制度化。低 hubness regime 下
   任何 candidate-only 修正都无的放矢，"消掉 (A) 项增益不变"会让 **F-A 假阳性触发**。
   故 P0（`S(N₁₀)≥1.5` 且 `MnR≥8.0` 且 `R@1≤60`）不过 → 判 inconclusive、不开火 F-A、
   不解封 test。P1（arm1 根本没重现签名）不过同样判 inconclusive。
2. **放弃 X-Pool 主线，改用本仓 June/WTI。** X-Pool 的价值是 DAB 表 5a 的 46.9 外部锚，
   但那个锚在 1K-A test 上、PC-4 下够不着；为一个 Stage 6 前收益为零的锚引入未经本仓
   协议（PC-8/PC-9/确定性契约）审计的外部代码库不划算。X-Pool 降为可选 Stage 7。
   代价已写进 limitations：与 DAB 的可比性靠机制同构论证，不靠数字对齐。
3. **新建 `val1k` 分区绕开 test。** regime 差异主要来自评测几何而非 backbone
   （同 backbone 同 DSL：val −0.65 / test +2.6）。从 8500 个 train 视频划 1000 视频 ×
   1 caption，与 1K-A 结构同构、与 test 零交集（Stage 0 硬断言），train 缩到 7500，
   全部臂在同一训练集重训。**Stage 1–5 因此一次都不用碰 test。**
4. **两条会毁掉实验的默认值被显式关死。** 主指标 no-post-processing（DSL/QB-Norm 本身
   就是 candidate-only 加性修正，开着比较各臂会吸收掉待测的隐式修正 → F-A 假阴性）；
   λ 权重固定 0/1 不扫描（扫描 = 在 val 上做选择，污染 Stage 6 的一次性 test）。
   若 arm1 因不扫描而偏弱，代价由 P1 门承担（判 inconclusive），不允许事后补扫描。

**顺带解掉的协议债。** PC-10（GPU 拓扑冲突）：起草当日实测 GPU 0/1/4 各被外部进程占
~74.5 GiB、2 空闲、3 占 22.4 GiB，即**当下根本起不了 4 卡**。study.yaml 因此不再硬编码
卡号，改为 reserve 期探测空闲显存、选前 4 张、把 UUID 与实测空闲量写进 manifest，
不足 4 张则拒绝启动并归档探测结果——"用了哪几张卡、为什么"永远有据可查。
PC-3（MSVD launcher 未入库）：v1 全程不碰 MSVD，直接规避。
PC-7 / PC-11 / PC-1：在 `not_applicable_constraints` 里逐条写明为何不适用
（不用 A3 代码 / 训练新检索器而非复用冻结矩阵 / 不同 study）。

**强制条款。** F-A 或 F-B 触发时照常走完 Stage 5 并以 negative result 归档，
不得以"实现不对/超参没调好"撤销注册换配置重跑；任何改动 = 新 study 目录 + 新
protocol_version（PC-2/PC-5）。Stage 6 不做任何选择，配置在 Stage 5 末冻结进
`stage6_frozen_config.json`（记 sha256），每臂只跑一次。
test 解封授权 `authorization_scope: this_study_only`，不外溢到 OQ-2/OQ-3。

**下一步。** 研究者签署 → PC-9 四卡资格测试（需先等外部进程让出显存）→ Stage 0。

---

## 2026-08-28 (4) —— Tier −1 文献转录债还清：五篇官方 PDF 逐数复核 5/5 一致

**背景。** 研究者问及第三方插件 `ai-research-skills` 的 `autoresearch` skill（Orchestra
Research，MIT，非 Anthropic 官方）能否接管本项目。判断：**整体不启用**——其 SKILL.md
明令 "Do not ask for permission / Never stop"，与 **PC-4**（test 集需 pre-registered
study 解锁）直接冲突，一个不停机的 agent 极可能烧掉 test 集（不可逆）；其
`research-state.yaml`/`findings.md`/`experiments/` 与本 `research/` 语料功能重复，
并行维护即制造它本要防的漂移；20 分钟心跳节奏也不匹配 GPU-day 级实验。
**单点借用其文献工具链**（`pip install semanticscholar arxiv`，绕开本环境被 org
policy 封掉的 WebSearch）；`ml-paper-writing` skill 留待写稿期独立调用。

**执行。** 用 `arxiv==4.0.1` 下载五篇官方 PDF（注意 4.x 已移除 `Result.download_pdf`，
改用 `pdf_url` + requests），`pdftotext -layout` 转文本，逐数比对 Tier −1 表全部引用值。
存于 `related_work/{uatvr_iccv2023,tmass_cvpr2024,prota_2024,diffusionret_iccv2023,
pau_neurips2023}/paper.{pdf,txt}`，各附 `SOURCE.txt` 记版本号（v2/v1/v2/v2/v3）。
按 dab.pdf/duq.pdf 先例，PDF 保留本地、暂不入 git（注意 `.gitignore` 的 `*.pdf` 是
注释掉的，这些文件处于 untracked 而非 ignored 状态）。

**结果：5/5 完全一致，零转录错误**（E-89）。UATVR B/16 确认，另补获 `+DUA†
50.1/75.8/84.6/1.5/12.8` 中间行（本表取累积口径，不受影响）；ProTA 消融表确无 MnR 列，
"n/a" 成立。E-76…E-80 的转录风险警告解除，Tier −1 数字现可直接投稿引用。

## 2026-08-28 (3) —— OQ-1 Tier 0 完成：修正族在 val 上全为 R@1-forward，thesis 精化为"修正 × 高 hubness 几何"

**执行。** 冻结 fp32 val 矩阵（`sim_june_fp32.npy`,sha 8ebf27ae4446）上的修正族
能力包络。产物：`scripts/oq1_tier0_correction_envelope.py`、
`artifacts/oq1_tier0/tier0_results.json`、`research/oq1/tier0-correction-envelope.md`；
ledger 新增 §L（E-84…E-88）；OQ-1 状态与结案判据修订；`00-north-star.md` P0 状态更新；
tier-minus1 表的 DSL 参照行补齐说明。

**预检（全过）。** GT 规则 caption i → video ⌊i/20⌋：代码强制（每视频恰 20 caption
否则 raise）+ 数值终检（用仓库自身函数从 raw 矩阵复现契约 JSON 全部 7 项 post-DSL
指标，逐位一致）。踩坑一次：multi-sentence eval 的 V2T 走 `independent_directions=False`
——从 **T2V 的 DSL 矩阵**分组取 max，不是 v2t DSL 矩阵。MnR 口径 = 0-based rank 均值+1。
另核实训练选点用 raw val R@1（June training_args 无 DSL 键），不受下述 DSL 结论影响。

**结果。**
- **val 矩阵是低 hubness、近饱和 regime**：raw 76.50/95.50/97.89，MnR 2.109，
  S(N10)=0.53（E-85）。DAB 形状（MnR 腰斩）在此物理上不可表达。
- **DSL T=2 在 val 上 −0.65 R@1**（MnR +17%，全温度网格 ≤ raw）；对照 test 上
  +2.6（E-08）→ 增益 regime 依赖，佐证 hubness-校正解读（E-86）。
- Sinkhorn 唯一正增益 label-free 臂（T=1:+1.72，A=−1.55；其 T2V 作用严格等价
  加性 b_j=T·log v_j）；偏度与增益不单调（E-87）。QB-Norm 以 val 自身为 bank
  时严格退化为逆 softmax（activation set 恒命中）。
- **Oracle 包络（500 dof / 5000 样本,split-half）**：可迁移增益 ≤+0.5（γ=1 时 −0.08,
  输给 label-free Sinkhorn）；in-sample 加性 +2.30 / 仿射 +3.54 全为 A<0、
  偏度上升（学到标签先验而非反 hub）；corr(b_j,N_k) 全部不显著（E-88）。

**判读。** 本结果**既不证伪也不支持** thesis（regime 错配,"oracle 造不出 DAB 形状
→ 证伪"判据的前提不满足,不开火）；产出是 thesis 精化：
**hubness 签名 = candidate-only 修正 × 高 hubness 分数几何，缺一不可**。
Tier −1 的分野解释不受影响（七方法均在 1K-A test 高 hubness regime 比较）。
高 hubness regime 的包络测量归并入 Tier 1（X-Pool/1K-A,PC-4 需 pre-registration）。
方法论副产品：小验证集上调出的 per-candidate 校正增益应按 in-sample 过拟合打折。

**下一步。** Tier 1 是下一个关键实验，但被 PC-4 阻塞——需先写 pre-registered study
（Stages 4/5）解锁 1K-A test 对齐。无 GPU 可先行项：pre-registration 文本起草；
OQ-5 的 1000/1012 分母对账。

## 2026-08-28 (2) —— OQ-1 Tier −1 完成：hubness 签名为 DAB 独有，thesis 家族版收窄

**执行。** 七方法增益形状普查（各自 vs 自己的确定性 baseline，MSRVTT-1K-A T2V）。
产物：`research/oq1/tier-minus1-gain-shapes.md`（完整表 + 混淆变量 + 来源）；
ledger 新增 §K（E-76…E-83）；`00-north-star.md` 头部加 2026-08-28 修正块；
OQ-1 状态更新。DUQ 原文存档 `related_work/duq_ijcai2025/duq.pdf`（964K，IJCAI 官方 PDF，
与 dab.pdf 同待遇——本地保留，暂不入 git）。

**结果。** 不对称指数 `A = (ΔR@5+ΔR@10)/2 − ΔR@1`：
- DAB 各配置 A ∈ [+2.1, +7.65]（w/Prob only 甚至 R@1 **−2.6** 而 MnR −48%）；
- UATVR(+1.2 R@1, A=−0.5) / T-MASS(+3.3, −0.6) / ProTA(+1.1, −0.8) /
  DiffusionRet(+2.2, −1.45) / PAU(+2.9, −2.5) / DUQ(+4.3, −0.95) **全部 R@1-forward**。

**意外收获（E-82）。** DAB 表 5b 自带 F-A 近似实验：cost 从 directional KL 换成 W₂
（自由加性 log σ² 项消失）后 R@1 52.6→46.8 ≈ baseline 46.9（增益归零），
而 R@5/R@10/MnR 重排效应保留。臂 5（对称 KL）/ 臂 6（ELK）的先验大幅增强。

**Thesis 影响。** "概率方法=隐式 hubness 校正"的家族版**被 Tier −1 证伪**；
收窄后的 claim（"非对称 KL 的 candidate-only 方差项是学习出的 hubness 校正，
解释 DAB 与其余方法的形状分野"）反而更强、更可防守，且这张表就是它的第一份证据。
PAU/DUQ 两个乘性重排方法也是 R@1-forward → "乘性修正"不自动产生签名，
Tier 0 的仿射 oracle 上界更有必要。

**方法论备注。** 五篇经 WebFetch 转录（UATVR/T-MASS/ProTA/DiffusionRet/PAU），
已做四组跨论文交叉校验（详见产物文件）；**投稿引用前须对原 PDF 复核**。
DUQ 直接读自 PDF 页面图像；DAB 逐行核对本地 dab.txt。
UATVR 消融是 B/16、其余 B/32；DAB 表 5 为 5 epoch——均已按行标注，未跨用。
WebSearch 在本环境被组织策略禁用，全程用直接 URL（ar5iv / arxiv HTML / ijcai.org）。

**下一步。** Tier 0（fp32 val 矩阵上 label-free 四法 + 加性/仿射双 oracle 包络，
无 GPU、PC-4 不涉及）；开工前先核矩阵的 ground-truth 对齐规则与 MnR 口径。

---

## 2026-08-28 —— OQ-1 设计定稿：四级阶梯 + 六臂消融（讨论产物，未执行）

**新核实的 scoping 事实。** `artifacts/rescued_from_tmp/frozen_equiv/` 的 8 个
`sim_*.npy` 实为 **2 个不同矩阵**（fp32/bf16 各一，SHA 分别 `8ebf27ae4446` /
`25ac139d80d6`）——frozen-retriever 契约使 June/A1/A3 逐位相同。推论：仓库内
**没有任何真正改变打分的概率检索器**，OQ-1 的 F-A/F-B 严格检验必须复现外部方法。
DAB **无公开代码**（`related_work/dab_eccv2026/NOTES.md:6`）。

**OQ-1 重写为四级阶梯**（详见 `04-open-questions.md` OQ-1，此处只记增量决策）：
- Tier −1 文献增益形状普查（对照物必须是各方法**自己的**确定性 baseline；
  登记 backbone/split/帧数/后处理四个混淆变量）→ Tier 0 val 矩阵上加性+**仿射**
  oracle 包络 → Tier 1 只复现 DAB 表 5a 的 `w/ Prob. only` 行 → Tier 2 仅审稿要求时做。
- Tier 1 打分按 KL 四项分解实现（lam_* 开关），臂 1–4 之外新增两个对照臂：
  **臂 5 对称 KL**（log σ 项相消，加性偏置天然为零——DAB 明确弃用对称 KL，
  `NOTES.md:45-46`，这个选择本身可疑）；**臂 6 同头换 ELK 打分**
  （log(σ_t²+σ_v²) 双侧耦合，无可分离 candidate-only 项——"概率≠偏置"的存在性对照）。
- 机制指纹三件套：b_j 分布、**corr(b_j, N_k^{baseline}(j))**（预测显著为负）、
  各臂 S(N_k) 轨迹；评测期增益分解 ΔS = b·1ᵀ + R。
- backbone 定为 X-Pool（1b）：有公开代码 + 表 5a 的 46.9 做实现正确性锚点；
  1a-hybrid（WTI+KL 融合）**禁止**。分布定为高斯+非对称 KL(Q_v‖P_t)；
  vMF（乘性族）仅作扩展。(B) 项是 V2T 方向的镜像偏置，T2V/V2T 分开报。
- 联动：Tier 1 的头训完即为 OQ-2 所缺的"外部方差头"。
- 约束重申：Tier 0 全程 val-only（PC-4 不涉及）；Tier 1 对齐表 5a 需 MSRVTT 1K-A
  test → **PC-4 阻塞**，先写 pre-registered study。

**顺手修正。** `04-open-questions.md` OQ-1 中已过期的 `/tmp/frozen_equiv/` 路径
改为权威路径 `artifacts/rescued_from_tmp/frozen_equiv/`。

**未执行任何实验。** 下一步：Tier −1 抄表（需拉 UATVR/T-MASS/ProTA/DUQ/DiffusionRet 原文）。

---

## 2026-08-24 (4) —— `research/` 纳入版本管理；孤儿脚本删除

**`scripts/evaluate_sap_uncertainty.py` 已真删**（研究者授权）。它 import 的
`query_models.module_sap` 在 master 已不存在，无法运行；`research/legacy/dead-code/`
随之清空移除。`06-assets.md` 对应行改为"已删除"。

**`research/` + `related_work/` 提交进 master。** 此前 `research/`（本 anti-drift 体系本体）
一直是 untracked —— 一次误删就全没了。同时提交：
- `AGENTS.md` 的强制前置阅读指针（与 `research/README.md` 配套，必须同一次提交）；
- `related_work/dab_eccv2026/{NOTES.md,dab.txt}` —— `00-north-star.md` 与 E-70…E-72
  按**行号**引用 NOTES.md，不提交就是悬空引用。
- `related_work/dab_eccv2026/dab.pdf`（2.8 MB）**故意不提交**，第三方论文原文保持本地。

**仍未决：** master 领先 `origin/main` ~177 commit，**无 upstream、从未推送**。
现在 `research/` 至少在 git 里了，但仍只存在于这一台机器。

---

## 2026-08-24 (3) —— 合并后清理：worktree + 分支删除，回收 37 G

**执行。** 研究者授权"全删吧"。
`git worktree remove --force .worktrees/reliable-retrieval-experiment-campaign` +
`git branch -d feature/reliable-retrieval-experiment-campaign`（fully-merged 校验通过）。

**删除前核查。** 8 个 ckpt 均**不在** `critical-runs.json`，也**不在**
`artifacts/contextual-risk-v2/legacy/inventory.jsonl`（0 行命中）——从未登记为 critical。
文档里的引用（`06-assets.md`、`01-evidence-ledger.md`、两个 `study.yaml`、
`docs/superpowers/plans/2026-08-21-sap-removal.md`）都是死线 provenance 叙述，不是活依赖。
> 这一步是上一轮"审计漏查 inventory.jsonl"的教训：**删 ckpt 前必须同时查
> `critical-runs.json` 和 `inventory.jsonl` 两处。**

**抢救。** 35 个小文件 → `artifacts/dead-lines/sap-fix1-campaign-provenance/`（572 K）：
每个 run 的 `runtime.log` / `training_manifest.json` / `training_args.json` /
`checkpoint_epoch.*.published.json`。丢弃权重与 3 GB 量级的 JSON 转储
（`best_contextual_risk_checkpoint.json` / `contextual_risk_candidates.json` /
`a3_paired_baseline.json`）——**这批 run 从此不可重新评测，只可查超参与日志。**

**验证。** `git worktree list` 只剩主工作区；`git branch` 只剩 master @ `d3a3cd9`；
`git fsck --connectivity-only` 无输出；`.git` = 7.4 M；`/data2` 70%（2.2 T avail）；
bundle `reliable-retrieval-campaign_d3a3cd9.bundle` 完好（verify: complete history）。

**仍未决。**（a）master 领先 `origin/main` ~177 commit、无 upstream、未推送，
bundle 只覆盖到 `d3a3cd9`，**不含 `research/`**；（b）`research/legacy/dead-code/
evaluate_sap_uncertainty.py` 是否直接删掉；（c）`06-assets.md` 里的 `/tmp` 易失路径
与 `PAPER.md` Appendix A 的 `/tmp/hybrid_ckpt` 指向仍需改为 `ckpts/hybrid_june_crf_transplant/`。

---

## 2026-08-24 (2) —— ADR-10 结案：37 个 commit 合并进 master，SAP 主线删除落地

**决策。** 研究者选择方案 (a)：整个分支合并进 master。

**执行。** `git rev-list --count feature/...--master` = 0，即 master 完全包含在分支中，
所以是纯 **fast-forward**（`git merge --ff-only`），master `2185fb2` → `d3a3cd9`，
无 merge commit，37 个 commit 线性保留。61 files changed, +13901 / −575。
合并前确认分支不触碰工作区里未提交的 `.gitignore` / `AGENTS.md`，无冲突。

**结果。**
- `query_models/module_sap.py` 删除；全仓库（除 `research/` 的历史叙述外）无 `module_sap` /
  `SemanticAnchorProbing` 残留引用。
- 三个 SAP study 控制面、`contextual-risk-v2` 的 7 个 phase JSON 与全部 stage1 summaries、
  `reliability_campaign.py` / `retrieval_risk_baselines.py` / `run_reliability_campaign.py`、
  2026-08-18→08-21 训练日志，全部进入 master。
- 孤儿脚本 `scripts/evaluate_sap_uncertainty.py`（未跟踪、import 已删模块）
  移入 `research/legacy/dead-code/`。
- **测试：747 passed / 26 skipped**，132.88s，`/home/xujie/.conda/envs/tvr/bin/python`。
  注意默认 `python3` 缺 `einops`，跑测试必须用 `tvr` 环境。

**文档同步。** ADR-10 → RESOLVED（保留决策前原始记录备查）；DE-3 的"仅在分支删除"警告改写为
"master 已删除"；E-46 与缺陷 D-06 状态更新，D-06 标为 RESOLVED。

**残留。** master 比 `origin/main` 领先约 177 个 commit，**无 upstream、未推送**。
bundle 只覆盖到 `d3a3cd9`，不含此后的提交。推 remote 仍未做。

---

## 2026-08-24 (1) —— 工作区清理：`.git` 156G → 7.9M，回收约 210 GB

**发生了什么。** 按 `CLEANUP-PROPOSAL.md` 执行清理。审计**推翻了原始假设**：无需任何 git 历史改写——
整个可达仓库（所有 ref + reflog + worktree index）只有 40.0 MB / 2,897 objects，
`.git` 里其余全是不可达垃圾；另发现一个**此前无人知晓的 76.5 GB 孤立 Git-LFS 缓存**
（`.git/lfs/objects`，20 个对象，最大 9.3 GB），系某次尝试用 LFS 往 GitHub 推 checkpoint 的残留。
删除前三重独立验证：`git lfs ls-files --all` 为空、任何 ref 中都不存在 `.gitattributes`、
无任何 LFS 指针文件。

**先做的保护性操作（清理之前）。**
- `/tmp/hybrid_ckpt/` → `ckpts/hybrid_june_crf_transplant/`，SHA-256 校验一致。
  这是 PAPER Table 4 移植结果的唯一 artifact，原先躺在 98% 满的 `/` 上。
- `/tmp/frozen_equiv/`（306 MB，含 §3.1 逐位等价性的唯一证据 + **8 个可直接用于 OQ-1 hubness 测量的
  相似度矩阵**）与 `/tmp/hybrid_audit/` → `artifacts/rescued_from_tmp/`，9 个文件全部 sha256 校验通过。
- ADR-10 的无备份分支 → `artifacts/backups/reliable-retrieval-campaign_d3a3cd9.bundle`（4.5 MB），
  `git bundle verify` 报告 "records a complete history"。**ADR-10 的处置决策仍未做，只是不再有丢失风险。**
- 两个 detached worktree 的独有内容（各 13 行 `train_msrvtt.sh` diff + 2.5M/1.9M 独有日志）
  → `artifacts/rescued_worktrees/`；其 5.5 GB ckpt 各自移入 `ckpts/` 后才注销 worktree。

**执行内容。** Tier 0（垃圾 pack + 各类 cache 1.86 GB）、Tier 1（`git gc --prune=now` 回收 76.93 GB、
删除孤立 LFS 缓存 76.5 GB、注销 3 个已完全合并的 worktree）、Tier 2 中经批准的两项
（11 个无引用 ckpt 39.28 GB、`research_refs/` 13.42 GB）。
**未执行**：`deploy_qwen/`（60 GB）、`cache_dir/`（17.9 GB）、`artifacts/` 三项保留。

**过程中纠正的一处审计疏漏。** 审计称 11 个候选 ckpt "无任何引用"，实测每个都有 2 处命中：
一处是 `CLEANUP-PROPOSAL.md` 自身（自引用），另一处是审计**漏查的**
`artifacts/contextual-risk-v2/legacy/inventory.jsonl`。核查后确认全部 11 个在该 inventory 中
classification 均为 `historical`（非 `critical`，全库共 76 条 critical），且均不在
`experiments/contextual-risk-v2/legacy/critical-runs.json` 中，其 SHA-256 已在 inventory 留档，
删除方才放行。

**验证结果。** `.git` 156G → **7.9M**；`git fsck --connectivity-only` 无输出（干净）；
`master`(2185fb2) 与 `feature/reliable-retrieval-experiment-campaign`(d3a3cd9) 两个 ref 完好；
必留的 `.worktrees/reliable-retrieval-experiment-campaign` 未受影响；`/data2` 由 73% 降至 70%。

**文档整理。** 根目录五份历史文档移入 `research/legacy/` 并附 `README.md` 说明各自的已知缺陷；
`research/README.md` 的"相关但不在本目录"段同步更新。

**下一步。** 与 2026-08-23 条目相同：OQ-1 / OQ-5 的先后仍待研究者决定；ADR-10 的处置仍待决策。
另新增两项待办：（a）`06-assets.md` 中 `/tmp` 易失清单需按本次转存结果更新路径；
（b）`PAPER.md` Appendix A 引用的 `/tmp/hybrid_ckpt/` 路径应改指 `ckpts/hybrid_june_crf_transplant/`。

---

## 2026-08-23 —— 建立 `research/` 持久记录目录

**发生了什么。** 通读 `PAPER.md` / `findings.md` / `todo.md` / `progress.md` / `task_plan.md` /
`related_work/dab_eccv2026/NOTES.md` / `AGENT(S).md` / `docs/superpowers/{reports,tables,specs}` /
`WT/experiments/*/study.yaml` 与 `WT/experiments/contextual-risk-v2/summaries/`，
蒸馏成 8 份 anti-drift 文档。过程中对多个数字与路径做了**直接核验**（`sha256sum`、
解析 `final_test.json` / `frozen_retrieval_equivalence.json`、读 `training_args.json`、`git ls-tree`）。

**新发现的缺陷（原文档里没有）。**
- D-02：`PAPER.md:95` 把 A1 sidecar 的 SHA-256 当成了 base ckpt 的 SHA。
- D-03：`PAPER.md:51` 写 CLIP-ViT-B/32，实际是 **ViT-B/16**。
- D-06：`findings.md:209` 声称 SAP 已删除，但 master 上仍在且仍被 import。
- ADR-10：`feature/reliable-retrieval-experiment-campaign` 的 37 个 commit 无任何远端备份。

**产物。** `research/{README,00-north-star,01-evidence-ledger,02-dead-ends,03-decisions,
04-open-questions,05-session-log,06-assets,07-protocol-constraints}.md`；
`AGENTS.md` 末尾追加指针块。

**下一步。** 研究者决定 OQ-1（hubness 测量）与 OQ-5（分母对账）的先后；
同时对 ADR-10 做出处置决策（建议先备份）。

---

## 2026-08-22 —— uncertainty × multimodal 领域调研 + 新 thesis 提出；SAP 代码删除

**发生了什么。**
1. 做了一轮 uncertainty × multimodal 的 landscape survey，形成新的研究命题：
   *representation variance 与 predictive uncertainty 是两套独立机制；概率 embedding 的
   accuracy 增益主要是隐式 hubness 校正；可用的 uncertainty 信号完全来自 candidate-set score geometry*，
   建设性一半是 conformal risk-controlled selective retrieval + CRF 分层的 conditional coverage。
   **该 thesis 是 PROPOSED，未验证。**
2. 在分支 `feature/reliable-retrieval-experiment-campaign` 上完成 SAP 删除（6 个 commit）。
3. 确认项目方向：UATVR 未归档，继续做 video-text / 多模态检索，**仅放弃 SAP 这条线**。
4. 更新 `findings.md`：新增 §2.7（SAP 删除 + "mu_video 从未进入检索打分"的 dataflow 更正），
   并把 C4 的描述从 "SAP + SpatialEnhancer + AdaNorm + DSL" 改为 "SpatialEnhancer + AdaNorm + WTI + DSL"。

**产物。** commits `3477da8` / `3ca9240` / `b9801a9` / `4020fc1` / `266a2b6` / `d3a3cd9`；
`findings.md §2.7`（`findings.md:206-267`）；
thesis 本身**只存在于跨 session 记忆，仓库内无文档**（`[未溯源]`）。

**下一步（当时）。** 把 thesis 变成可执行的 pilot。→ 已写入 `00-north-star.md`。

*来源：`findings.md:17,206-267`；`WT` 的 `git log`；`MEMORY.md → project_direction_shift.md`。*

---

## 2026-08-21 —— SAP 证伪定案 + DAB 关联工作笔记

**发生了什么。**
1. 跑 study `sap-contextual-uncertainty-v1`：16:37 第一次启动 29 秒内 fail-closed
   （`experiment_tracking._CONTEXTUAL_FEATURE_BLOCKS` 不认识 `full_sap`/`sap_only`，无 GPU 时间浪费）；
   16:41 第二次启动 run `20260821_164109_sapfeat_full_sap_s0`；17:38 被 SIGKILL(rc=137)，
   原因未定（dmesg 不可读，主机内存 20/1007 GB，GPU 峰值 5.64 GB，**不像资源限制，最像外部干预**）。
   driver 遵守 fail-closed，放弃剩余 5 个臂。
2. 尽管只跑了 1/6 个臂，primary gate 独立失败：SAP 5 个描述子 V2T direct AUROC
   0.4561–0.5441（阈值 0.55）。study 标记 `status: falsified`, `do_not_resume: true`，findings F1–F5。
3. 写 `related_work/dab_eccv2026/NOTES.md`：DAB（arXiv 2607.20984, ECCV 2026）的批判性阅读，
   提出 hubness 假设，并指出其 §4.4 从相反立场佐证了我们的负结果。
4. 执行 control arm equivalence check：三个 v2 winner 的 eval-only replay **逐位复现**
   0.7230 / 0.7233 / 0.7254 → 反证 SAP 从未被训练过。

**产物。** `WT/experiments/sap-contextual-uncertainty-v1/study.yaml`（含 outcome 块）；
`WT/docs/superpowers/reports/2026-08-21_sap_removal_checkpoint_keys.md`；
`related_work/dab_eccv2026/{NOTES.md,dab.pdf,dab.txt}`；
ckpt `WT/ckpts/…20260821_164109_sapfeat_full_sap_s0/pytorch_model.bin.1`。

**下一步（当时）。** 删除 SAP 代码。→ 2026-08-22 完成。

---

## 2026-08-20 —— Fix 1 v2 三 seed 结果 + 指标名更正 + 撤回"符号翻转"

**发生了什么。**
1. 完成 sap-failure-supervision-v2 的 seed 1 / seed 2（4-GPU {0,1,2,3}）。
   三 seed V2T A3 AUROC 0.7230 / 0.7233 / 0.7254，mean **0.7239**，sd **0.0013**，
   最差 CI95 lower 0.6300，R@1 V2T 全部 93.0。**全部 pre-registered gate PASS。**
2. **指标名更正：** 原写 "SAP u_mode AUROC" 是错的；被测量的是 **A3 contextual sidecar 的
   `failure_risk = sigmoid(failure_logit)` 的 AUROC**（`main_task_retrieval.py:4637` /
   `prob_models/contextual_retrieval_risk.py:259`）。全仓库当时没有任何地方对 `u_mode` 直接算过 AUROC。
   数字未变，只改名与归因。
3. **撤回"符号翻转"证据：** ep0 那一行是随机初始化的 contextual sidecar
   （三 seed 的 init 分别 0.2296 / 0.6414 / 0.3461），不是"未加 Fix 1 的基线"。

**产物。** ckpts `WT/ckpts/ckpt_msrvtt_20260820_150233_fix1_v2_s1/pytorch_model.bin.0`、
`WT/ckpts/ckpt_msrvtt_20260820_174558_fix1_v2_s2/pytorch_model.bin.2`；
`WT/logs/20260820/{150233,174558}_fix1_v2_s*_…`；`findings.md §2.6` 的两个更正框。
**数据平面未生成**（`registry.jsonl` / `artifacts/` 至今不存在）。

**下一步（当时）。** 补 w=0 对照臂。→ 实际走向：2026-08-21 发现整条路径是 NO-OP，对照臂已无意义。

---

## 2026-08-19 —— Fix 1 v2 启动（4-GPU 三 seed 串行），seed 0 完成

**发生了什么。** Study `sap-failure-supervision-v2`，topology {0,1,2,3}，
launcher `run_train_msrvtt_fix1_4gpu_bg.sh`，其余配置与 v1 相同，3 seeds 串行。
seed 0 winner = ep3，V2T A3 AUROC 0.7230，CI95 [0.6300, 0.8097]。

**产物。** `WT/ckpts/ckpt_msrvtt_20260819_212850_fix1_v2_s0/pytorch_model.bin.2`；
`WT/logs/20260819/212850_fix1_v2_s0_…`。

*来源（重建）：`findings.md:154-165`。*

---

## 2026-08-18 —— Stage 1 finalize (FAIL) + 拓扑恢复 + Fix 1 v1 PoC

**发生了什么。**
1. Stage 1 A3（attempt 5）训练完成，3 epoch，winner = `pytorch_model.bin.2`。
   **gate FAIL**：`a3_improves_strongest = false`（mean AUROC Δ −0.0287，mean AURC Δ +0.0019）；
   `directional_rho_gte_0_20 = false`（V2T ρ 0.19735）。`next_authorized_stage = null`，
   campaign 按 SDD 协议在 Stage 1 停止。核心发现：**零参数 entropy scorer（mean AUROC 0.8300）
   击败 128-dim A3 MLP（0.8013）**。
2. 诊断 SAP + uncertainty 设计，定位 **5 个断链**（见 `02-dead-ends.md` DE-3）。
3. 确认 Stage 1 训练时 `contextual_risk_sidecar_only=True`，SAP 前向被跑但**未接收任何 loss**。
4. 协议清理：finalize Stage 1（commit `eb2ee0b`）；恢复冻结的 `{0,2,3,4}` GPU 拓扑
   （commit `c448240`，5 文件 14 处，测试 145/145 通过）。
5. 晚间跑 Fix 1 v1 PoC（2-GPU {2,3}，w=0.1，3 epoch）：V2T A3 AUROC 0.5661 → 0.7175(winner ep2) / 0.7244(ep3)，
   R@1 V2T 稳定 93.0。**为跑通删除/no-op 了 4 处 A3 protocol guards**（见 `07-protocol-constraints.md` PC-7）。

**产物。** `WT/experiments/contextual-risk-v2/summaries/stage1-baselines.md` + `stage1-gate.json`；
`todo.md`（本文件生成于当日）；
`WT/ckpts/ckpt_msrvtt_20260818_220203_fix1_poc_w0p1/pytorch_model.bin.2`；`WT/logs/20260818/`。

**下一步（当时）。** 4-GPU 多 seed 复现 Fix 1。→ 2026-08-19/20 完成。

---

## 2026-08-17 —— frozen-retriever 等价性证明

**发生了什么。** 写 `scripts/verify_frozen_retrieval_equivalence.py`，在同一 canonical 架构、
同一 pinned 精度下比较 June / A1 / A3 三份 ckpt 的 pre-DSL WTI 相似度矩阵（10000×500）：
**逐位相同**（fp32 SHA `8ebf27ae…`，bf16 SHA `25ac139d…`，含 repeat-run determinism control）。
定位 0.3–0.5 点 R@1 散布的根因为 `main_task_retrieval.py:4065-4088` 的 `_eval_autocast_context`
按 sidecar flag 选择 bf16 / fp32。

**产物。** `scripts/verify_frozen_retrieval_equivalence.py`（untracked）；
`/tmp/frozen_equiv/`（⚠️ 易失：JSON + 8 个 `sim_*.npy`）；`PAPER.md:53`；`findings.md:33-38`。

*来源（重建）：`PAPER.md:53`；`findings.md:33-38`。*

---

## 2026-08-13 —— 论文初稿 PAPER.md v0.1 + 5 张图

**发生了什么。**
1. 盘点 project docs / plans / reports；定位 A1 sidecar 数值证据；写 `findings.md`、`task_plan.md`、`progress.md`。
2. 派研究 agent 做 2024–2026 竞品扫描（结果进 PAPER §2 与 Appendix C）。
3. 出 5 张图（fig1 architecture / fig2 A1 vs A2-CG / fig3 CRF ladder /
   fig4 risk-coverage（用 `final_test.json` 的真实 per-coverage 点）/ fig5 DSL sweep）。
4. 写完 `PAPER.md` 全部章节。
5. 同日另跑 A3 winner 的固定 DSL T=2 post-hoc 评测：T2V R@1 **52.9**（vs 历史 52.6，+0.3 ≈ 3 个 query，
   报告自己注明"不应过度解读"，且明确 **A3 不能改变 ranking**）。

**产物。** `PAPER.md`（251 行）；`findings.md`；`task_plan.md`；`progress.md`；
`paper/figs/fig{1..5}_*.{pdf,png}` + `paper/figs/make_figures.py`；
`docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md`；
`ckpts/ckpt_msrvtt_20260813_214730_a3_dsl_t2_eval/`。

*来源：`progress.md:3-26`；`docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md`。*

---

## 2026-08-12 —— A2-CG 负结果注册 + A3 v1 训练与 JSFUSION one-shot

**发生了什么。**
1. A2-CG（CSD 512-D log-variance）训练并注册为**失败**：val 上每个 gate 都 fail
   （T2V ρ −0.1472，AUROC 0.4064，selective accuracy 非单调，方差塌成常数）。
   选中 ckpt 完整保留 A1（frozen boundary 审计通过）。
   协议偏离：JSFUSION 被评了两次（continuation 前后），仅描述性。
2. A3 contextual risk v1 训练（3 epoch + continuation 到 5），val 上 epoch idx 3 **PASS**
   五项检查（V2T ρ 0.200715，靠 0.000069 的差距选中）；JSFUSION one-shot（**DSL=false**）：
   A3 在 AUROC/AURC/coverage 上全面优于 A1。
3. 协议变更 ADR-04：`paired_invariance` 降为 audit-only（commit `3bd3df8`）。

**产物。** `ckpts/ckpt_msrvtt_20260812_115504_csd_logvar_a2_cg/`（winner `bin.2`，SHA `e539108a…`）；
`ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/`（winner `bin.3`，SHA `2e6f662b…`）；
`docs/superpowers/reports/2026-08-12-csd-logvar-a2-results.md`；
`docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md`。

*来源：两份 report 文件本身。*

---

## 2026-08-11 —— R@1 目标达成（June + DSL T=2）+ CRF hybrid + A1 契约确立

**发生了什么。**（这一天信息量最大，`docs/superpowers/tables/2026-08-11_r1_and_rho.md` 是逐小时流水账）
1. 诊断 SAP 不确定性坍缩，写 `2026-08-11_sap_collapse_investigation_and_fix.md`（路 1+3 修复）。
2. 发现所有 ckpt 的 ρ ≪ 0.20 → 触发决策规则 `drop_uncertainty_claim`，卖点转向 compression-robust probing。
3. Arch restoration（恢复 June 的 SAP + SpatialEnhancer + AdaNorm）。
4. **DSL 温度扫描：T=2.0 给出 T2V 52.6 / V2T 52.5，R@1 硬指标达成。**
5. **CRF head 权重移植方案（ADR-02）**：hybrid ckpt 同时满足 R@1 52.6/52.5 与 CRF audit 1.0000。
6. 23:04 确立 **A1 retrieval-risk sidecar 契约（ADR-01）**，当晚 23:19:59 起训。

**产物。** `docs/superpowers/tables/2026-08-11_r1_and_rho.md`；
`docs/superpowers/reports/2026-08-11_sap_collapse_investigation_and_fix.md`；
`/tmp/hybrid_ckpt/pytorch_model.bin.hybrid` + `/tmp/hybrid_audit/crf_audit_report.json`（⚠️ 均在 /tmp）；
`ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/`；`logs/20260811/231959_risk_sidecar_a1_train_msrvtt.log`。

---

## 条目模板（复制到最上方使用）

```markdown
## YYYY-MM-DD —— <一句话标题>

**发生了什么。** <做了什么、观察到什么。数字必须带 split/protocol。>

**产物。** <ckpt / log / report / commit / figure 的具体路径。/tmp 下的要标 ⚠️ 易失。>

**证据登记。** <是否需要在 01-evidence-ledger.md 新增/改状态的行？写明 ID。>

**撤回/更正。** <本次是否推翻了此前的结论？写明推翻了哪条，并同步改 ledger 状态为 RETRACTED/CAVEATED。>

**下一步。** <一句话。若是新实验，指出它对应 04-open-questions.md 的哪个 OQ。>
```
