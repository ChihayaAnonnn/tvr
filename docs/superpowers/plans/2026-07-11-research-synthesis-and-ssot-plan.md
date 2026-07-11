# Research Synthesis and SSOT Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 形成四篇多模态检索论文的可追溯综合分析，重写科研唯一事实源，并删除已经被取代的旧 SAP/不确定性报告与失效文档入口。

**Architecture:** 长篇论文事实、复算与公平性审计写入非 SSOT 的 analysis 文档；项目采纳结论、P0–P4、停止条件和实验顺序只写入 Roadmap。三个根级旧报告与新 Roadmap 原子替换，文档导航只引用长期有效内容；`AGENTS.md` 在后续代码计划完成删除后再写入最终工程事实。

**Tech Stack:** Markdown、Git、`rg`、`pdfinfo`、`pdftotext`；本地 PDF 与参考代码只读。

## Global Constraints

- 全部交流、长期文档和提交说明使用简体中文。
- `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` 是科研决策、证据、停止条件和实验顺序的唯一 SSOT。
- 不启动训练，不创建 checkpoint，不访问 JSFusion 1K 进行调参。
- `research_refs/` 只作本地只读证据源，不加入 Git。
- 不修改或提交当前用户改动：`.gitignore`、`docs/reference/uatvr_backbone_upgrade_strategy.md` 的删除、`research_refs/`。
- Roadmap 在代码清理完成前使用“代码删除待完成”的过渡墓碑；不得提前声称 SAP 代码已经删除。
- `AGENTS.md` 的最终清理延后到代码计划完成活动实现删除之后，避免 Plan A 提前把尚存在的代码写成“不存在”。
- Plan A 修改范围内的长期文档不复述旧 SAP 网络、实验、消融和未来设计；只保留 Roadmap 的一条墓碑记录。延期的 `AGENTS.md` 由 Plan B 最终清理。
- 不建立跨协议 SOTA 排名；论文绝对 Recall 不能冒充 trusted-v1 基线。
- 所有真实文件编辑使用 `apply_patch`，不用 shell 重定向、`cat` 或 Python 写文件。

---

### Task 1: 建立四篇论文综合分析

**Files:**

- Create: `docs/analysis/multimodal_retrieval_research_synthesis.md`
- Read only: `research_refs/papers/multimodal-retrieval/2026_cvpr_eaglenet.pdf`
- Read only: `research_refs/papers/multimodal-retrieval/2025_neurips_gare.pdf`
- Read only: `research_refs/papers/multimodal-retrieval/2025_iclr_tempme.pdf`
- Read only: `research_refs/papers/multimodal-retrieval/2026_cvpr_gravialign.pdf`
- Read only: `research_refs/papers/multimodal-retrieval/2026_cvpr_gravialign_supplemental.pdf`
- Read only: `research_refs/EagleNet/`
- Read only: `research_refs/GARE-text-video-retrieval/`
- Read only: `research_refs/TempMe/`

**Interfaces:**

- Consumes: 五份本地 PDF、三份本地参考实现、批准规格中的证据等级与研究边界。
- Produces: `docs/analysis/multimodal_retrieval_research_synthesis.md`，供 Roadmap 的“外部研究证据采纳表”链接；不直接决定实验优先级。

- [ ] **Step 1: 记录可恢复的 Plan A Git 基准并处理并发文件边界**

Run:

```bash
set -euo pipefail
for plan in \
  docs/superpowers/plans/2026-07-11-research-synthesis-and-ssot-plan.md \
  docs/superpowers/plans/2026-07-11-sap-retirement-code-cleanup-plan.md
do
  git cat-file -e "HEAD:${plan}"
  git diff --quiet -- "$plan"
  git diff --cached --quiet -- "$plan"
done
if ! git show-ref --verify --quiet refs/tags/uatvr-plan-a-base-20260711; then
  git tag uatvr-plan-a-base-20260711 HEAD
fi
git rev-parse refs/tags/uatvr-plan-a-base-20260711
test -f docs/analysis/multimodal_retrieval_research_synthesis.md
```

Expected: 两份计划已存在于 `HEAD` 且 worktree/index 均无差异，tag 输出执行起点 SHA；最后一条 FAIL、退出码 1。

若命令意外通过，立即运行：

```bash
git status --short -- docs/analysis/multimodal_retrieval_research_synthesis.md
git log -1 --oneline -- docs/analysis/multimodal_retrieval_research_synthesis.md
```

Expected: 若是未知未提交修改，停止并请求协调；若是前序已提交文件，只基于现有内容补缺，不整体覆盖。

- [ ] **Step 2: 盘点全部 PDF 页数、表格和图，不生成仓库文件**

Run:

```bash
for pdf in \
  research_refs/papers/multimodal-retrieval/2026_cvpr_eaglenet.pdf \
  research_refs/papers/multimodal-retrieval/2025_neurips_gare.pdf \
  research_refs/papers/multimodal-retrieval/2025_iclr_tempme.pdf \
  research_refs/papers/multimodal-retrieval/2026_cvpr_gravialign.pdf \
  research_refs/papers/multimodal-retrieval/2026_cvpr_gravialign_supplemental.pdf
do
  pdfinfo "$pdf" | awk -v file="$pdf" '/^Pages:/ {print file, $2}'
done
```

Expected:

```text
research_refs/papers/multimodal-retrieval/2026_cvpr_eaglenet.pdf 11
research_refs/papers/multimodal-retrieval/2025_neurips_gare.pdf 34
research_refs/papers/multimodal-retrieval/2025_iclr_tempme.pdf 22
research_refs/papers/multimodal-retrieval/2026_cvpr_gravialign.pdf 11
research_refs/papers/multimodal-retrieval/2026_cvpr_gravialign_supplemental.pdf 1
```

Run:

```bash
for pdf in \
  research_refs/papers/multimodal-retrieval/2026_cvpr_eaglenet.pdf \
  research_refs/papers/multimodal-retrieval/2025_neurips_gare.pdf \
  research_refs/papers/multimodal-retrieval/2025_iclr_tempme.pdf \
  research_refs/papers/multimodal-retrieval/2026_cvpr_gravialign.pdf
do
  echo "$pdf"
  pdftotext -layout "$pdf" - | rg -o 'Table [0-9]+' | sort -Vu | tr '\n' ' '
  pdftotext -layout "$pdf" - | rg -n 'Figure [0-9]+' | head -40
done
pdftotext -layout \
  research_refs/papers/multimodal-retrieval/2026_cvpr_gravialign_supplemental.pdf -
```

Expected: EagleNet Tables 1–5、GARE Tables 1–11、TempMe Tables 1–17、GraviAlign Tables 1–5 均被盘点，supplemental 可见一页 Gaussian overlap 推导。后续按 PDF 物理页、表题和图题定位，不依赖易漂移的 `pdftotext` 行号，不猜数值。

- [ ] **Step 3: 用 `apply_patch` 创建结构、事实类型规则和七张空表骨架**

文档必须使用以下完整一级/二级结构，不得写“同上”“待补”或空节：

```markdown
# 多模态文本—视频检索研究综合分析

> 本文记录外部论文事实、复算和公平性审计，不是 UATVR 科研决策的单一事实源。
> 当前优先级、停止条件和实验顺序只见 [科研路线图](../project/RESEARCH_ISSUES_AND_ROADMAP.md)。

## 1. 分析范围与证据规则
### 1.1 纳入论文与版本
### 1.2 证据等级
### 1.3 数值复算与协议比较规则

## 2. EagleNet
### 2.1 研究矛盾
### 2.2 核心创新与数据流
### 2.3 与常规检索表示的实质差异
### 2.4 训练目标与最终检索分数
### 2.5 实验设计与协议
### 2.6 主结果、消融、效率与鲁棒性
### 2.7 未获支持的强主张、复现缺口与理论风险
### 2.8 对 UATVR 可迁移与不可照搬部分

## 3. GARE
### 3.1 研究矛盾
### 3.2 核心创新与数据流
### 3.3 与常规检索表示的实质差异
### 3.4 训练目标与最终检索分数
### 3.5 实验设计与协议
### 3.6 主结果、消融、效率与鲁棒性
### 3.7 未获支持的强主张、复现缺口与理论风险
### 3.8 对 UATVR 可迁移与不可照搬部分

## 4. TempMe
### 4.1 研究矛盾
### 4.2 核心创新与数据流
### 4.3 与常规检索表示的实质差异
### 4.4 训练目标与最终检索分数
### 4.5 实验设计与协议
### 4.6 主结果、消融、效率与鲁棒性
### 4.7 未获支持的强主张、复现缺口与理论风险
### 4.8 对 UATVR 可迁移与不可照搬部分

## 5. GraviAlign
### 5.1 研究矛盾
### 5.2 核心创新与数据流
### 5.3 与常规检索表示的实质差异
### 5.4 训练目标与最终检索分数
### 5.5 实验设计与协议
### 5.6 主结果、消融、效率与鲁棒性
### 5.7 论文陈述、公式事实、未获支持主张与复现风险
### 5.8 对 UATVR 可迁移与不可照搬部分

## 6. 横向比较
### 6.1 创新发生层级
### 6.2 Candidate-conditioned 与独立编码边界
### 6.3 训练—推理一致性
### 6.4 协议、额外数据与后处理
### 6.5 效率与部署扩展性
### 6.6 证据强度与不可支持的强主张

## 7. 对本项目的综合判断
### 7.1 已采纳原则
### 7.2 延后机制
### 7.3 明确拒绝或不可直接迁移机制
### 7.4 需要独立规格的问题

## 附录 A：PDF 页码、表号与关键数字索引
```

正文开头明确四种陈述类型：`论文事实`、`作者主张`、`本地代码事实`、`本项目推论`。关键代码结论使用 `path:line`，关键论文数字使用 PDF 物理页与表/公式号；不得把作者主张写成已核验事实。

正文必须包含以下七张具名表格，每张表对四篇论文各至少一行，所有单元格必须填写明确内容或“论文未报告”，不得留空：

```markdown
| 论文与证据源 | 版本 | PDF 页数 | 补充材料 | 本地代码 | 复核状态 |
|---|---:|---:|---|---|---|
| EagleNet | CVPR 2026 | 11 | 主文附录 | `research_refs/EagleNet/` | 已核验 |
| GARE | NeurIPS 2025 | 34 | 同一 PDF | `research_refs/GARE-text-video-retrieval/` | 已核验 |
| TempMe | ICLR 2025 | 22 | 同一 PDF | `research_refs/TempMe/` | 已核验 |
| GraviAlign | CVPR 2026 | 11 | 单独 1 页 | 未找到 | 论文已核验、实现不可复算 |
```

其余六张表使用以下固定列：

```markdown
| 实验协议对照 | 数据集/任务 | split 与选模 | backbone/预训练 | 输入帧与尺寸 | batch/seed | test 使用 | 后处理/额外数据 | 证据定位 | 判定 |
| Headline 复算 | 作者主张 | 实际比较基准 | 基准值 | 方法值 | 绝对差 | 相对差 | 是否成立 | 证据定位 |
| 方法机制矩阵 | 表示层 | 交互层 | 时序层 | 概率层 | 最终 score | 训练 loss | candidate-conditioned | 可固定索引 | 训练—推理一致 |
| 效率与候选规模 | 论文报告指标 | 报告值 | 是否计入候选对数 | 全库扩展成本 | 延迟/吞吐 | 显存 | 参数/FLOPs | 证据强度 |
| 证据质量 | 多 seed | 方差 | 显著性 | 独立数据集 | 校准 | 失败案例 | 实现可复算性 | 综合等级 |
| UATVR 采纳矩阵 | 目标问题 | 机制 | 证据等级 | 采用/拒绝/延后 | Roadmap 阶段 | 前置证据 | 停止条件 | 综合分析锚点 |
```

每张表的首列行值分别为 EagleNet、GARE、TempMe、GraviAlign；每行使用明确的“已核验 / 弱证据 / 未验证 / 协议不兼容 / 不可复算”标签。

- [ ] **Step 4: 完成 EagleNet 小节并逐项核验**

必须覆盖 Tables 1–5、方法/训练/最终打分数据流和公开实现中的 pair-conditioned RGAT/文本采样。写清与固定 embedding/普通 WTI 的差异，给关键代码 `path:line`，并验证：Tables 1–2 不是所有指标领先；Table 3 完整模型 MSRVTT/DiDeMo T2V R@1 为 51.0/51.5；FRL 在两个数据集的边际作用不同；公开脚本使用 JSFusion test 选模，协议不兼容。

- [ ] **Step 5: 完成 GARE 小节并逐项核验**

必须覆盖 Tables 1–11、`t_i + Δ_ij` 的 pair-conditioned residual、训练目标与最终 score。给 residual/gap、batch Gaussian VIB、数据协议和代码选模的证据位置；明确全库推理是候选对级计算，项目迁移只能先考虑有候选上限的 top-k reranker。

- [ ] **Step 6: 完成 TempMe 小节并逐项核验**

必须覆盖 Tables 1–17，尤其分离 Tables 8–10 的 temporal modeling 与 token reduction，核对 Tables 1/4/17 的 GFLOPs、吞吐或显存。明确精度主要来自跨帧注意力，压缩主要负责效率且可能损失精度；不得把它表述为 uncertainty 增益。

- [ ] **Step 7: 完成 GraviAlign 主文、补充推导与证据边界**

必须覆盖主文 Tables 1–5 和单页 supplemental；只把 supplemental 证明的 Gaussian `B+C` log-overlap 标为公式事实。Term A、semantic mass、independent veto 和自然防塌缩分别标为作者主张或本项目推论；写明没有公开代码、多 seed、显著性与实测效率闭环，MSR-VTT/DiDeMo 并非所有指标领先。

- [ ] **Step 8: 完成横向比较、项目判断和附录索引**

七张表全部填满，并明确写入：

- backbone 只作匹配控制变量，不能替代方法贡献；
- 未在所有数据集 SOTA 不等于研究失败，贡献新颖性、因果闭环和可证伪实验更重要；
- EagleNet/GARE 是 candidate-conditioned scorer 或受限 top-k reranker，必须报告候选数；
- TempMe 是可独立评价的时序/效率变量；
- 新 uncertainty 必须为 query-video pair-level，并进入最终排序或风险决策；
- 新 alignment 先回答“何时、为何错配”，不继续堆无解释辅助 loss；
- 各机制必须单变量验证，四篇论文的绝对 Recall 均不得与 trusted-v1 直接归因。

附录对每篇论文至少列出主结果表、关键消融表、效率证据和公式/代码位置各一项。

- [ ] **Step 9: 运行结构、事实类型、证据等级与来源检查**

Run:

```bash
set -euo pipefail
test -f docs/analysis/multimodal_retrieval_research_synthesis.md
test "$(rg -n '^## [2-5]\. (EagleNet|GARE|TempMe|GraviAlign)$' docs/analysis/multimodal_retrieval_research_synthesis.md | wc -l)" -eq 4
test "$(rg -n '^### [2-5]\.[1-8] ' docs/analysis/multimodal_retrieval_research_synthesis.md | wc -l)" -eq 32
test "$(rg -n '^### 6\.[1-6] ' docs/analysis/multimodal_retrieval_research_synthesis.md | wc -l)" -eq 6
test "$(rg -n '^### 7\.[1-4] ' docs/analysis/multimodal_retrieval_research_synthesis.md | wc -l)" -eq 4
rg -q '^## 附录 A：PDF 页码、表号与关键数字索引$' \
  docs/analysis/multimodal_retrieval_research_synthesis.md
for label in 已核验 弱证据 未验证 协议不兼容 不可复算; do
  rg -Fq "$label" docs/analysis/multimodal_retrieval_research_synthesis.md || exit 1
done
for kind in 论文事实 作者主张 本地代码事实 本项目推论; do
  rg -Fq "$kind" docs/analysis/multimodal_retrieval_research_synthesis.md || exit 1
done
for table in 论文与证据源 实验协议对照 'Headline 复算' 方法机制矩阵 效率与候选规模 证据质量 'UATVR 采纳矩阵'; do
  header="| $table |"
  test "$(awk -v h="$header" 'index($0, h) == 1 {n++} END {print n + 0}' \
    docs/analysis/multimodal_retrieval_research_synthesis.md)" -eq 1
  block=$(awk -v h="$header" '
    index($0, h) == 1 {capture=1}
    capture && $0 == "" {exit}
    capture {print}
  ' docs/analysis/multimodal_retrieval_research_synthesis.md)
  if printf '%s\n' "$block" | rg -q '\|[[:space:]]*\|'; then
    exit 1
  fi
  for paper in EagleNet GARE TempMe GraviAlign; do
    test "$(printf '%s\n' "$block" | grep -Fc "| $paper |")" -ge 1
  done
done
for paper in EagleNet GARE TempMe GraviAlign; do
  rg -q "$paper.*(物理页|Table [0-9]|表 [0-9]|公式|research_refs/.+:[0-9]+)" \
    docs/analysis/multimodal_retrieval_research_synthesis.md || exit 1
done
```

Expected: 全部退出 0。

Run:

```bash
rg -n 'SAP|SemanticAnchorProbing|AnchorWTI|QC-SAP|wti_prob_mu|wti_anchor_wti|wti_qc_sap' \
  docs/analysis/multimodal_retrieval_research_synthesis.md
```

Expected: 无输出，`rg` 退出 1。

- [ ] **Step 10: 暂存、检查精确提交范围并提交综合分析**

Run:

```bash
set -euo pipefail
git add -- docs/analysis/multimodal_retrieval_research_synthesis.md
git diff --cached --check -- docs/analysis/multimodal_retrieval_research_synthesis.md
git diff --cached --name-status
```

Expected: 无空白错误；cached manifest 只有新增综合分析，用户已有 `.gitignore`、backbone 文档删除和 `research_refs/` 未被暂存。

Commit:

```bash
git commit -m "docs: 综合多模态检索研究" -- \
  docs/analysis/multimodal_retrieval_research_synthesis.md
```

---

### Task 2: 重写科研 SSOT 并原子删除旧报告

**Files:**

- Rewrite: `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`
- Delete: `report.md`
- Delete: `report_SAP.md`
- Delete: `report_uncertainty.md`
- Reference: `docs/analysis/multimodal_retrieval_research_synthesis.md`

**Interfaces:**

- Consumes: Task 1 的论文证据分析、trusted-v1 稳定事实、批准规格。
- Produces: 唯一科研 SSOT；代码计划完成前只声明“科研路线终止，代码删除待完成”。

- [ ] **Step 1: 运行旧 Roadmap 失败契约**

Run:

```bash
test "$(rg -n '^### P0' docs/project/RESEARCH_ISSUES_AND_ROADMAP.md | wc -l)" -eq 1
test "$(rg -n 'SAP|SemanticAnchorProbing|AnchorWTI|QC-SAP|wti_prob_mu|wti_anchor_wti|wti_qc_sap' \
  docs/project/RESEARCH_ISSUES_AND_ROADMAP.md | wc -l)" -eq 1
```

Expected: FAIL；当前 Roadmap 有两个 P0，并包含多段旧路线。

- [ ] **Step 2: 用 `apply_patch` 重写 Roadmap**

Roadmap 必须完整采用以下结构：

```markdown
# UATVR 科研问题与路线图

> 更新时间：2026-07-11。本文是科研决策、证据、停止条件和实验顺序的唯一事实源。
> 外部论文事实与复算见 [多模态检索研究综合分析](../analysis/multimodal_retrieval_research_synthesis.md)。
> 历史结构、旧报告、日志与 checkpoint 仅从 Git 历史追溯。

## 1. 当前决策摘要
## 2. P0：可信 WTI-only 基线
### 2.1 固定协议
### 2.2 稳定实现事实
### 2.3 尚未完成的结果
### 2.4 P0 完成门槛
## 3. 外部研究证据采纳表
## 4. 后续研究问题
### P1：跨模态错配与检索风险诊断
### P2：Pair-level uncertainty
### P3：Candidate-conditioned multimodal alignment
### P4：时序建模与效率支线
## 5. 成功标准与停止条件
### 5.1 不再采用的成功标准
### 5.2 机制筛选标准
### 5.3 稳定性与泛化标准
### 5.4 盲测边界
### 5.5 逐阶段停止条件
## 6. 固定实验顺序
## 7. 已关闭路线
```

在 §2 写入这张固定协议表，不改变数值：

```markdown
| 字段 | 固定值 |
|---|---|
| 数据协议 | `trusted-v1` |
| split seed | 42 |
| train / internal val | 8500 / 500 |
| test | JSFusion 1K，仅在方法、超参数和 checkpoint selection 冻结后显式盲测 |
| 主损失 | 按精确 `video_id` 构造的双向多正例 InfoNCE |
| backbone | OpenAI CLIP ViT-B/16 |
| global forward batch | 256 |
| GPU / micro / accumulation | 4 / 64 / 1 |
| LayerNorm | native FP16，保留 FP32 master affine；`CLIP_LAYER_NORM_PRECISION=fp32` 只作回退 |
| checkpoint selection | internal-val T2V R@1 |
```

在 §3 写入以下采纳表：

```markdown
| 论文机制 | 证据等级 | 决定 | 对应阶段 | 理由 | 综合分析证据 |
|---|---|---|---|---|---|
| EagleNet candidate interaction | 协议不兼容、部分已核验 | 延后 | P3 | 先证明固定 embedding 存在系统性候选错配 | [EagleNet](../analysis/multimodal_retrieval_research_synthesis.md#2-eaglenet) |
| GARE candidate-conditioned correction | 协议不兼容、部分已核验 | 延后 | P3 | 必须限制为 top-k 并报告候选规模与重排成本 | [GARE](../analysis/multimodal_retrieval_research_synthesis.md#3-gare) |
| TempMe temporal merging | 已核验边界 | 独立支线 | P4 | 不与 P2/P3 首轮实验混合 | [TempMe](../analysis/multimodal_retrieval_research_synthesis.md#4-tempme) |
| GraviAlign Gaussian overlap | 弱证据 | 仅采纳研究启发 | P2 | 需要重新有界化、推导并验证校准 | [GraviAlign](../analysis/multimodal_retrieval_research_synthesis.md#5-gravialign) |
```

§4 的四个阶段必须写入以下操作性边界：

- P1：在不改 backbone、主损失和正例矩阵的前提下，分 T2V/V2T 统计 correct/incorrect Top-1 的 score、margin、rank、token alignment；比较 high-similarity hard/fuzzy negatives 与真实多正例；区分 query ambiguity、video ambiguity 和 pair mismatch；只做诊断，不生成伪标签。
- P2：后续规格必须定义 `u(text_i, video_j)` 的预测对象、正负候选监督、进入 final logits/排序/selective retrieval 的方式、方差塌缩/爆炸与无界分数防护，并证明它不是 WTI score 或 margin 的单调复制。
- P3：只有 P1 提供系统性固定 embedding/WTI 错配证据才解锁；任何 candidate-conditioned residual/reranker 同时报告候选规模、延迟、吞吐和显存。
- P4：temporal modeling/token merging 永远作为独立支线，首轮不得与 P2/P3 同时改变。

§5 必须明确：

- 单 seed 只作机制筛选；进入主线前，基线与新方法使用同一固定 split 和同一组至少三个训练随机种子；
- backbone、split、forward contrastive batch、optimizer steps 和 checkpoint selection 必须完全匹配；
- 训练与推理使用同一个最终 score，一次直接消融只改变一个 causal variable；
- 常规检索结果报告 R@1/R@5/R@10、MdR/MnR 和 T2V/V2T；
- 声称 uncertainty/risk 时额外报告 AURC、错误检测 AUROC/相关性、分桶校准与离散度；
- 至少一个独立数据集或预定义错误切片验证方向一致性；
- 没有机制信号、只有 test 提升、收益只随换 backbone 出现，或需要同时换标签/多项 loss 时立即停止。

§6 的固定顺序为：P0 可信基线 → P1 只读诊断 → 单变量 P2 → 有证据后才进入 P3；P4 始终为独立支线。EVA 只作匹配 backbone control，不包装成方法贡献。

§7 只允许以下过渡墓碑，不写旧结构、结果或消融：

```markdown
| 已关闭路线 | 当前状态 | 追溯方式 |
|---|---|---|
| SAP 及其依赖链 | 科研路线已终止，代码删除待已批准清理计划完成 | 不再恢复；历史细节仅见 Git |
| Hard negative 主线 | 已终止 | 只保留独立诊断入口，不再 sweep 或 repeat |
| UACL 主线及活动接口 | 主线已终止；活动接口待 Plan B 删除 | 不恢复；历史细节仅见 Git |
| Semantic soft target / 伪标签 | 禁止 | 正例只由精确 `video_id` 定义 |
```

- [ ] **Step 3: 用 `apply_patch` 删除三个旧报告**

Delete exactly:

```text
report.md
report_SAP.md
report_uncertainty.md
```

不要把其中正文复制到新文件。仅保留已经在 Roadmap 中用中性语言重新表达的四项原则：pair-level、进入最终排序、训练/推理一致、校准/风险评价。

- [ ] **Step 4: 验证 SSOT 结构和旧报告删除**

Run:

```bash
set -euo pipefail
test ! -e report.md
test ! -e report_SAP.md
test ! -e report_uncertainty.md
test "$(rg -n '^## 2\. P0：可信 WTI-only 基线$' docs/project/RESEARCH_ISSUES_AND_ROADMAP.md | wc -l)" -eq 1
test "$(rg -n '^### P[1-4]：' docs/project/RESEARCH_ISSUES_AND_ROADMAP.md | wc -l)" -eq 4
test "$(rg -n 'SAP|SemanticAnchorProbing|AnchorWTI|QC-SAP|wti_prob_mu|wti_anchor_wti|wti_qc_sap' \
  docs/project/RESEARCH_ISSUES_AND_ROADMAP.md | wc -l)" -eq 1
for key in \
  trusted-v1 '8500 / 500' 'JSFusion 1K' '双向多正例 InfoNCE' \
  'global forward batch' 'native FP16' '至少三个训练随机种子' AURC
do
  rg -Fq "$key" docs/project/RESEARCH_ISSUES_AND_ROADMAP.md || exit 1
done
```

Expected: 全部退出 0；唯一 SAP 命中是过渡墓碑。

- [ ] **Step 5: 检查提交范围并提交 SSOT 替换**

Run:

```bash
set -euo pipefail
git add -- docs/project/RESEARCH_ISSUES_AND_ROADMAP.md
git add -u -- report.md report_SAP.md report_uncertainty.md
git diff --cached --check -- \
  docs/project/RESEARCH_ISSUES_AND_ROADMAP.md \
  report.md report_SAP.md report_uncertainty.md
git diff --cached --name-status
```

Commit:

```bash
git commit -m "docs: 转向 SAP 之后的科研路线" -- \
  docs/project/RESEARCH_ISSUES_AND_ROADMAP.md \
  report.md report_SAP.md report_uncertainty.md
```

Expected: 提交只包含 Roadmap 和三个报告删除；不得带入用户已有改动。

---

### Task 3: 更新文档导航并冻结 Query 历史快照

**Files:**

- Modify: `docs/README.md`
- Modify: `docs/analysis/query_branch_analysis.md`
- Defer: `AGENTS.md`，在 Plan B 删除活动代码后再一次性写入最终事实。

**Interfaces:**

- Consumes: Task 1 的综合分析路径和 Task 2 的 Roadmap 路径。
- Produces: 无歧义的长期文档导航；Query 文档只保留为历史结构快照，不提前改变工程事实入口。

- [ ] **Step 1: 运行导航失败契约**

Run:

```bash
rg -n 'reference/uatvr_backbone_upgrade_strategy|multimodal_retrieval_research_synthesis' docs/README.md
rg -n '历史快照，非当前科研路线' docs/analysis/query_branch_analysis.md
```

Expected: 第一条仍有失效 backbone 链接且没有综合分析入口；第二条无输出。

- [ ] **Step 2: 用 `apply_patch` 将 `docs/README.md` 替换为长期导航**

Use exactly this structure and wording:

```markdown
# UATVR 文档入口

## 科研决策

- [`project/RESEARCH_ISSUES_AND_ROADMAP.md`](project/RESEARCH_ISSUES_AND_ROADMAP.md)：科研问题、证据、停止条件与实验顺序的唯一 SSOT。

## 研究分析

- [`analysis/multimodal_retrieval_research_synthesis.md`](analysis/multimodal_retrieval_research_synthesis.md)：外部论文事实、复算与公平性审计，非 SSOT。
- [`analysis/query_branch_analysis.md`](analysis/query_branch_analysis.md)：2026-03-02 的 Query 分支历史结构快照，非当前科研路线。

## 工程与数据说明

- [`deploy_qwen/README.md`](deploy_qwen/README.md)：Qwen3-VL 属性生成说明。

## 归档规则

- 历史日志、旧报告、已执行计划和过期 checkpoint 只通过 Git 历史追溯。
- `research_refs/` 是本地论文、第三方代码和权重目录，不属于项目文档或提交范围。
```

- [ ] **Step 3: 用 `apply_patch` 冻结 Query 分析的历史语义**

在 `# Query 分支深度分析` 后插入：

```markdown
> **历史快照，非当前科研路线**
>
> 本文记录 2026-03-02 的旧 Query 分支结构和 legacy 协议指标。
> 其中 49.1、37.8、10.2、30.4 等结果不能与 trusted-v1 基线直接比较；
> “改进方向”不构成当前实验优先级。当前决策只见
> [科研 Roadmap](../project/RESEARCH_ISSUES_AND_ROADMAP.md)，外部方法证据见
> [多模态检索研究综合分析](multimodal_retrieval_research_synthesis.md)。
```

将标题：

```markdown
## 6. 改进方向（待评估）
```

改为：

```markdown
## 6. 历史候选方向（已失效，不构成当前计划）
```

其余结构和历史数值不改写，不把历史候选升级为新路线。

- [ ] **Step 4: 运行跨文档一致性检查**

Run:

```bash
set -euo pipefail
rg -q 'multimodal_retrieval_research_synthesis' docs/README.md
rg -q '历史快照，非当前科研路线' docs/analysis/query_branch_analysis.md
rg -q '不能与 trusted-v1 基线直接比较' docs/analysis/query_branch_analysis.md
rg -q '历史候选方向（已失效，不构成当前计划）' docs/analysis/query_branch_analysis.md
rg -q '\[科研 Roadmap\]' docs/analysis/query_branch_analysis.md
rg -q '\[多模态检索研究综合分析\]' docs/analysis/query_branch_analysis.md
```

Expected: 全部退出 0。

Run:

```bash
rg -n 'reference/uatvr_backbone_upgrade_strategy|superpowers/specs|report\.md|report_SAP\.md|report_uncertainty\.md' \
  docs/README.md docs/analysis/query_branch_analysis.md \
  docs/analysis/multimodal_retrieval_research_synthesis.md \
  docs/project/RESEARCH_ISSUES_AND_ROADMAP.md
```

Expected: 无输出，`rg` 退出 1。

- [ ] **Step 5: 暂存、检查提交范围并提交导航更新**

Run:

```bash
set -euo pipefail
git add -- docs/README.md docs/analysis/query_branch_analysis.md
git diff --cached --check -- docs/README.md docs/analysis/query_branch_analysis.md
git diff --cached --name-status
```

Commit:

```bash
git commit -m "docs: 更新当前科研文档入口" -- \
  docs/README.md docs/analysis/query_branch_analysis.md
```

Expected: 提交仅包含两个导航文档；`AGENTS.md` 保持与尚未清理的代码一致，工作树仍保留未暂存的用户既有改动。

---

## Plan A Completion Gate

Run:

```bash
set -euo pipefail
files=(
  docs/README.md
  docs/analysis/query_branch_analysis.md
  docs/analysis/multimodal_retrieval_research_synthesis.md
  docs/project/RESEARCH_ISSUES_AND_ROADMAP.md
)
test "$(rg -n 'SAP|SemanticAnchorProbing|AnchorWTI|QC-SAP|wti_prob_mu|wti_anchor_wti|wti_qc_sap' "${files[@]}" | wc -l)" -eq 1
test ! -e report.md
test ! -e report_SAP.md
test ! -e report_uncertainty.md
git diff --check -- "${files[@]}"
test -z "$(git ls-files -- research_refs/)"
base=$(git rev-parse refs/tags/uatvr-plan-a-base-20260711)
git diff --check "$base"..HEAD
git diff --name-status "$base"..HEAD
test "$(git diff --name-status "$base"..HEAD)" = $'M\tdocs/README.md\nA\tdocs/analysis/multimodal_retrieval_research_synthesis.md\nM\tdocs/analysis/query_branch_analysis.md\nM\tdocs/project/RESEARCH_ISSUES_AND_ROADMAP.md\nD\treport.md\nD\treport_SAP.md\nD\treport_uncertainty.md'
scope=("${files[@]}" report.md report_SAP.md report_uncertainty.md)
git diff --quiet -- "${scope[@]}"
git diff --cached --quiet
actual_status=$(git status --short --untracked-files=normal)
expected_status=$' M .gitignore\n D docs/reference/uatvr_backbone_upgrade_strategy.md\n?? research_refs/'
test "$actual_status" = "$expected_status"
git tag -d uatvr-plan-a-base-20260711
```

Expected:

- Plan A 修改范围内的长期文档唯一 SAP 命中是 Roadmap 的过渡墓碑；`AGENTS.md` 明确延后，不包含在此结论中。
- 三个旧报告不存在。
- `research_refs/` 没有 tracked 文件。
- `git diff --name-status "$base"..HEAD` 精确为以下 7 项，不多不少：

```text
A	docs/analysis/multimodal_retrieval_research_synthesis.md
M	docs/project/RESEARCH_ISSUES_AND_ROADMAP.md
M	docs/README.md
M	docs/analysis/query_branch_analysis.md
D	report.md
D	report_SAP.md
D	report_uncertainty.md
```

- 最终 porcelain 状态必须精确保留这三项用户改动，不得出现其他 staged/unstaged 路径：

```text
 M .gitignore
 D docs/reference/uatvr_backbone_upgrade_strategy.md
?? research_refs/
```

不要在 Plan A 中修改 `AGENTS.md` 或归档批准规格；代码清理完成后，Plan B 再写入最终 Agent 事实、把墓碑改为“已删除”并统一归档规格与计划。
