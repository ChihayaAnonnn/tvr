# Semantic Pilot C1: Official Labels and Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现新研究的路径/授权守卫、train-only 来源审计、官方关系导入及显式 benchmark 数据合同；主流程无需逐视频人工语义标注，C1 验收仅用合成数据。

**Architecture:** 新 pilot policy 在管理器预留与 worker 启动统一检查。CPU 工具处理来源 allowlist、官方 ID/内容 ID 映射、组件划分和 benchmark manifest 编译；Episode v2 显式区分标签空间，v1 语义审核约束保持严格。来源报告用于确认官方数据身份，不代替语义真值也不伪造 reviewed。

**Tech Stack:** Python、现有 PyYAML/jsonschema/pytest、标准库 hashlib/csv/json、Scheme A；不新增依赖或人工标注平台。

**Spec:** [修订后的实验设计](../specs/2026-10-01-semantic-retrieval-experiment-design.md)，重点为第 3、4、9、10 节。

状态：2026-10-01 按用户指令同步设计修订，待实施；全部任务未执行。本计划不生成真实 run、读取真实视频/caption 或启动 GPU。

## Global Constraints

- MSRVTT/MSVD test sealed；只有身份可证明的 train-only snapshot 可导入。未知来源、混合 test 注释或不安全路径先拒绝，不读取内容后再补来源声明。
- 连通组件覆盖查询与视频来源；按设计的 namespace 哈希顺序依次分配 locked≥1,000、cal≥500、dev≥500 官方视频 ID，余量 train。完整组件不拆分；held-out 未入库成员不回流。
- 库目标 locked=1,000、cal=500、dev=500，train≥6,500 官方 ID；每库≥129 canonical（支持128项 absent），locked≥800 来源组件。每视频一条稳定 caption，数量门不足停止。
- `benchmark_annotation` 非配对关系是 benchmark 负例，不是已审核语义负例。多 caption、alias 均不制造跨视频多正例；本版正例数不是一时停止并报协议不匹配。
- 缺失/异常标签不能转 negative；自然查询的标注 GT 必须在所属库内，候选移除 GT 只改变候选存在事件。
- 无 576 查询语义配额、32 项小库、双人审核事件或 192 人时入口条件；真实语义扩展另立协议。
- 新控制面 `experiments/semantic-answerability-pilot-v1/`、数据面 `artifacts/semantic-answerability-pilot-v1/<dataset>/<stage>/<run-id>/` 先提交路径合同再用。
- 批量预留提交后启动；worker 不改 tracked 文件；failed/cancelled 不复用 ID/原地重试。旧 study/registry/输出保持不可变。
- C1 不取得四卡资格、不训练科学模型、不解封 test；科学授权拒绝守卫留到 C4 才凭证据升级。

## Review Focus

1. train 声明、allowlist 与实际身份不符或 symlink 越界：打开视频/caption 前拒绝；Task 2。
2. 官方标签被塞进 reviewed v1，或跨标签空间校准：v2 范围与 identity 明确，v1 约束不放宽；Tasks 3/5。
3. 同视频多 caption 被当多视频正例，精确重复 alias 同时正负：分别维护官方/canonical 关系；Tasks 3/4。
4. query 来源不在候选中导致跨 split 漏审，或切开大组件补配额：共同组件与数量门；Tasks 2/4/5。
5. CPU 准备权限复用于科学入口、未知 study 绕 gate：manager/worker 双重拒绝；Tasks 1/6。

## Scope and File Structure

| 文件 | 责任 |
|---|---|
| `src/semantic_retrieval/runtime/pilot_policy.py` | 路径/phase 授权，纯读控制面 |
| `src/semantic_retrieval/data/pilot_catalog.py` | 来源、内容身份、组件与分区 |
| `src/semantic_retrieval/data/benchmark_labels.py` | 官方关系、ID 映射、标注范围及候选存在计算 |
| `src/semantic_retrieval/data/pilot_compile.py` | 原始 catalog/关系到 v2 manifests/准备报告 |
| `scripts/prepare_semantic_pilot.py` | 受保护 CPU inventory/compile 入口 |
| `tests/semantic_retrieval/test_pilot_{policy,catalog,compile,cli}.py`、`test_benchmark_labels.py` | 合成数据与身份拒绝验证 |

C2 再写 minibatch/DDP、K/panel 调度、A0–A4/P1 与父 checkpoint；C3 再写跨库/候选分布校准、指标及图；C4 再写四卡资格及自动编排。C1 提供不依赖模型分数的关系表/库/identity；CLIP 排序后的实际候选由 C2 生成，不在 C1 偷跑。

## Task 1: 路径合同与拒绝未授权科学运行

**Files:** Modify `AGENTS.md`, `docs/agent/experiment-artifacts.md`, `scripts/manage_experiment_run.py`, `src/semantic_retrieval/runtime/artifacts.py`; Create `src/semantic_retrieval/runtime/pilot_policy.py`, `experiments/semantic-answerability-pilot-v1/{README.md,study.yaml,registry.jsonl}`, `tests/semantic_retrieval/test_pilot_policy.py`。

**Interface:** `assert_pilot_authorized(*, repo_root: Path, registry_path: Path, manifest: dict, operation: str) -> None`，operation=`reserve`/`start`。manager 在目录/reservation 生成前调用，worker 在输入打开前调用。study、控制面路径、数据面 namespace 必须一致；改 study 名仍用 pilot 路径/variants 必须拒绝。

- [ ] 写失败测试：draft 全拒绝；design_approved 只允许 dataset=msrvtt、stage=baseline、D1 variant=`source_inventory`/`benchmark_compile`、gpu_count=0、test_authorized=false。A0–A4/P1、GPU、test、未知 variant 在 reserve/start 都拒绝。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_policy.py`，确认失败来自缺少 policy 而非无效 fixture。
- [ ] 实现 policy，新 study 只为 `status: design_approved`、`scientific_authorized: false`、`stage0_passed: false`，记录已提交 spec 的 commit/SHA、修订数量门及 176 GPU 小时预算；registry 零字节。引用 SHA 从真实文件计算，不填占位。复用既有 manifest 严格结构，D1 编码 baseline+variant，不改旧 ALLOWED_STAGES。
- [ ] 提交 AGENTS/归档文档的新路径合同与启动前必读要求；保留旧研究规则。验证控制面 tracked、身份/哈希一致；路径允许不等于科学允许，C1 对科学分支固定拒绝，改 YAML 布尔值不能绕过。
- [ ] 重跑 policy 及 `tests/test_manage_experiment_run.py tests/semantic_retrieval/test_runtime_artifacts.py`，确认旧拒绝项不放宽，提交任务。

## Task 2: 来源 catalog、内容组件与确定性分区

**Files:** Create `src/semantic_retrieval/data/pilot_catalog.py`, `tests/semantic_retrieval/test_pilot_catalog.py`。

**Interfaces:** 冻结 `CatalogEntry(asset_id, kind, path, sha256, official_video_id, canonical_id, source_ids, caption_group_ids, frame_count, origin_split)`；kind=video/query，origin_split 仅 source_train，query 通过来源 ID 与 video 关联。`load_pilot_catalog(path: Path, *, expected_sha256: str, allowed_root: Path) -> tuple[CatalogEntry, ...]`；`assign_pilot_splits(entries, *, targets) -> SplitAssignment`，返回 asset→split/component、每库官方 ID、余量与数量报告。

- [ ] 写失败测试：unknown/test/mixed 来源、越界/symlink、伪 SHA/官方 ID 重复或身份冲突拒绝；视频/caption opener spy 在授权前调用数=0；读取 train-only catalog 元数据不扫描目录。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_catalog.py`，确认预期失败。
- [ ] 实现严格 catalog 及独立来源报告/allowlist SHA 绑定。元数据证明先于视频/caption 内容；只哈希/打开列明允许的文件，不全库搜索、不读取混合 test 文件。来源证据不足给 missing-input/stop report；不靠程序自签声明证明 train 身份。
- [ ] 以精确内容 SHA、canonical/source/caption 节点构造组件，component_id 为排序成员 asset_id canonical JSON SHA；官方 ID 哈希选择 held-out 库，caption 用稳定官方/行 ID。组件完整分配，数量依 spec，不以模型得分换库。
- [ ] 合成测试覆盖输入乱序映射不变、查询来源泄漏、组件跨目标边界、余量不回流和大组件导致数量门失败。目标参数只供小型 fixture；生产只能用冻结 spec 目标，测试通过后提交。

## Task 3: 官方关系与标注范围，替代人工审核事件

**Files:** Create `src/semantic_retrieval/data/benchmark_labels.py`, `tests/semantic_retrieval/test_benchmark_labels.py`。

**Interfaces:** `BenchmarkLabels(protocol_id, annotation_sha256, relation_sha256, id_map_sha256, positives_by_query, official_to_canonical)`；`load_benchmark_labels(snapshot, *, catalog, expected_sha256, protocol_id) -> BenchmarkLabels`；`candidate_presence(query_id, candidate_canonical_ids, labels) -> bool`。完整关系表/ID 映射另存不可变文件及 SHA，报告不嵌大型数组。

- [ ] 写失败测试：官方 caption→video 关系得到单标注正例，同视频多 caption 不扩充该 query 的视频集合；缺失、悬空 query/video、unknown 或非 train ID 不自动成为负例；规范化重复行只在一致时接受并报告，不覆盖矛盾关系。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_benchmark_labels.py`，确认预期失败。
- [ ] 实现官方 ID 正例 R_b 与 canonical R_c 的独立视图；内容 alias 正例用 R_c 合并，其余非成员只在显式 benchmark 范围下 negative。不引入语义相似检索或 LLM 自动补标签；来源通过后才解析允许的文本关系。
- [ ] 拒绝无正例/多 canonical 正例的本版 query，生成可追踪原因；缺失标签单独计 unknown/invalid。candidate_presence 仅判断 R_c 与 C 交集，输入候选在同库且 canonical 去重；删除候选 GT 不改变 gallery truth。
- [ ] 测试 GT 在/不在候选、GT 在全库、alias 复制及官方 ID 展开映射，确认 benchmark_negative 未产生 reviewed_semantic provenance；通过后提交。

## Task 4: 自动准备报告、数量门与候选 truth

**Files:** Modify `src/semantic_retrieval/data/benchmark_labels.py`; Create `src/semantic_retrieval/data/pilot_compile.py`, `tests/semantic_retrieval/test_pilot_compile.py`。

**Interfaces:** `summarize_pilot_inputs(*, catalog, assignment, labels) -> dict` 返回 ready/stop_reasons、split/hash、库/query/component 计数与正例直方图；`build_benchmark_panel(*, query_id, canonical_order, k, mode, labels) -> Panel`，mode=`natural`/`present`/`absent`/`controlled_miss`。函数纯 CPU，order 是显式输入，本任务不构造 CLIP 排序。

- [ ] 写失败测试：数量不足、跨来源泄漏、caption 选择不稳定、库 canonical<129、locked 组件<800、query GT 不在库或映射缺失都 ready=false；合成 fixture 用显式较小目标，生产配置不允许改目标。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_compile.py`，确认预期失败。
- [ ] 输出官方/canonical/query/component 数、未选库余量、缺失来源/标签及无真实语义审核声明；没有旧语义类别配额/预算。失败仍保存报告，不挑模型分数修数据。
- [ ] 实现 present=K−1 非正例+GT、absent=K 非正例、controlled_miss=GT 第 33；先验证 order 完整性、K 范围及非正例数量，再返回候选和离线 truth。自然/受控 provenance 分开，面板类别/GT 不进入 QueryInputs；候选集合按 canonical ID 表示，扩检 rank 单独保存。
- [ ] 小型合成排序测试自然 membership、成对 y_C=1/0、受控初始32缺失/64找回、alias 不增样本，输出引用原 query/component。边界测试128个 canonical 不支持128项 absent、129个支持，缺项报错而非补假视频；通过后提交。

## Task 5: 显式 benchmark v2 与完整身份传递

**Files:** Modify `src/semantic_retrieval/data/episodes.py`, `src/semantic_retrieval/config.py`, `src/semantic_retrieval/runtime/checkpoint.py`; extend `src/semantic_retrieval/data/pilot_compile.py`, `tests/semantic_retrieval/test_pilot_compile.py`, `tests/semantic_retrieval/test_episodes.py`, `tests/semantic_retrieval/test_end_to_end.py`, `tests/semantic_retrieval/test_trainer.py`; Create `tests/semantic_retrieval/test_benchmark_identity.py`。

**Interfaces:** `compile_pilot_manifests(*, catalog, assignment, labels, input_summary, preprocessing: EpisodePreprocessing) -> dict[str, dict]` 返回四分区 v2 episode payload、原生 gallery payload及独立标签/映射引用。C1 用 canonical ID 顺序生成合同样例，不充当科学粗检索；C2 在 CLIP 排序冻结后重新编译科学 episodes，并绑定新 SHA。v2 顶层显式 `relevance_space`/`label_protocol`；benchmark protocol 绑定官方 annotation、关系、ID 映射与来源报告 SHA。模型配置/checkpoint digest 必须含范围/协议 identity，现有 v1 路径继续用原合同。

- [ ] 写失败测试：官方 benchmark 已知标签可读，但不靠 reviewed=true；未带范围/协议的 v2、伪 provenance、跨范围 config/checkpoint 混用拒绝；v1 reviewed=false known/full-gallery-no 拒绝项仍有效。
- [ ] 运行相关 Episode/config/checkpoint 与 compile 测试，确认新断言预期失败。
- [ ] 实现严格 schema dispatch，不宽泛忽略未知字段。preprocessing=(32,8,224)，视频真实 frame_count 与内容 SHA 经解码/哈希核验；不把8当原视频帧数。benchmark gallery truth 表示标注 membership，不复制语义 gallery_audit；本版各自然 query 为 gallery=yes。
- [ ] `audit_episode_splits` 纳入 query_source_ids/query_component_id，与 Task 2 共用编码；官方/canonical ID 映射引用传递到配置/checkpoint，label scope 也供后续 C3 校准/指标绑定。C1 不实现跨库校准、不声称校准状态已兼容 v2，未升级消费者必须显式拒绝而非静默按 v1 解释。
- [ ] 用合成小视频验证 compile→read_episodes→联合 split audit→配置/checkpoint roundtrip，篡改哈希、跨源别名、标签范围冲突均拒绝；窄测通过后提交。

## Task 6: 受保护 CPU 入口与 C1 验收

**Files:** Create `scripts/prepare_semantic_pilot.py`, `tests/semantic_retrieval/test_pilot_cli.py`; Modify `docs/agent/semantic-retrieval-system.md`。

**Interfaces:** CLI 只接受 `--config` 和核对身份的 `--run-id/--manifest-sha256/--source-sha`；config 明确 runtime、mode=`inventory`/`compile`、catalog/allowlist/source_report/annotation path+SHA、allowed_root、protocol identity，inventory 尚未消费的 annotation 显式 null。所有已消费非 null 输入须在 manifest dataset_splits 绑定，mode/variant 一致，无任意 output-dir。

- [ ] 写失败测试：未提交 reservation、脏源码、mode/variant 不符、GPU 请求、错误 SHA、test/未知源、缺少 label scope 拒绝；使用临时 Git repo 和合成资产，不读真实数据。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_cli.py`，确认预期失败。
- [ ] 实现薄入口 prepare_run→policy→身份核验→start_run→inventory/compile→publish_metrics；输出只 run-local，终态沿既有 transition/finalize 收束。输入不可变，inventory 后 compile 是新 run 及 lineage，不在同 run 改 inputs/重试。
- [ ] 验证 protected roundtrip、失败证据保留、tracked 零漂移；数组用 ArtifactArrayStore、引用带 SHA。文档明确官方标签主流程、语义扩展可选、C2/C3/C4 未实现，不能声称能一键跑科学实验。
- [ ] 先跑全部新增测试（含 benchmark identity）及改动涉及的归档/Episode/config/checkpoint 测试；仅新失败/关切才扩大。最后 Ruff、git diff --check，提交实现与 C1 验收报告。

## Verification Environment

`python` 指 `/home/xujie/.conda/envs/tvr/bin/python`，环境 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:.`；pytest 带 `-p no:cacheprovider`。所有 fixture 为合成临时数据/仓库。真实输入导入、reserve/worker、资产下载、语义审核或 GPU 不属于本计划测试完成条件。

## Self-Review and Handoff

- spec 第3/4节由 Tasks 2–5 覆盖，第10节路径/守卫由 Tasks 1/6 覆盖；C1 只做无模型分数的数据准备。
- Task 1/6 身份在临时 Git repo 实测；范围/来源/数量门须有拒绝证据，不能只 mock success。
- Task 5 的 query 来源、组件/ID 映射与 Task 2/3 同源；v1 语义审核不放松，v2 官方标签不伪造人审。
- 尚无真实输入 SHA、CLIP 权重、run ID 或 GPU 资格，不虚构。真实 D1 仍须源码/路径合同提交、输入证明及整批 reserved 提交。
- 本次仅同步修订计划，不执行上述 checkbox，不重复要求用户批准已经指示的设计方向。
