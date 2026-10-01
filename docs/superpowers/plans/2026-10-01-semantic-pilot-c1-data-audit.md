# Semantic Pilot C1: Data and Audit Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现新研究的路径/授权守卫和确定性数据审核工具，能从明确允许的来源编译可追溯 episode；C1 验收仅使用合成数据。

**Architecture:** 新 pilot policy 在管理器预留和 worker 启动两处统一检查。数据工具依次处理来源 allowlist、连通组件划分、双人审核事件及只读 episode 编译，复用现有归档生命周期与 Episode schema，不引入旧 UATVR 模型。

**Tech Stack:** Python、现有 PyYAML/jsonschema/pytest、标准库 hashlib/csv/json、Scheme A；不新增依赖，不调用外部审核平台。

**Spec:** [已通过审阅的实验设计](../specs/2026-10-01-semantic-retrieval-experiment-design.md)，重点为第 3、9、10 节。

状态：待实施计划审阅，所有任务未执行。设计批准已记录；本计划不生成真实 run 或读取真实数据。

## Global Constraints

- MSRVTT/MSVD test 保持 sealed。来源未知直接拒绝，在授权判定前不得打开源视频或 caption。
- 分区按 canonical/source/caption-group 连通组件，hash namespace=`semantic-answerability-pilot-v1:`；70%/10%/10%/10%，余量归 train。
- 四分区各 32 个 canonical 视频、8 个主题各 4 个；每库候选审核上限 64。
- query 配额 train=(48,40,40)、development=(20,20,24)、calibration=(32,32,32)、locked=(96,96,96)，顺序为单/多/无答案。
- 40,000 次关系判断、192 人时先到停止；双人独立审核、第三人仲裁均计成本，unknown 不变 negative。
- locked 至少 200 个来源组件、每类至少 80；一致率低于 85% 停止；筛查查询数不超过各分区配额两倍。
- 新控制面 `experiments/semantic-answerability-pilot-v1/`；新数据面 `artifacts/semantic-answerability-pilot-v1/<dataset>/<stage>/<run-id>/`，先提交路径契约再使用。
- 先批量预留并提交，再启动；worker 不改受跟踪文件。failed/cancelled 不复用 ID、不原地重试。
- C1 不取得四卡资格，不训练科学模型，不解封 test。旧 study、registry、schema 和历史输出保持不可变。

## Review Focus

1. allowlist 声明与实际文件不同，或封存路径藏在符号链接中：打开内容前拒绝身份/路径违规；Task 2 验证。
2. 查询源视频未在候选库内，泄漏检查只审候选：查询来源仍纳入跨分区连通组件；Tasks 2/5 验证。
3. 重复提交同一审核事件或者第三人仲裁超预算：不能悄悄重复计数或覆盖原记录；Tasks 3/4 验证。
4. 某 query 有 31 个 negative 和 1 个 unknown：不能声明 32 项库无答案；Tasks 4/5 验证。
5. CPU 数据工具的授权被科学入口复用，或新 study 名绕过 gate：管理器/worker 双重拒绝；Tasks 1/6 验证。

## Scope and File Structure

| 文件 | 责任 |
|---|---|
| `src/semantic_retrieval/runtime/pilot_policy.py` | study/phase 授权检查，纯读控制面 |
| `src/semantic_retrieval/data/pilot_catalog.py` | 来源合同、组件与分区 |
| `src/semantic_retrieval/data/pilot_audit.py` | 审核事件校验、预算与仲裁状态 |
| `src/semantic_retrieval/data/pilot_compile.py` | 审核结果到原生 Episode/Gallery manifest |
| `scripts/prepare_semantic_pilot.py` | 受保护的 CPU 准备入口 |
| `tests/semantic_retrieval/test_pilot_{policy,catalog,audit,compile,cli}.py` | 五个职责的合成验证 |

C2 再写 deterministic minibatch/DDP schedule、A0–A5 模式和父 checkpoint 初始化的详细计划；C3 再写跨库校准、指标/可视化；C4 再写四卡资格及批次运行。C1 输出的 schema、身份和报告引用是后三部分的输入；C1 不声称覆盖整个实验实现。

## Task 1: 路径合同与拒绝未授权科学运行

**Files:** Modify `AGENTS.md`, `docs/agent/experiment-artifacts.md`, `scripts/manage_experiment_run.py`, `src/semantic_retrieval/runtime/artifacts.py`; Create `src/semantic_retrieval/runtime/pilot_policy.py`, `experiments/semantic-answerability-pilot-v1/{README.md,study.yaml,registry.jsonl}`, `tests/semantic_retrieval/test_pilot_policy.py`。

**Interfaces:** `assert_pilot_authorized(*, repo_root: Path, registry_path: Path, manifest: dict, operation: str) -> None`；operation=`reserve` 或 `start`。管理器在生成目录/追加 reservation 前调用，worker 在打开绑定输入前调用。manifest study、控制面路径与数据面 namespace 必须一致；改 study 名但仍使用 pilot 的路径/variants 直接拒绝。未在路径契约登记的新 study 拒绝，已有 study 走其既有合同，不把新状态规则写回旧控制面。

- [ ] 写失败测试：draft study 拒绝全部操作；design_approved 仅允许 dataset=msrvtt、stage=baseline、variant=`audit_export`/`audit_compile`、gpu_count=0、test_authorized=false 的 D1 CPU 准备；A0–A5/P1/P5、任何 GPU、test、未识别 variant 一律拒绝。start 同样拒绝，不能只守 reserve。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_policy.py`，确认失败来自缺少 policy，非 fixture 不合法。
- [ ] 实现上述 policy；study 只设 `status: design_approved`、`scientific_authorized: false`、`stage0_passed: false`，并记录已通过设计的 commit 与完整预算/配额。新 registry 为零字节。spec SHA 必须取提交后文件实际哈希，不能填占位值。旧 schemas 的严格结构可复用，不修改 `ALLOWED_STAGES`；D1 用 baseline + 明确 variant 编码。
- [ ] 更新 AGENTS 的允许路径表与启动前必读要求：保留旧 study 规则，对新 study 只增加该独立物理路径；路径允许不等于科学运行允许。保护函数验证控制面受跟踪、契约身份/哈希一致；不允许修改 YAML 一个布尔值就绕过后续 Stage 0/数据证据。C1 对科学分支保持固定拒绝，后续 C4 才实现证据 gate。
- [ ] 重跑上述测试及 `tests/test_manage_experiment_run.py tests/semantic_retrieval/test_runtime_artifacts.py`；确认旧契约拒绝项未放宽；提交本任务。

## Task 2: 明确来源、连接组件与确定性划分

**Files:** Create `src/semantic_retrieval/data/pilot_catalog.py`, `tests/semantic_retrieval/test_pilot_catalog.py`。

**Interfaces:** 冻结 dataclass `CatalogEntry(asset_id, kind, path, sha256, canonical_id, source_ids, caption_group_ids, frame_count, origin_split)`；kind=`video`/`query`，origin_split 只接受显式 source_train，frame_count 对 video 为正整数、query 为 null。`load_pilot_catalog(path: Path, *, expected_sha256: str, allowed_root: Path) -> tuple[CatalogEntry, ...]`；`assign_pilot_splits(entries: tuple[CatalogEntry, ...]) -> tuple[dict[str, str], dict[str, str]]` 分别返回 asset_id→split 与 asset_id→component_id，component 由函数计算，不信任输入声明。

- [ ] 写 `test_sealed_source_rejected_before_open`：unknown/test origin 或不安全路径引发 ValueError，视频/caption opener spy 调用数=0；写 `test_catalog_hash_and_identity_mismatch`：伪造 SHA、重复 ID、不一致 canonical 身份拒绝；写 `test_component_split_is_permutation_invariant`：同源/caption-group/canonical 连接不可跨分区，乱序输入的两张映射均相同。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_catalog.py`，确认预期失败。
- [ ] 实现严格 JSON catalog：schema_version=1、source_provenance、assets；拒绝未知字段。source_provenance 带 reviewed/source/sha256，资产路径只在已声明 allowed_root 内。元数据判定先于内容哈希；内容哈希采用现有 `file_sha256`，不扫描目录，不打开任何未列入 catalog 的文件。catalog 的语义来源声明仍需人审，代码不声称自行证明训练集身份。
- [ ] 组件合并依据所有 canonical/source/caption 节点；component_id 取排序成员 asset_id 的 canonical JSON SHA256。组件按 spec 的 namespace hash 排序；先取 floor(0.7N)、floor(0.1N)、floor(0.1N)、floor(0.1N)，剩余组件归 train。query 本身与其来源 video 的节点关联，不能只检查候选。
- [ ] 测试加入 100 组件精确计数 70/10/10/10、103 组件余量归 train，以及 query-source 跨库泄漏拒绝；验证后提交。

## Task 3: 双人盲审事件与可核对的成本台账

**Files:** Create `src/semantic_retrieval/data/pilot_audit.py`, `tests/semantic_retrieval/test_pilot_audit.py`。

**Interfaces:** `AuditEvent(event_id, query_id, canonical_id, reviewer_id, role, label, elapsed_seconds, supersedes_event_id)`；role=`primary`/`adjudicator`，label=`positive`/`negative`/`unknown`。`validate_audit_events(events: tuple[AuditEvent, ...]) -> None`；`audit_cost(events: tuple[AuditEvent, ...], *, screening_seconds: int) -> dict`。

- [ ] 写失败测试：同一关系两名 primary 必须不同 reviewer；主审核不能看对方标签；重复 event_id 拒绝；第二次 primary 不能覆盖第一次。仲裁只接受真实分歧、第三名不同 reviewer、指向已有事件的 lineage；unknown 未经合法仲裁不能变 known。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_audit.py`，确认预期失败。
- [ ] 实现 append-only JSONL 单事件合同和纯校验；不存模型分数/预测。export 产生两个彼此独立的本地任务清单，import 合并审核事件；不发送邮件、消息或调用外部审核平台。不将签名字段当真人身份认证的替代，报告注明操作者声明。
- [ ] 成本按 primary/adjudicator 判断条数和全部非负 elapsed_seconds 统计，筛查/看库/仲裁时间纳入总时长；40,001 次判断或 192 小时之后的提交拒绝并保留原历史。失败导入不写半批事件；发现重复历史不能静默去重。
- [ ] 加边界测试 40,000 可接受、40,001 拒绝，192 小时可接受、超限拒绝，负时长与循环/悬空仲裁 lineage 拒绝；重跑测试后提交。

## Task 4: 配额、完整库审核与停止报告

**Files:** Modify `src/semantic_retrieval/data/pilot_audit.py`; extend `tests/semantic_retrieval/test_pilot_audit.py`。

**Interfaces:** `summarize_pilot_audit(*, catalog, split_map, component_map, galleries, events, screening_log) -> dict` 返回 ready、stop_reasons、配额/一致率/unknown/来源组件计数和预算，报告不能包含大数值数组。

- [ ] 写失败测试：31 negative+1 unknown 不能成为 no-answer；alias 复制不是第二正例；多答案需两个不同 canonical 正例；配额不足、库未达 32、主题不是 8×4、库候选超过 64、筛查超过两倍上限必须 ready=false。
- [ ] 运行审计测试，确认对应断言失败。
- [ ] 实现先审核后归类；只用两名原始 primary 计算一致率，unknown=unknown 可以一致，但另报 positive/negative 混淆与 unknown 比例；不得用仲裁标签改善原始一致率。完整归类之外的事件仍保存与计费。
- [ ] 验证 576 查询配额、locked≥200 个组件且每类≥80，以及一致率 85% 边界；任何不足只生成 stop report，不补伪标签、不按模型表现换库。
- [ ] 合成完整/不完整数据分别得到 ready=true/false，确认所有 stop_reasons 可溯源；提交。

## Task 5: 编译原生数据合同并补齐查询来源审计

**Files:** Create `src/semantic_retrieval/data/pilot_compile.py`, `tests/semantic_retrieval/test_pilot_compile.py`; Modify `src/semantic_retrieval/data/episodes.py`。

**Interfaces:** `compile_pilot_manifests(*, catalog, split_map, component_map, galleries, events, audit_summary, preprocessing: EpisodePreprocessing) -> dict[str, dict]` 返回四分区的 episode/gallery payload。C1 输出完整 32 项候选及审核 truth；CLIP 排序/K 前缀由 C2 生成，C1 不读取模型权重。

- [ ] 写失败测试：budget/ready=false 拒绝编译；全负已审核 query 可得 gallery no；unknown 不能得 gallery no；跨分区 query_source_ids 即使不在候选里也被审计拒绝；失败前不发布部分 manifest。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_compile.py`，确认预期失败。
- [ ] 复用 Episode/Gallery schema_version=1，固定 preprocessing=(max_text_length=32,max_frames=8,image_size=224)。provenance 新增 pilot query_source_ids/query_component_id、审核事件/完整报告引用与 SHA，gallery no 另附 scope=complete。标签不写入 QueryInputs，不生成伪 GT。
- [ ] 扩展 `audit_episode_splits` 对带 pilot provenance 的 query sources/components 做交叉检查；既有非 pilot fixture 的合同保持可读。四 manifest 联合审计，不仅单文件 read_episodes。核对 catalog 的真实视频 frame_count，使用现有 `sample_frame_indices`/`decode_video` 完成所需帧解码，元数据与实际可解码帧不符则拒绝并记录；不把 max_frames=8 当原视频帧数。
- [ ] 用合成小视频完成 compile→read_episodes→联合 audit roundtrip；错内容 SHA、同来源别名、query 来源泄漏均拒绝；运行 `tests/semantic_retrieval/test_episodes.py tests/semantic_retrieval/test_video_and_splits.py`，通过后提交。

## Task 6: 受保护 CPU 入口与整个 C1 验收

**Files:** Create `scripts/prepare_semantic_pilot.py`, `tests/semantic_retrieval/test_pilot_cli.py`; Modify `docs/agent/semantic-retrieval-system.md`。

**Interfaces:** CLI 只接受 `--config` 及核对身份的 `--run-id/--manifest-sha256/--source-sha`。config 精确包含 runtime、mode=`export`/`compile`、catalog path/hash、audit inputs path/hash、screening log path/hash、allowed_root；export 的无事件输入显式 null。所有非 null 输入必须在 manifest dataset_splits 绑定，mode 与对应 variant 一致。

- [ ] 写失败测试：未提交 reservation、脏源码、mode/variant 不符、额外 --output-dir、GPU 可见、错误 SHA、test 或未允许源路径拒绝；使用临时 Git 仓库与合成 catalog，不读取真实数据。
- [ ] 运行 `python -m pytest -q tests/semantic_retrieval/test_pilot_cli.py`，确认预期失败。
- [ ] 实现薄入口；prepare_run→pilot policy→输入核验→start_run→export/compile→publish_metrics。所有输出只入 run-local；生命周期与终态由既有管理器收束，不绕过 transition/finalize。export 是离线任务清单，compile 必须绑定既有审核输入；不在同 run 反复修改 inputs。events/import 成为新 immutable input 的新 run，保留 lineage。
- [ ] 验证 protected roundtrip、预期失败证据保留及 tracked 文件零漂移；报告引用采用 SHA，数值数组复用 ArtifactArrayStore。文档说明 C1 工具与真实 D1 人审/运行的区别。
- [ ] 先跑全部 `test_pilot_*.py`，再运行被改动的归档与 Episode 测试；新增失败或关切才扩大到全套。最终运行 Ruff 与 `git diff --check`，提交 C1 验收报告和实现。

## Verification Environment

上述 `python` 均为 `/home/xujie/.conda/envs/tvr/bin/python`，加 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:.`；pytest 使用 `-p no:cacheprovider`。fixture 仅合成临时数据/临时仓库。空新 registry、真正 reserve/worker、审核邀请、资产下载和任何 GPU 运行不在本计划的测试完成条件中。

## Self-Review and Handoff

- spec 第 3 节由 Tasks 2–5 覆盖；第 10 节的路径/拒绝守卫由 Tasks 1/6 覆盖。模型、统计图和 Q0 属于明确的 C2/C3/C4 边界，不声称 C1 已实现。
- Tasks 1/6 的运行身份必须在临时 Git repo 实测，不能只 mock 出 success；来源/预算测试必须有拒绝证据。
- Task 5 的 query 来源 provenance 是 Task 2 的 source/component 输出；不得使用不同 ID 编码。
- 尚无真实输入 SHA、CLIP 权重、run ID 或 GPU 拓扑，本计划不虚构。真实审核运行要在 source 提交、输入审核和整批 reserved 提交后另行执行。
- 推荐沿用当前会话逐项执行；实施前先审阅此计划。设计审阅通过已记录，不重复请求同一设计的批准。
