# 无人值守总控、自动恢复与 Codex 巡检 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the current session. Steps use checkbox (`- [ ]`) syntax for tracking. 不默认派生子代理；需要独立审阅时按适用技能执行。

**Goal:** 实现服务器上持久化跨阶段执行、可认证中断后的 checkpoint 续训、Codex 定时巡检及可查看的实验总览。

**Architecture:** Python campaign 总控调用现有配置工厂与 Scheme A 管理器；独立 supervisor 持有真实 worker 退出证据，重启后按认证归档接续。Codex 只提交结构化建议，确定性动作校验器决定是否执行；systemd 管理总控与巡检生命周期。

**Tech Stack:** 现有 Python/PyTorch、strict JSON/jsonschema、Linux /proc/flock/process group、systemd 用户服务、服务器 codex exec；不新增数据库、工作流平台或模型 API SDK。

**Spec:** [设计文档](../specs/2026-10-01-unattended-pilot-design.md)。2026-10-01 用户要求按此设计编写开发计划；本计划不启动实验或部署服务。

## Global Constraints

- 唯一科学 study 为 semantic-answerability-pilot-v1；历史 study 不重开，MSRVTT/MSVD test 始终 sealed。
- 冻结拓扑为物理 GPU 0、2、3、4 的认证 UUID，四 CUDA rank、FP32、TF32 off；新实现须重新通过新源码四卡 Q0。
- 保留全部失败证据和 immutable terminal run；successor 使用后续唯一 phase/run ID，绑定 predecessor、自动审核报告、策略与 checkpoint 引用。
- 第8条已删除，不再逐次要求人工重试批准；不能跳过其余归档、源码、输入、预算或研究 gate。
- worker/supervisor 存活期间不得修改受跟踪文件；先提交整批预留，后启动；全部 worker 和输出写入者停止后再提交终态。
- artifacts/semantic-answerability-pilot-v1/_orchestration/<campaign-id>/ 是新增总控数据目录；数值数组仍采用现有外置引用规则。
- 最多2次训练 successor，退避30秒、120秒，所有尝试累计预算：training144、qualification8、evaluation24、总计176 GPUh；每逻辑训练run8 GPUh，不因重试重置。
- 自动checkpoint续训仅P1a/P1b/P2a/P2b；非训练失败、OOM、NaN/Inf、DDP/代码错误、未知退出、checkpoint损坏、gate失败停止；资源不足尚未启动则等待。
- 轻量巡检60秒；Codex普通巡检30分钟、异常冷却5分钟、单次超时10分钟；每日最多48次普通＋12次异常调用，日界线Asia/Shanghai。
- Codex网络/认证/额度异常只降级巡检；模型不可直接执行命令、改超参、改冻结GPU、解封test或部署修复代码。
- 用户stop持久生效；不默认启用无限后台实验，不默认发送邮件或外部消息。

## Review Focus

1. 总控在 Popen 与 PID 元数据发布之间被杀：身份可证时接管，不可证时停止，绝不启动第二份训练（任务2、6）。
2. 主机重启、时钟回拨或 supervisor 退出导致退出码/成本缺失：不推断成功，预算采用保守上界，不能借故障清零（任务2、3）。
3. 最新 checkpoint 或其外置数组损坏：不能偷偷回退到更旧 checkpoint；只有未完整发布的临时文件可忽略（任务3）。
4. 同一 arm/seed 成功两次、前驱有checkpoint后多余日志：拒绝歧义，只拼接恢复实际继承的迭代区间（任务4）。
5. 日志中含指令、建议过期、Codex缺失usage：视日志为数据，动作需重新核验；usage无法核算则暂停巡检，健康worker不受影响（任务7）。

## 文件与交付边界

新建 runtime/campaign/ 包，各文件只负责一种职责：contracts.py（严格类型/配置）、store.py（journal/锁）、supervisor.py（进程监督）、resources.py（GPU与心跳）、recovery.py（故障/预算/checkpoint）、evidence.py（逻辑阶段与日志链）、transactions.py（幂等控制面操作）、controller.py（状态机）、inspection.py（Codex调用和建议）、reporting.py（认证结果汇总）。路径前缀均为 src/semantic_retrieval/。

文件段落中 runtime/ 均展开为 src/semantic_retrieval/runtime/；所有未写目录的 test_campaign_*.py 与 test_pilot_*.py 均位于 tests/semantic_retrieval/；CONTROL 固定展开为 experiments/semantic-answerability-pilot-v1/。

CLI 为 scripts/run_semantic_campaign.py、scripts/supervise_semantic_worker.py、scripts/inspect_semantic_campaign.py。不让原 orchestrate_semantic_pilot.py 同时成为服务、监督和智能代理。

分三组可独立验收的交付：A监督与持久合同（任务1–2）；B恢复和跨阶段编排（任务3–6）；C巡检、服务、报告与最终验收（任务7–9）。任务按依赖顺序执行，逐项提交；源码验收期间仅使用CPU/Gloo/假Codex，真实GPU资格单独归档。

## 任务1：campaign 合同、锁和 journal

**Files:** 新建 runtime/campaign/{__init__,contracts,store}.py、schemas/semantic-campaign.schema.json；修改 AGENTS.md 与 docs/agent/experiment-artifacts.md 仅登记总控数据路径；测试 tests/semantic_retrieval/test_campaign_contracts.py。

**Interfaces:** `load_campaign(path: Path, repo: Path) -> CampaignConfig`；`open_store(config: CampaignConfig) -> CampaignStore`；store提供`append(event: dict) -> str`、`snapshot() -> CampaignSnapshot`、`request_stop(reason: str) -> None`。类型均在contracts.py定义，snapshot含campaign_id/revision/state/active_runs/stop_reason；状态限定ready/waiting/running/recovering/blocked/completed/stopped，巡检degraded另设字段。

- [ ] 写RED测试：非法study/test/路径穿越/符号链接/非有限数拒绝；同campaign双锁失败；journal重复operation_id不重写，不同payload拒绝；末尾半条记录只保留隔离证据，不擅自补写。
- [ ] 运行 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/test_campaign_contracts.py`，确认缺失接口失败。
- [ ] 实现严格配置：campaign_id、code_anchor_sha、environment/input refs、固定uuid、阶段序列、每卡最低空闲显存、完整预算、恢复策略、巡检限额。controller HEAD 可因纯控制面提交变化，但manifest每次仍绑定精确启动SHA；code_anchor_sha通过现有源码比较守卫约束。生产配置必填各卡最低显存字节值，不猜测阈值；源码最终完成后才生成可启动配置。
- [ ] 用flock分别管理campaign独占锁、资源协作锁和控制面写入锁；supervisor持共享写入屏障直至最后输出关闭，总控需独占屏障才能提交。journal带序号/前哈希/操作身份，fsync后原子发布snapshot；恢复真源仍为认证registry/manifest/output。
- [ ] 上述测试PASS、Ruff和diff检查后提交 `feat: add persistent campaign contracts and locks`。

## 任务2：独立 supervisor 和资源状态

**Files:** 新建 runtime/campaign/{supervisor,resources}.py、scripts/supervise_semantic_worker.py；测试 test_campaign_supervisor.py、test_campaign_resources.py。

**Interfaces:** `start_supervisor(config: CampaignConfig, identity: dict) -> SupervisorRef`；`probe_supervisor(ref: SupervisorRef) -> SupervisorObservation`；`sample_resources(config: CampaignConfig) -> ResourceSnapshot`；`resources_ready(snapshot: ResourceSnapshot, config: CampaignConfig) -> bool`。类型在contracts.py，observation保存身份、进程存活、实际exit_code或None、故障证据、成本上界和证据引用。

- [ ] RED测试：真实短子进程exit0/7、总控子进程退出后supervisor继续收集exit7、PID复用/boot变更拒绝、启动元数据写入异常后清理仅自己进程组、启动窗口不可证明则block；模拟GPU占用只能wait。
- [ ] 运行任务2两个测试文件，确认缺失接口失败。
- [ ] supervisor先核验verify-run并写launch-plan，再Popen一次sealed argv；argv不能来自Codex。记录boot ID、/proc start ticks、PID/PGID、命令hash、启动时间。launcher退出后清理其残留rank并等待日志关闭，再发布成本/退出凭证和supervisor完成标记。监督进程独立于总控的systemd cgroup，使用独立transient unit；CPU测试可用独立进程模拟，不能拿模拟声称systemd通过。
- [ ] 资源监控解析nvidia-smi结构化字段，启动前二次核验UUID/显存且尊重已有compute进程；60秒采样，不杀外部进程。心跳仅报告活性；长训练/分析无进度不得直接判定可重试。退出码缺失保持None；预算回拨或计时证据不可靠时取登记上限，不创造资源余额。
- [ ] 验证两测试文件PASS后提交 `feat: supervise sealed workers independently of campaign controller`。

## 任务3：中断分类、预算与 checkpoint 选择

**Files:** 新建 runtime/campaign/recovery.py；必要时给runtime/checkpoint.py增加只读验证公共包装，保留load_checkpoint的rank检查；测试 test_campaign_recovery.py。

**Interfaces:** `classify_interruption(observation: SupervisorObservation, evidence: dict) -> FailureClass`；`decide_recovery(config: CampaignConfig, attempt_chain: list[dict], observation: SupervisorObservation) -> RecoveryDecision`；`validate_resume_checkpoint(parent: dict, config: SystemConfig) -> CheckpointRef`。FailureClass限定host_reboot/recorded_preemption/user_stop/budget_exhausted/code_or_data/unknown；decision含allowed/reason/backoff_seconds/remaining_gpu_hours/checkpoint或None。

- [ ] RED测试：exit137/143无证据拒绝；认证boot变化可分类；无checkpoint、超过2次successor、30/120退避、累计超过8run/144training/176total拒绝；clock回拨成本保守计入；最新已发布checkpoint或array损坏拒绝，不选旧文件。
- [ ] 运行该测试文件观察RED。
- [ ] 复用已认证terminal输出inventory、checkpoint二进制/meta/数组哈希及配置/模型验证；检查progress、sampler和4rank RNG完整，不在CPU只读检查中绕过实际恢复world_size检查。忽略未原子发布的临时文件，最新已发布文件按progress而非mtime排序。
- [ ] 只接受先于故障持久保存、身份绑定的preemption/服务停止凭证或可认证boot变化；用户stop及超预算无自动恢复。取可证明的elapsed上界含全部前驱与重算；时间证据丢失则计满该尝试剩余预算，并据此可能停止。
- [ ] 测试PASS并提交 `feat: validate bounded checkpoint recovery decisions`。

## 任务4：successor phase 与逻辑阶段认证聚合

**Files:** 新建 runtime/campaign/evidence.py；修改runtime/{pilot_protocol,pilot_workflow,pilot_evidence,artifacts}.py、必要的schemas/semantic-campaign.schema.json；测试 test_campaign_evidence.py及现有test_pilot_protocol.py/test_pilot_evidence.py/test_pilot_workflow.py。

**Interfaces:** `build_successor(repo: Path, config: CampaignConfig, parent: dict, decision: RecoveryDecision, attempt_id: str) -> dict`返回可提交report/template/phase；`verify_logical_phase(repo: Path, reference: dict, logical_phase: str) -> dict`返回与既有verify_phase_evidence相同的phase/summary/runs结构；`read_training_chain(repo: Path, run: dict) -> dict`返回现有losses/gradients报告结构。

- [ ] RED测试：原批部分完成后successor仅含需要恢复的成员；缺arm、双成功、换seed/config/数据、前驱不terminal、伪造policy/report、模型代码变化拒绝；跨run正确续训日志覆盖iteration1..120/240且梯度覆盖20间隔。
- [ ] 运行新测试及现有三文件，观察新合同RED并保留既有batch限制断言。
- [ ] 明确marker版本2：logical_phase、attempt_index、原始phase引用、恢复report/policy引用；phase_id继续为固定logical ID，phase_path/phase名称为新唯一attempt，只有该合同允许subset。常规版本1仍要求固定完整batch。不扩展通用manager为任意subset放行。
- [ ] 自动review report位于CONTROL/recovery/<campaign-id>/<attempt-id>.json，提交后由training_resume lineage引用；approved表示策略核验通过，另标reviewer_kind=policy，不冒充人工。保持ResumeConfig字段兼容，强化artifacts._validate_resume的Git内容/hash、策略和前驱终态核验；绑定全部输入/meta/arrays和原始科学依赖。
- [ ] 逻辑summary单独类型，包含全部attempt summary引用和按contract顺序唯一完成成员；完整失败批次绝不标completed。verify_phase_evidence在识别类型后递归认证，不接受混合伪summary。训练日志按checkpoint继承边界拼接：前驱只取<=恢复iteration，后继取>该iteration；只读并保留前驱重算尾部；多级链检查无环、连续、无重叠。pilot_evidence与C0训练门使用read_training_chain。
- [ ] 新旧测试PASS后提交 `feat: authenticate recovered logical phases and training evidence`。

## 任务5：幂等控制面事务和断档收尾

**Files:** 新建runtime/campaign/transactions.py；修改scripts/orchestrate_semantic_pilot.py提取可复用plan/config/finalize函数；测试 test_campaign_transactions.py、现有test_pilot_orchestration.py。

**Interfaces:** `ensure_phase(config: CampaignConfig, logical_phase: str, evidence: dict, operation_id: str) -> dict`；`reconcile_attempt(config: CampaignConfig, run: dict, observation: SupervisorObservation) -> dict`；`publish_logical_summary(config: CampaignConfig, logical_phase: str) -> dict`（返回{path,sha256,commit}）。

- [ ] RED测试逐个注入：配置发布后未commit、phase提交后未reserve、reserved追加后未commit、worker已validating但未terminal、terminal registry追加后未commit、summary发布后未commit；重复操作不双预留、不覆盖不一致字节。模拟任何live supervisor/worker时控制面提交拒绝。
- [ ] 运行新旧事务测试确认RED。
- [ ] 使用现有manager reserve/verify/transition/finalize与strict writer；事务journal先持久记录确定的operation_id/目标字节/预期HEAD。只能收尾完全一致的未完成操作，不能把任意dirty源码作为自己操作提交。registry append-only，不批量改历史。
- [ ] archive结果仅在supervisor完成、所有输出写入者退出并验证exit凭证后终态；没有exit凭证不能按产物齐全推断PASS。boot中断可归档明确失败/未知退出并附成本上界。旧reserved代码不兼容时列清单block，需单独明确处置，不能原地更新或自动消失。
- [ ] 测试PASS后提交 `feat: reconcile campaign archive transactions idempotently`。

## 任务6：跨阶段总控 CLI

**Files:** 新建runtime/campaign/controller.py、scripts/run_semantic_campaign.py；测试 test_campaign_controller.py。

**Interfaces:** `reconcile_campaign(config: CampaignConfig, store: CampaignStore) -> CampaignSnapshot`；`run_campaign(config: CampaignConfig) -> int`。CLI：`--config ABS {validate,status,run,stop}`；stop可带--reason，不要求取得正在运行总控的独占锁才能发布stop request。

- [ ] RED测试CPU假的已认证阶段链从D1/Q0P推进到V0；重启在reserved/running/validating各窗口不重复launch；成功siblings不因另一成员失败被取消；phase所有成员终态后再处理有限successor；资源不足wait；用户stop后service重启不续跑；C0 gate失败拒绝V0E/V0S/V0。
- [ ] 运行该文件确认RED。
- [ ] 以任务1–5接口串联固定DAG；不自创科学phase。启动先validate所有输入/环境/权重、clean tree与source anchor、现有reserved冲突。沿用认证D1/Q0P，Q0针对最终可执行源码新预留；控制面commit不破坏资格判定。资源/结果等待期间仅写数据面。
- [ ] stop先持久写请求再由总控通知自己的supervisor清理worker，保留取消证据；blocked/stopped/completed是持久退出状态，服务重启不改变。新成功phase引用传给既有工厂，C0/V0仍绑定冻结model/gallery/policy身份。
- [ ] 测试PASS、CLI --help和validate假配置后提交 `feat: run and resume the authenticated pilot campaign`。

## 任务7：Codex 巡检及受限动作校验

**Files:** 新建runtime/campaign/inspection.py、scripts/inspect_semantic_campaign.py、schemas/campaign-inspection.schema.json；测试 test_campaign_inspection.py。

**Interfaces:** `collect_inspection(config: CampaignConfig, snapshot: CampaignSnapshot) -> dict`；`invoke_codex(config: CampaignConfig, payload: dict, kind: str) -> InspectionResult`；`validate_suggestion(config: CampaignConfig, current: CampaignSnapshot, result: InspectionResult) -> ActionDecision`。类型在contracts.py，建议必须包含campaign_id/snapshot_revision/phase/run_id/manifest_sha256/action/reason/evidence_refs，additionalProperties=false；无目标动作字段允许null，拒绝shell/argv/config_patch。

- [ ] RED测试假CLI输出合法建议、过期revision、外来run、日志prompt injection、额外command字段、无final JSON/exit非0、网络/额度错误、超时、无usage、跨午夜日计数、双巡检锁及5分钟冷却。所有错误均不杀健康worker。
- [ ] 运行该文件确认RED。
- [ ] 调用本地已核验CLI：codex exec --sandbox read-only --json --output-schema ABS_SCHEMA -o ABS_RESULT -C ABS_REPO，prompt通过stdin传入，不shell拼接；凭证留CLI自身auth，不继承给worker。在受控Codex profile中禁用额外MCP/hooks/tools，能力仅读取准备好的状态数据；验收可写性确实受限，不能仅凭prompt认定只读。
- [ ] 调用前持久计数，防崩溃后无限免费调用；使用Asia/Shanghai日窗口，普通48、事件12、600秒timeout；输入最多64KiB，单日志尾部最多8KiB，总输出最多1MiB。配置提供单次与每日token阈值，解析usage日志累计；缺usage则degraded停止下一调用。CLI未提供已验证的硬token额度控制：阈值可有一次调用超额，记录超额并停止后续，严格硬上限依赖账号/provider额度，不虚称精准限额。
- [ ] 建议只投递journal，由总控在下一reconcile重新计算resources/gates/recovery许可。Codex stop建议只对认证严重故障生效，纯主观优化建议不能停止健康实验；diagnose不会自动修科学代码。认证/网络异常退避30/120秒后等下一窗口，不自改登录凭证。
- [ ] 测试PASS后提交 `feat: add bounded Codex inspection and validated campaign advice`。

## 任务8：systemd 服务、总览与操作文档

**Files:** 新建deploy/semantic-campaign/{campaign.service.in,inspection.service.in,inspection.timer.in}、scripts/install_semantic_campaign_service.py、runtime/campaign/reporting.py、docs/agent/unattended-semantic-campaign.md；测试 test_campaign_service.py、test_campaign_reporting.py。

**Interfaces:** `render_units(config: CampaignConfig) -> dict[str,str]`；`write_campaign_report(config: CampaignConfig, snapshot: CampaignSnapshot) -> Path`；安装CLI默认dry-run，仅显式--install安装、--enable启用。报告输出在campaign数据目录，引用认证指标/图表/summary；未完成标unavailable。

- [ ] RED测试unit路径带空格正确转义、重复安装字节一致、环境中无训练凭证、timer不重叠、stop持久化、report引用hash失配拒绝、degraded但训练运行显示一致；通知状态不变不重复，无邮件发送。
- [ ] 运行两测试文件观察RED。
- [ ] systemd总控Restart=on-failure、RestartSec=30；CLI blocked/stopped/completed退出0避免反复拉起；巡检timer每30分钟，Persistent=true补一次而非补48次，事件由总控请求同一巡检unit且受cooldown/锁限制。独立supervisor transient units的cgroup不隶属总控，KillMode=control-group使其自身stop只清理自己的worker。
- [ ] 安装前systemd-analyze --user verify；验证登录退出后的user manager存活/linger实际状态，未启用就明确验收不通过，不默默请求系统管理员权限。机器开机自动恢复的部署条件包括user manager开机启动和持久auth可用。账号不可用只降级巡检。
- [ ] 文档给出validate、install dry-run/install/enable、run/status/stop、journalctl命令与卸载服务而不删实验产物的步骤；报告列各18训练臂状态、检索/校准/消融/敏感性输出、失败原因、预算、待处理事项。状态变更写append-only通知，不默认外发。
- [ ] 测试PASS、unit静态验证后提交 `feat: package campaign services and authenticated progress reports`。

## 任务9：整体故障注入、源码资格与运行交接

**Files:** 新建tests/semantic_retrieval/test_campaign_e2e.py；扩展runtime/pilot_qualification.py与test_pilot_qualification.py的恢复证据检查（如需要）；新建docs/superpowers/reports/2026-10-01-unattended-pilot-acceptance.md、CONTROL/configs/campaign-template.json（只读模板，非自动启用配置）。

- [ ] RED集成测试采用真实短CPUworker＋独立supervisor＋假Codex，至少覆盖总控被杀后真实exit7存证、接管不双启动、事务中断接续、P1/P2两级successor、未知故障blocked、恶意建议不执行、服务停止不重启训练。不能用mock掉整个总控证明链路通过。
- [ ] 实际两rank Gloo连续与checkpoint续训比较：参数/AdamW状态max_abs_delta=0，sampler/RNG/scheduler一致；拼接loss/probe日志被现有stability/gradient门接受。损坏meta/array及4rank配置不匹配拒绝，不能修改guard以适配测试。
- [ ] 先运行新增campaign相关测试及修改的pilot/runtime测试，再 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests -p no:cacheprovider`；Ruff检查所有改动Python，git diff --check。不用默认全目录pytest误收集未跟踪deploy_qwen。
- [ ] 做真实user systemd临时服务验收：总控退出不会杀supervisor，巡检timer/stop生效；一次最小非交互Codex只读巡检验证认证/schema/usage，报告真实额度消耗。网络失败可判degraded，不能宣称巡检可用；掉电测试若仅模拟boot变化须明确标模拟。
- [ ] 审阅全部实现，提交最终可执行源码，再生成绑定该源码的campaign配置和新Q0预留，保留旧Q0记录与冲突处置报告；整批reserved提交后verify-run。四卡不空闲时wait，不拿双卡预检代替Q0。Q0必须实际四CUDA rank、测梯度/恢复/C3资格并提交成功终态，未完成前不能科学启动。
- [ ] acceptance报告分开记录CPU、systemd、真实Codex、四卡Q0、科学启动五项证据，未执行不标PASS；生成最终一条启动命令和结果路径，安装/启用需明确运行范围和budget，按既有用户授权执行，不启无限后台实验。
- [ ] 提交 `test: verify unattended campaign recovery and deployment handoff`。执行完本任务才允许声称无人值守服务已实现；只有真实服务启用和资格通过才允许声称实验已启动。

## 计划自审与执行交接

覆盖：持久合同/锁→1；身份与exit/资源→2；故障/checkpoint/预算→3；successor/逻辑证据/训练日志→4；归档事务→5；DAG/stop/gates→6；Codex/配额/降级/动作→7；systemd/总览→8；实际故障与资格验收→9。五项Review Focus均对应明确测试。

计划阶段仅提交本文；不修改运行代码、预留新run或启用服务。当前会话执行方式保留，由主agent逐项实施；用户审阅计划后进入任务1。开发源代码变动期间不启动已预留旧Q0；保留归档引用的现有worktree。
