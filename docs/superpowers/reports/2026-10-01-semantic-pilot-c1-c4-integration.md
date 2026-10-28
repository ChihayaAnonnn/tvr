# C1–C4 集成与验收记录（2026-10-01）

C1–C4 软件实现已集成到 `codex/semantic-pilot`；真实 D1S、D1 和新增 Q0P 已成功归档。新版 Q0 四卡 worker 尚未启动，不能将软件回归或历史 B 阶段资格视为新版四卡 PASS。科学训练和 test 均未启动/解封。

## 已完成

- C1：独立 train-only 快照包含官方 9000 视频、180000 训练 caption；test caption 解码为 0。全部训练视频帧检查通过。四分区为 7000/500/500/1000，各查询恰好一个官方 canonical GT，来源组件隔离。见[真实数据验收](2026-10-01-semantic-pilot-c1-real-data.md)。
- C2：四 rank 动态全局 16 查询、K/present/absent 面板、A0/P1/A1–A4、严格 P1 分支初始化、采样/RNG/Adam/checkpoint 恢复与梯度诊断已接通。
- C3：自然/受控分别校准，受控开发 K32 选择阈值；官方/去重 Recall、RC/AURC、概率校准、配对组件 bootstrap、消融/敏感性/分层图及真实扩检与 4×5 不变性探测已接通。训练父身份、分数组和数值文件均按完成归档校验。
- C4：完整模板工厂、阶段依赖、整批预留/提交、不可变 worker 命令、终态归档、预算上限和停止规则已接通；Q0P/Q0/E0/训练/评分/C0/V0 均有生产入口。开发数据、四卡资格、梯度和稳定性门必须通过才放行锁定评分。

完整链路：`D1 → Q0P/Q0 → E0 → P1a/P1b/P2a/P2b → C0S → C0 → V0E → V0S → V0`。18 个正式训练 run 的配置由实际 E0 与 P1 成功归档生成，不提前伪造父 checkpoint 或科学候选。科学 E0 必须另绑定真实预训练 ViT-B/32 权重；Q0P 的随机权重只能用于软件资格。

## 验证与审阅

最终源码 `39527da`：全仓库 CPU `pytest -q -p no:cacheprovider` **538 passed**，157.55 秒；一个现有 TypedStorage 弃用提示。Ruff 对 pilot 运行时、分析和入口全部通过；`git diff --check` 通过。

独立审阅复现并修复了真实归档目录字段/相对摘要路径、V0 混入开发分数组、未通过开发门仍放行锁定评分等接口问题；真实提交归档边界和拒绝失败门的最终针对性回归 9 项通过，无剩余 Important finding。开发门不包含锁定后才测量的不变性，不以确认性 80% 功效要求替代 pilot 判定。

## 实际 Q0P

- run：`synthetic-baseline-pilot-q0-prep-seed000-20261001T104445Z-4ff307be570d`
- source：`4ff307be570d576d2f78e7d922bb81a9ea7bc325`
- manifest SHA-256：`9cf246f55211b82bdae1aa312092d0f32fccd06d917959743c5391654eac4652`
- 整批 reserved 先提交（`e97b2dc`），worker 完成后 terminal/摘要提交（`4e3d401`）。状态 completed，exit 0，144 个输入引用，129 个独立合成视频和正式 ViT-B/32 容量随机权重。
- [受跟踪摘要](../../../experiments/semantic-answerability-pilot-v1/summaries/Q0P-20261001T104445Z.json)及全部输出哈希已通过 `verify_phase_evidence` 复核。这是实际 CPU 软件输入准备，不是 GPU 或研究效果证据。

## 待执行

新版 Q0 固定物理卡 0、2、3、4。当前 0、2 正在执行其他任务，3、4 空闲；不启动占用中的四卡 worker。Q0 通过前拒绝科学 GPU 实验；本报告后的 Q0 预留只表示身份和命令已冻结。没有失败 run 被重试或复用。

启动与归档命令见[编排指南](../../agent/semantic-pilot-orchestration.md)，架构见[系统合同](../../agent/semantic-retrieval-system.md)，分析输入/输出见[分析指南](../../agent/semantic-pilot-analysis.md)。已注册的绝对输入路径引用本工作树，保留该工作树以维持历史及后续输入可用性。
