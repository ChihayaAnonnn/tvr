# Evidence-conditioned retrieval：本轮研究证据（2026-10-10）

软件功能已实现并验证；本轮冻结 pilot 未达到 B2 可检验性门槛，停止 B3/B4 和扩展。C1（不确定性描述子的独立贡献）和 C2（完整收益分布的必要性）均未得到科学证实。这是本配置证据不足，不是对该研究方向的普遍否定。

## 数据与研究范围

MSRVTT trusted-v1 validation；T2V；DSL=false；seed713；每角色64个独立来源组件、每组件一个视频/查询、各自64视频gallery、固定初始CLIP top8。不是官方1k test benchmark。保留validation500，从训练8500中排除与其同原始YouTube ID或同文件SHA-256传递闭包的254个视频，剩8246；未知跨URL视觉重复未排除。D0实际导出/校验五角色共320视频，完整解码57400帧。V_locked 的清单/标签参与输入完整性校验，但未用于模型评价或选择；MSRVTT/MSVD test始终sealed。

## B1：全部五个matcher

| matcher | 初始正确/64 | 最终正确/64 | 纠错 | 变坏 | rich-score proxy NLL |
|---|---:|---:|---:|---:|---:|
| deterministic | 16 | 14 | 1 | 3 | undefined |
| fixed-scale | 18 | 18 | 1 | 1 | 0.9322144571512565 |
| learned-temperature | 16 | 14 | 1 | 3 | undefined |
| standard-probabilistic | 16 | 15 | 1 | 2 | -0.28139390562901767 |
| gaussian | 16 | 17 | 2 | 1 | -0.578366819947636 |

Gaussian相对deterministic的最终正确率差为+4.6875个百分点，配对来源组件bootstrap 2000次、95% CI=[−1.5625,+12.5]个百分点，区间跨0；fixed-scale点估计18/64高于Gaussian17/64。所有臂如实保留，后续Gaussian/deterministic按预注册指定，不按表现更换。rich-score NLL使用冻结教师的标量分数作代理目标，不代表语义不确定性真值。

每臂均frame_requests=4096、clip_frames=4096、temporal_tokens=27648；MC elements为deterministic/learned-temperature各4608，其余各36864。相同预算上限不等于相同实际计算。初始索引的256请求/256 CLIP另行记账。五worker并发共享CPU/I/O，不能把并发耗时当作独占延迟排名。

## B2：全部六个冻结转移生产者

| matcher | role | 完整组件 | 漏召回组件 | 转移数 | corrected组件 | worsened组件 |
|---|---|---:|---:|---:|---:|---:|
| gaussian | T_value | 64 | 6 | 1526 | 4 | 1 |
| gaussian | V_development | 64 | 4 | 1526 | 4 | 4 |
| gaussian | V_calibration | 64 | 5 | 1526 | 1 | 0 |
| deterministic | T_value | 64 | 6 | 1526 | 2 | 2 |
| deterministic | V_development | 64 | 4 | 1526 | 2 | 3 |
| deterministic | V_calibration | 64 | 5 | 1526 | 0 | 2 |

Gaussian V_development corrected=4、worsened=4个不同来源组件，预注册门槛为至少8/4，passed=false。全部六臂归档后才执行门槛判定；完整覆盖、unknown、零合法动作组件、实际候选和producer/input/数组哈希均核验。逐组件明细见B2-component-coverage.md。

B3价值学习、科学校准、dev阈值选择和B4八种锁定策略对照均未运行；不生成其指标，不声称完成这些科学比较。价值头/校准/预算执行的软件实现及实际模块通路已通过CPU和四CUDA rank软件验证。未调K、帧数、MC、matcher、抽样或门槛追逐过关。

## 已实现架构与代码

冻结CLIP B/32 → 初始4帧粗检索固定K=8 → query独立观测与时序编码 → pair loc/scale → 身份稳定MC概率排序 → 当前状态/动作特征 → 三分类收益头（或geometry/mean对照）→ 期望收益/成本决策 → 补帧/停止，动作和预算持久化。冻结rich-view教师只用于离线分数监督/诊断，不进入在线观测。匹配训练和价值学习分阶段冻结，不联合回传。

| 功能 | 代码（src/semantic_retrieval/evidence/） |
|---|---|
| 观测隔离/预算/失败停止 | types.py, views.py, encoding.py, frame_cache.py, execution.py |
| Gaussian匹配/对比学习/稳定MC | model.py, matcher.py, sampling.py, objectives.py, training.py |
| 来源与转移 | source_audit.py, source_preparation.py, data.py, transitions.py |
| 收益/校准/策略 | value.py, calibration.py, policy.py, baselines.py |
| 恢复与数值评估 | checkpoint.py, journal.py, recovery.py, evaluation.py, timing.py |
| 严格配置/保护入口/四卡 | config.py, schemas.py, runtime.py, authorization.py, cli.py, qualification.py |

## 实际软件验证

最终产品提交1d16e57；真实worktree最窄测试9 passed（5.42s），受影响回归690 passed、0 failed（306.44s），包括实际OpenCV视频、冻结CLIP、编译转移、query隔离、预算失败路径、Gloo/DDP与断点恢复；完整命令/日志SHA见gain-mse-integration-review.json。临时副本曾有两项依赖Git元数据的环境失败，真实worktree回归已解决，原记录保留。没有删测试或放宽断言。

最终Q0-R5：四张A800（0/2/3/4），正式B/32容量随机权重、FP32、合成数据；13/13检查、五matcher/三head、578数组引用全部通过；controller67.146496s，峰值CUDA allocation每rank2,011,554,304 bytes。验证真实前反向/loc-scale-value梯度、unknown rank的全局归一化、各rank参数一致、checkpoint exact resume和动作恢复。真实数据训练成功或这些软件检查均不构成科学有效性证明。

独立最终代码审查final_scientific_code_review复核最终实现/测试哈希，无剩余阻塞发现，独立窄测试9 passed（5.51s）；审查记录见gain-mse-integration-review.json及Superpowers review。

## 资源与归档

全部终态run：{'completed': 25, 'failed': 6, 'cancelled': 4}。科学分配GPU wall hours=12.108477896449，软件资格=0.469454636127，总计=12.577932532576；含所有失败和未启动取消，CPU准备GPUh=0。未超科学32/软件8 GPUh，单次科学7200s/软件1800s上限。逐run的source SHA、manifest SHA、controller时间与完整终态输出清单核验见final-resource-accounting.json。

控制面：experiments/evidence-conditioned-retrieval-v1/{registry.jsonl,phases,summaries,reports}；数据面：/data2/hxj/project/UATVR/artifacts/evidence-conditioned-retrieval-v1/<dataset>/<stage>/<run-id>/。各run的manifest.json/command.sh/worker-config/输入与checkpoint/外置数组哈希、status及append-only audit保留。保留原B1五个经诊断后主动终止的软件失败（SIGTERM/143，约1282秒，未达到7200秒上限）、R1一次软件预停和四个未启动取消；修复后新身份R2五臂全部完成，未覆盖历史。

## 协议偏差和限制

1. 冻结PREREGISTRATION.md关于“无正负监督访问跳过”的字面措辞不准确：rich-score-only监督使Gaussian/fixed-scale更新121步，其余110步（均128访问）。不修改冻结历史，不宣称更新步数或监督相等，不把B1当作sigma的单独因果作用。
2. 冻结文档“V_locked unread”应限定为未用于模型评价/选择；清单/标签已由D0/runtime读取校验。没有用locked模型结果调参。
3. 外层临时行政调度脚本没有历史启动时的源码哈希。事后留存脚本只证明所审代码，不补造历史绑定；worker本身的source/config/input/command/checkpoint及输出完整性有独立核验。
4. 当前行政调度只在单phase串行工作流中核算已有终态GPUh和本批最坏预算；缺跨phase并发原子额度锁和可执行后继次数通用守卫。本轮无跨phase并发、预算超限或超额重试；不宣称已实现任意并发campaign的完备强制约束。
5. 官方单正例annotation不等于完整语义相关性；来源定义不能排除未知跨URL视觉重复。只有一个小规模seed，有限MC，低统计功效；不宣称创新性、泛化或不确定性必要性被证明。
6. 恢复测试覆盖明确的checkpoint/动作窗口，不能保证所有OS崩溃情形；不可证明的未完成执行拒绝恢复。B1历史offline诊断计时范围较新版本窄，不能事后补出遗漏耗时。

最终实验完整性审计与result-to-claim将以本证据快照为输入，另存EXPERIMENT_AUDIT与CLAIM_VERDICT及完整审查trace；同族独立审查仅provisional。

## 按逻辑单元的主要实现提交

```text
a804583 feat: define evidence state and strict configuration contracts
13754f4 feat: isolate nested observations and stable temporal positions
8f1f48d feat: implement probabilistic contrastive scores with stable sampling
72d672d feat: train evidence matcher against a frozen rich-view reference
9108068 feat: compile provenance-bound evidence acquisition transitions
c33f017 feat: learn calibrated acquisition outcomes and mean-value baselines
3c1f17a feat: execute budgeted evidence actions with recoverable transactions
3774bac feat: evaluate evidence utility against information-matched baselines
4e8e233 feat: restore evidence training and acquisition state exactly
f5d2fad feat: bind evidence workers to protected experiment identities
df5850b test: qualify the protected evidence retrieval pipeline on CPU
1db61d2 fix(evidence): bind semantic query lineage and active stage objectives
00f3656 fix(evidence): close reviewed lineage recovery and evaluation gaps
619e2ff fix(evidence): audit real source groups and prepare pinned pretrained input
e4db5aa feat(evidence): isolate approved source roles and freeze falsification pilot
72e99ad fix(archive): bind large phase inventories and validate exact writer bytes
8ba5a11 fix(evidence): include gallery preparation in index resource accounting
d3fbc3f fix(evidence): enforce split scopes and account failed initial observations
748be74 fix(evidence): hash recovery trees safely without repeated ancestor scans
f80918b fix(evidence): isolate query journals and bind complete transition cohorts
1d16e57 fix(evidence): report comparable gain errors and complete diagnostic costs
```

实验预留与终态提交另见git log、各phase summary和registry；最终源码/命令身份按每个manifest的完整SHA认定。
