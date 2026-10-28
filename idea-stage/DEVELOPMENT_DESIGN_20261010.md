# 主线代码开发设计草案

日期：2026-10-10。状态：设计建议，待评审；未实现、未创建study、未启动实验。
依据：[查新结论](NOVELTY_CHECK.md)、当前 `src/semantic_retrieval/` 代码与研究记忆。

## 1. 开发目标与方案选择

要交付的最小系统：给定query、当前观测帧和固定竞争候选，输出参与排序的预测分布，并预测一次合法补帧动作导致排序变好、不变、变差的概率；在实际预算内执行动作并重新排序。

| 方案 | 内容 | 判断 |
|---|---|---|
| A：只加方差头 | 确定性检索器外接不确定性分数，用阈值补帧 | 容易开发，但不能保证概率参数参与排序，也不足以支撑查新剩余贡献 |
| B：概率匹配与收益预测分阶段开发 | 概率量参与训练和推理；固定匹配器后生成补帧转移监督；训练动作收益预测 | 推荐：可分别追查排序改善与不确定性价值 |
| C：端到端学习补帧策略 | 同时学习匹配、证据选择、预算策略 | 暂缓：策略分布变化、目标漂移及归因混在一起 |

第一版只在固定候选集合内给某个候选追加一个预定义帧块。检索候选扩容、语言澄清、强化学习、多模式概率族不属于该版本。

## 2. 已核验的代码入口

- `models/model.py:SemanticRetrievalModel`：CLIP、TemporalEncoder、CoarseProjector、PairMatcher与answerability的组合。当前代码入口不是历史 `modules/modeling.py`。
- `models/pair_matcher.py:PairMatcher`：有共享的token/frame投影，输出PairScores，其中features可供概率头使用；现有排序本身是确定性的。
- `objectives/retrieval.py:compute_objective`：已知标签限定的对比目标，unknown=-1；固定LOSS_NAMES，不宜直接把研究新增损失塞进旧pilot配置。
- `runtime/trainer.py:Trainer`：训练参数allowlist、FP32、DDP按有效样本数归一化及运行身份绑定可借鉴。
- `data/video.py:sample_frame_indices`：对各帧数分别linspace，不能保证少帧集合嵌套在多帧集合。
- `models/temporal.py:TemporalEncoder`：当前位置编码依赖输入槽位，增加帧后重新打包可能改变旧帧的位置身份。
- `retrieval/cache.py:FeatureCache`：缓存的是VideoFeatures，可能已经时序混合，不能对完整视频特征切片来模拟少帧。
- `retrieval/policy.py:RetrievalPolicy`：expand是增加候选K，不是追加帧。需独立证据动作与预算。

保留原有pilot调用链与配置语义。新增研究路径复用组件和归档机制，不复活SAP/CSD/A3，不借用旧pilot的授权。

## 3. 第一项交付：严格的观测与动作合同

定义以下独立数据类型，推理对象不含监督：

- `ObservationState`：query_id、gallery/candidate-set identity、candidate_ids、每候选已观测的原始frame_ids/timestamps、有效mask、预算账本。
- `EvidenceAction`：candidate_id、将获取的frame_ids、action_id、估计成本；仅由当前状态和公开视频长度产生。
- `EvidenceTargets`：相关性标签、rich-score目标、动作转移标签、监督来源与模型快照哈希；不得传给policy。
- `ActionOutcome`：实际成功/失败的frame_ids、实际成本、状态变化；重复帧和解码失败均明确记录。

建立单一去重的帧池及确定性优先顺序，再取前缀形成嵌套视图。例如软件样例上限16帧，层级4→8→16；短视频按可用唯一帧截断。这些数值是待评审的软件例子，不是注册研究配置。

位置使用原始时间或帧池稳定槽位；同一帧增加观测后不得换位置ID。只改变合法观测mask，且mask必须在时序注意力之前生效。不能先用全部帧运行TemporalEncoder再切片。

缓存只允许冻结CLIP的逐帧、时序混合前特征；身份至少包括视频内容、frame_id、预处理、CLIP权重与配置哈希。新缓存API必须按请求帧返回，不能把未获取特征作为模型输入。每次补帧后对已观测集合重新时序编码，新增CLIP编码与重复时序计算分别计费。

离线缓存用于反事实数据编译时，与真实decode运行分开报告成本。预览索引也只使用各方法共同允许的初始帧；密集视频索引不能隐式帮助稀疏方法。索引构建成本与在线成本分列，冷缓存和跨查询缓存复用分别记账。

## 4. 第二项交付：真正进入排序的概率匹配

建议先用标量Gaussian分数分布，避免第一版实现高维概率embedding或混合模型：

`S_j = μ(q,O_j) + σ(q,O_j) ε_j，σ=softplus(raw_scale)+ε_min`。

复用PairMatcher输出作为μ及交互特征；scale头使用同一query/video交互信息和可观测的帧位置/覆盖信息。初版按候选分数条件独立近似，不能宣称识别了完整联合后验；共享query引起的候选相关性是该近似的限制。

对候选集合进行联合采样并计算：

`p_j = mean_m softmax(S^(m)/τ)_j`。

训练在已知相关性候选上进行，正例目标采用显式声明的多正例规则，例如对已知正例的 `−log p_j` 取平均。推理对完整当前候选集合计算同一MC预测，按 `log p_j` 排序。必须实现 `−log(mean(probability))`，不能无意写成 `mean(−log(probability))`；不直接调用现有接受单一logits的contrastive_rows来冒充MC目标。

该p是候选间的排序偏好分布，不自动等于每个视频真实相关的边际概率，特别是在没有正例的候选集合中。unknown标签不作为负例；没有已知正负对照的行跳过对比项并记录有效计数。

第一版固定温度，检查scale与温度的相互补偿；σ通过采样softmax同时影响loss和实际排序。固定噪声诊断下改变σ应能改变预测，不能只保存σ而推理退回μ。

### scale的可观测监督

仅靠对比loss无法保证σ有“观测不足”的意义。推荐增加一个明确可测的分数预测任务：

1. 先训练或绑定一个固定的确定性参考匹配器，固定其权重、分数尺度与时间编码规则。
2. 指定rich-view生成协议；目标T为参考匹配器在该更丰富观测下的分数。
3. 少帧模型预测 `T | q,O,rich-view protocol`，用Gaussian NLL拟合：`log σ + (T−μ)²/(2σ²)`（忽略常数）。
4. 将该项与概率对比目标组合，并分别记录两个目标对μ/σ的梯度与影响。

这给σ一个可检验对象，但预测的是固定参考分数的条件离散性，**不是真实语义后验或天然的检索失败概率**。参考匹配器不是语义真值；它的预测和独立相关性标签分别评估。不要用正在更新的同一模型生成漂移的rich目标，也不要把任意增强噪声称为缺失证据不确定性。

该辅助任务是本设计新增的具体化建议，尚待验证，不能回填为原提案已经成立的贡献。若它与相关性对比目标冲突，应如实报告；NLL和σ下界需要数值检查，不能通过大权重强迫漂亮方差。

## 5. 第三项交付：补帧反事实转移数据

概率匹配器训练完成后固定模型快照，再在train来源组件上枚举合法的当前状态与补帧动作；动作标签对应的before/after始终由同一个固定快照产生，不能跨模型版本混用。

对固定候选集合、明确标签口径，定义：
`e(O)=1−1[当前top1相关]`，
`g(a)=e(O)−e(O⊕a) ∈ {−1,0,+1}`。

- +1：补帧纠错；0：正确性不变；−1：原本正确、补帧后变错。
- 同时保存原始rank变化以作分析；第一版收益目标仅为top1风险变化，不能称已优化所有排名指标。
- before/after需要独立相关性标签；若top1关系unknown，转移标签为unknown，不擅自写0。
- 标注benchmark只能按benchmark_annotation口径报告；不能把官方未配对项自动称为经人工审核的语义负例。
- 不为了制造纠错收益强行把真值候选塞进推理shortlist。候选漏召回必须进入总体失败和可恢复上限统计。
- 数据的候选顺序、视图、动作、随机数和模型哈希全部绑定；同一个转移的MC噪声使用约定的共同随机数或足够精度，避免把MC波动当纠错。
- 按来源组件拆分训练、开发、校准与锁定验证，不能让同视频的不同描述或帧视图跨集合。
- 离线枚举所有合法动作而非只记录当前policy挑中的动作，减少动作标签的选择偏差。实际枚举范围和覆盖率明确记录。

推理policy看不到before/after标签、完整视图目标、未来帧特征或“哪个候选正确”。

## 6. 第四项交付：收益预测与预算动作

第一版收益头采用三分类：
`P(g=−1), P(g=0), P(g=+1)`，
`expected_gain=P(g=+1)−P(g=−1)`。

输入：当前匹配交互特征、候选集score geometry、概率匹配的μ/σ与MC统计、当前帧覆盖状态、动作时间/成本描述。新增帧内容在获取之前不可见，第一版只选择“给哪个候选补下一个预定义帧块”；不声称已知道具体语义证据在哪一帧。

用转移标签的categorical NLL训练收益头。三分类使恶化收益可以表达，避免只学“纠错概率”而忽略伤害。entropy是该转移预测的不确定性，不命名为参数epistemic。

收益头训练时冻结匹配器。这样匹配器、目标和策略版本一致；后续更换匹配器需重新编译转移并绑定新身份。第一版不联合回传，避免policy目标随匹配器漂移。

策略在预算内选择 `expected_gain / estimated_cost` 最大的合法动作；最大预测净收益未超过经validation确定的停止阈值、无合法动作、预算不足时停止，保留当前排序。执行后按实际解码结果更新状态并重算概率匹配。严格区分 estimated_cost 与 actual_cost，不超额执行；按预算预留动作上界并结算真实成本。

必须有同输入、同监督、同容量/成本的确定性收益回归对照，以及去掉μ/σ之外概率描述子的匹配对照，判断收益是否只是另一套确定性选帧系统。

## 7. 拟议模块边界

下表新增文件尚不存在，不代表已经编写。

| 路径（相对src/semantic_retrieval） | 职责 |
|---|---|
| `evidence/types.py` | 观测、动作、预算、预测和监督的严格类型 |
| `evidence/views.py` | 帧池、嵌套视图、稳定时间身份、合法动作生成 |
| `evidence/frame_cache.py` | 时序混合前逐帧缓存与按观测读取 |
| `evidence/matcher.py` | Gaussian分数、MC预测、复用PairMatcher |
| `evidence/objectives.py` | 概率对比、rich-score NLL、转移分类loss及有效计数 |
| `evidence/transitions.py` | 固定快照下编译补帧转移与监督身份 |
| `evidence/value.py` | 三分类收益头与确定性收益对照 |
| `evidence/policy.py` | 获取动作、停止和预算账本 |
| `evidence/evaluation.py` | 分布评分、纠错/恶化、预算曲线、候选漏召回 |
| `evidence/config.py`、`evidence/runtime.py` | 新路径显式配置与分阶段trainer，绑定现有归档/恢复组件 |

接口关系：
`views.propose_actions(state) → actions`；
`matcher.predict(observed_features, pair_indices, offsets, rng) → PairPredictiveScores`；
`value.predict(state_features, actions, predictive_scores) → ActionValueDistribution`；
`policy.decide(state, actions, values) → acquire或stop`；
`runtime.acquire(action) → ActionOutcome`。

新增入口建议放在scripts下，分别负责参考匹配、概率匹配、转移编译、收益训练及策略评估；入口必须使用新路径配置与run身份，不接受自由输出目录绕过归档。代码共享集中在组件，不把本研究伪装成旧pilot的A0–A4变体。

TemporalEncoder需要小范围扩展显式position_ids/timestamps的接口；默认旧调用仍保持原行为。旧ModelConfig/LOSS_NAMES/TrainConfig的严格校验保持语义，新配置显式定义新增字段，不通过放宽未知字段校验兼容。

## 8. 开发顺序与软件验收

| 顺序 | 可独立验收的交付物 | 关键CPU合成测试 |
|---|---|---|
| D1 | 观测合同、稳定位置和合法动作 | 嵌套、去重、短视频、失败解码；未获取帧变化不改变当前输出 |
| D2 | Gaussian匹配、MC概率对比、可观测分数目标 | μ/σ有限梯度；σ影响真实推理；mask、unknown、多正例；固定噪声重放 |
| D3 | 固定快照的转移编译器 | 构造纠错/不变/恶化三例；标签隔离；模型哈希不符拒绝 |
| D4 | 概率收益头和确定性对照 | 同输入维度、相同监督；类别概率归一化；负收益不被截断 |
| D5 | acquire/stop闭环 | 不重复获取、不预算超支；拒绝非法动作；连续运行与断点恢复一致 |
| D6 | DDP及归档接线 | 有效计数全局归一化、各rank同步、采样器/RNG/观测/预算/进度恢复 |

synthetic闭环通过只代表软件链路成立，不证明论文创新或真实数据收益。不要以“输出shape正确”替代梯度、信息隔离和恢复验证。

关键基线：固定帧预算、随机同预算补帧、margin/entropy触发、确定性收益预测、固定方差、可学习温度、标准概率匹配；每项必须匹配信息和预算。开发时实现接口和对照能力，科学比较另按研究gate办理。

## 9. 评价对象

- 分数预测：针对固定rich-score目标的NLL、区间覆盖与宽度；不把其校准等同真实相关概率校准。
- 动作预测：三类NLL/Brier、纠错与恶化概率校准、预测收益与实际收益。
- 系统结果：排序、纠错与恶化数量、相同成本下的效能曲线，含candidate shortlist漏召回和无收益动作比例。
- 归因：概率预测对margin/entropy与确定性收益预测的独立增量；teacher分数预测、真实相关性和检索改进分别报告。
- 统计：按来源组件估计不确定区间，不把重复caption或多个MC样本当独立样本。

## 10. 开发与科学运行的边界

本次可完成的是设计说明。开发可先用CPU合成fixture，保留现有精确checkpoint、FP32与受控输入合同；不放宽strict load_state_dict。

若采用本方向，科学阶段另建独立study和protocol_version，不修改已停止的contextual-risk-v2或既有pilot。所有科学输出经唯一run预留，绑定干净源码SHA、命令、配置、输入哈希；数值数组外置。真实训练、完整视图GPU目标编译和策略评估也属于科学计算，不能改名为预处理绕过资格与研究gate。

已核对的semantic-answerability-pilot-v1仍为scientific_authorized=false，四卡资格另有未终结预留；双卡软件预检不替代新链路四卡资格。现有授权不自动覆盖本主线。test继续sealed；调参与校准使用按来源隔离的validation子集。

## 11. 设计自检与未被证明的假设

- 概率分数与动作收益定义不同，没有直接用高σ触发补帧。
- σ参与对比训练和推理；另有清楚但有限的可观测监督对象。
- 候选内容相关性未建联合协方差；第一版因子化Gaussian只是工作近似。
- 参考rich分数可能无法改善语义检索；这是需要检验的假设，不以教师一致性证明成功。
- 动作收益快照固定，监督不进入推理；无未来特征、无标签选候选。
- 预算包含扫描/解码/新增编码/时序重算/概率采样，不只数最终保留帧。
- 第一版只选择候选与预定义帧块，具体语义位置定位未包括。
- 没有复活旧study、授予GPU运行或宣称已完成实现。

