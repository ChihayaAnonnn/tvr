# 来源隔离的小规模证伪协议 v1

冻结日期：2026-10-10。依据已批准设计、Native 实施计划及用户来源修订批准。此协议只为本 study 打开有证据支持的逐阶段 gate；MSRVTT/MSVD test 始终 sealed。软件成功与科学成立分别判定。

## 命题与反命题

C1：在相同候选、初始观测和四项资源约束下，匹配分布的额外描述子能改善真实补帧纠错预测与分配。关键对照为同一 Gaussian matcher 上的 GeometryValue，以及确定性 matcher 的同流程 DistributionValue。GeometryValue 去除显式 log-scale，保留当前排序的 geometry；它不代表删除全部由概率排序产生的信息。

C2：三分类收益输出相对相同完整输入的 MeanRegressor 有独立用途。proper scoring 改善与决策改善分别报告；当前策略只使用均值，不能由分类校准更好推断分布决策必要。

需排除的解释：额外监督、候选插入答案、训练来源重叠、额外计算预算或仅匹配正则收益。没有 LLM/VLM 动作原语；不添加额外模块。

## 固定数据与输入

原 trusted-v1 validation 500 不变。按相同原始 YouTube ID 或相同视频 SHA-256 的传递闭包分组；训练 8500 中排除与 validation 同组的 254，剩 8246。未知跨 URL 视觉重复不声称已排除。

用 seed=0 对来源组件哈希排序：训练按 80%/20% 分 T_match/T_value；validation 按 60%/20%/20% 分 V_development/V_calibration/V_locked。边界按整数向下取整，全部组件成员同角色。原始集合不覆盖。

首轮每角色取 seed=713 哈希最小的 64 个组件，每组件取哈希最小的一个视频，每视频取 query_id 哈希最小的一条 caption。共 320 视频/查询；兄弟片段保留原角色且不在本轮使用。选择仅依赖身份，不依赖内容语义、标签结果或模型预测。五角色各自完整 gallery=64；这是一项控制机制 pilot，不是 1k benchmark 成绩。

官方 benchmark_annotation 仅有一个官方正视频，不解释为完整语义真值。CPU allowlist-first 导出只解码 9k 训练 ID 的 caption；正式 worker 只读导出的 320 条及独立标签表。D0 必须完整解码并绑定所选视频、组件、关系表和清单。W0 官方 CLIP ViT-B/32 SHA=40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af。

## 固定科学配置

冻结 CLIP text/frame，FP32，时序 dim512/1层/8头，pair coarse_dim256，scale_floor=.01。帧池最多16，嵌套4/8/16；候选 K=8，使用冻结 CLIP 初始4帧索引，按 canonical ID 打破并列，不插入正例。M_train=M_eval=8，温度1，identity-normal-v1，seed713。报告有限 MC 近似，不把数值噪声归因于证据。

参考和五种 matcher 均为128次查询访问、AdamW lr1e-4 wd.01、constant scheduler，每64次存档；无法形成正负监督的访问跳过并报告真实 optimizer steps。相同访问序列与合法候选，所有 matcher 从同一固定 reference 开始。reference 为 deterministic、pc=1/score=0；gaussian 与 fixed_scale 为 pc=1/score=1；deterministic、learned_temperature、standard_probabilistic 为 pc=1/score=0。上述损失差异是显式方法定义，不能解释为单纯 sigma 因果效应。五种：deterministic/fixed_scale/learned_temperature/standard_probabilistic/gaussian 全部归档。

每查询上限：frame_requests=64，clip_frames=64，temporal_tokens=4096，MC elements=8192；成本权重=(1,1,1/16,1/128)。记录真实四项计数、初始索引构建/反事实编译/在线耗时及峰值显存，成本代理不冒充延迟等价。

匹配器冻结后，T_value、V_development、V_calibration 包含初始状态和至多2步固定随机轨迹，对每个状态保留全部合法、预算可行的单步动作。固定 P_ref、模型/候选/标签/采样/视图哈希；不根据预测收益筛选训练动作。Gaussian 和 deterministic 都编译，用于后续同流程控制。

收益头 hidden128：DistributionValue 和 GeometryValue 无类权重 categorical NLL；MeanRegressor 相同完整输入回归 g，MSE。全部 T_value 转移，每次更新使用全有效转移，256更新，lr1e-4 wd.01。不存在 outcome-driven 类采样。分类头仅 V_calibration 拟合单一温度；MeanRegressor 无温度。策略阈值只在 V_development 的 {0,.001,.005,.02} 选择：最大最终 top1，然后最少 frame_requests，再最小阈值。K/模型训练预算不搜索。

## 按证据推进与停止

B0（必须）：D0 真实五角色验证、W0 输入验证、最终 executable source 的新增链路四 CUDA rank Q0。失配或不完整即停止依赖工作。

B1（必须）：参考与五 matcher；V_development 固定初始观测及固定预算 fixed_frames 评价，归档全部结果。冻结 Gaussian 与 deterministic，任何 arm 表现较好均不替换预先指定主方法。

B2（必须，尽早证伪）：编译上述两个 matcher 的三角色转移，逐组件报告 corrected/unchanged/worsened、候选漏召回和覆盖。在 Gaussian 的 V_development 转移中，至少8个不同来源组件出现 corrected、至少4个出现 worsened，才进入收益学习。这是可检验性/标签变化 gate；未达标记为本 pilot 不充分或机制空间不足，不声称主线已被普遍证伪。不调 K、帧数、MC、matcher 或数据抽样来追逐过关；停止 B3/B4 和扩展，保留未运行项及原因，V_locked 保持未读。

B3（B2通过后必须）：Gaussian 上3种收益头与 deterministic 上 DistributionValue；V_calibration 独立校准，V_development 选择停止阈值。完整分布相对 geometry 的 transition NLL/Brier、修正/变坏概率及收益 MSE；MeanRegressor 比较收益 MSE与实际策略，绝不伪造分类 NLL。

B4（B3完毕后一次性锁定 validation）：冻结已提交配置，在 V_locked 比较 fixed_frames/random/margin/entropy/MeanRegressor/DistributionValue/GeometryValue/DeterministicMatcherValue。相同候选/观测和资源上限；按实际消耗报告帕累托/匹配预算，不能仅凭相同上限声称实际成本相同。候选漏召回、解码失败包含总体分母，unknown 单列。

统计以来源组件为单位，配对 bootstrap 2000次，95%区间；首轮 seed713 是证伪 pilot，任何一次正向点估计不支持稳定有效性。C1 在本轮至少需要 distribution 对 geometry 的 NLL 差值区间低于0，且 locked 策略准确率增量区间高于0并未增加四项平均成本；否则不扩展。C2 若 MeanRegressor 达到同等策略效果，则收窄至概率预测，不声称完整收益分布决策必要。这里 N=64 的功效有限，区间跨0归为不充分证据而非全局无效。

仅上述正向 gate 支持扩展为 seeds713/714/715 与更大训练组件；扩展必须先追加冻结协议和实际资源预算，不能看 locked 结果后反复调参。首轮负/不充分即终止本 campaign 的扩展，报告负结果及剩余条件。

## 预算、恢复和产物

CPU D0/W0 不占 GPUh。软件资格每次1800s、累计8 GPUh；科学每次7200s、累计32 GPUh，最多2个诊断清楚的软件失败后继。单 GPU reference/matcher/transition/value/evaluation；允许不同 run 在空闲卡并发，不干扰无关进程。达到预算、身份不一致、数值非有限或重试无新信息即停止。

依赖顺序：D0/W0→新Q0→S1 reference→S2五matcher→B1评价→S3转移→B2 gate→S4收益→S5校准与dev选择→B4锁定validation。必须先提交每批 reserved；source/config/input/manifest/parent/checkpoint/action state 全绑定；worker期间无tracked修改；终态verify、summary和批提交。修复源代码需新SHA及重新四卡资格，不覆盖旧run。

控制面遵守仓库路径，覆盖 experiment-plan 技能默认 refine-logs 输出：本 PREREGISTRATION.md、EXPERIMENT_TRACKER.md、phases/、summaries/、reports/；所有运行数据只在 artifacts/evidence-conditioned-retrieval-v1/。不得使用旧 study 授权。最终独立审查及结果完整性核验后更新研究证据；未完成阶段明示原因。
