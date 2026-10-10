# Evidence-conditioned retrieval

本路径为冻结 CLIP → 当前观测的时序编码 → 查询–候选标量 Gaussian 匹配 → MC 排名概率 → 三类动作收益 → 预算内帧获取。Gaussian 分布描述参考分数，不能直接解释为真实语义不确定性；软件验证不证明科学有效性。

## 配置及入口

配置版本 `schema_version: 1`、归档 marker `evidence-retrieval-v1`，严格拒绝未知字段。五个来源互斥的角色为 `T_match`、`T_value`、`V_development`、`V_calibration`、`V_locked`；查询来源、字幕组、候选来源、内容哈希及连通分量不得跨角色。所有消费的视频、权重、split、初始索引规范、父模型及转移索引必须在 manifest 中绑定哈希。

- `scripts/train_evidence_retrieval.py`：reference、matcher、value。
- `scripts/compile_evidence_transitions.py`：冻结 matcher 下的全部预算可行动作。
- `scripts/evaluate_evidence_retrieval.py`：calibration、evaluation。
- `scripts/qualify_evidence_retrieval.py`：tiny CPU 合成闭环或正式容量四卡 GPU 合成资格。

入口只接受 `--config` 以及可选且必须与 config 相等的 `--run-id`、`--manifest-sha256`、`--source-sha`。运行身份不可省略；config 中的 runtime 必须对应已提交的整批预留。CLI 本身不负责提交终态：worker 成功进入 validating 后，controller verify-run、transition completed、finalize-phase，再提交终态 registry 和摘要。

初始索引规范包含 `schema_version: evidence-initial-index-v1`、clip_sha256、preprocessing_sha256、view_sha256、gallery_version、roles；每个角色声明排序后的完整可用 gallery canonical ID 列表。索引只使用冻结 CLIP 的初始稀疏观测，候选选择后才绑定标签；不会补入正例。未标注语义候选保持 unknown，官方 benchmark 标签只用于 benchmark_annotation 范围。

`model.variant` 显式选择 gaussian、deterministic、fixed_scale、learned_temperature 或 standard_probabilistic。`value.kind` 为 distribution、geometry 或 mean_regressor；`policy.baseline` 为 distribution、mean_regressor、fixed_frames、random、margin、entropy。均值头使用相同输入及 MSE，不构造三类概率、不调用温度校准。GeometryValue 仅遮蔽显式 log-scale，仍须通过实验判断是否足以隔离不确定性信息。启发式策略和学习策略共用真实预算。

## 成本和恢复

四类成本为帧请求（包括失败）、成功 CLIP 帧、时序重计算 token 和 MC 元素。初始观测、每次推理及动作均先检查并预留预算；缓存重用不免除获取请求成本。失败帧进入 attempted，不重复请求；视频耗尽后停止。每次推理结果外置为哈希验证的数组，以便恢复时不重复计费。

动作日志 prepare → execute → commit；推理日志 predict_prepare → predict。只有能证明完成的动作/推理才能恢复；prepare 后缺少执行证据时拒绝自动重放。日志按 query 独立恢复，恢复保留初始排名、动作 ID、最终状态及成本。checkpoint 精确绑定父链、配置、源码、manifest、world size，并保存每个 rank 的 RNG、MC 与采样进度；任何终态目录及 checkpoint 均不得覆盖。

全局损失以所有 rank 的有效权重总和归一化；局部无标签仍保留连通梯度并参与 DDP，全局无标签不推进优化器或 scheduler。新入口统一 FP32、关闭 TF32 并启用确定性算法，遇到不支持的算子明确失败。

## 资格及科学 gate

只允许独立 study `experiments/evidence-conditioned-retrieval-v1/`。旧 study 或移除 marker 均不能获得新路径权限。四卡资格必须来自本链路的正式容量、随机权重、四个不同 CUDA 设备，检查真实 CLIP、loc/scale/value 梯度、无标签 rank、参数一致、精确恢复和动作恢复；可执行源码改变后需重新资格。随机权重与科学预训练权重可不同，但模型容量必须一致，科学变体必须有资格证据。

研究阶段还需对应 study 的已提交 gate 证据；CPU 科学路径同样受 gate 约束。test 永不由本入口解封。所有阈值、K、校准及模型选择只能使用 validation，locked validation 只用于冻结后的评估。通过软件资格不能自动把科学 gate 标为通过。

CPU 端到端测试在临时已提交源码上运行真实 CLI、合成视频及 CLIP，验证教师训练与冻结、Gaussian 梯度、转移与收益训练、获取/停止、checkpoint 精确继续，以及执行后 commit 前恢复。三类标签检查为明确的 toy 语义变化，只证明标签程序。GPU 资格结果另行归档，不得将 CPU 报告冒充四卡资格。

初始索引成本报告 `evidence-initial-index-report-v2` 从 gallery 准备前开始计时，
包含稀疏帧解码、CLIP 编码、索引组装、query 搜索和 embedding 发布；输入校验及
报告自身写入不计入。`resource_counts` 仅记录当前 worker 实际新执行的请求和
编码，失败请求计数，恢复缓存不重复计数。该离线成本与逐查询在线预算分开报告。
历史 v1 只计准备后的组装/搜索时间，不可与 v2 的总索引时间直接比较。

科学权限必须显式声明 `science.stage_eval_splits`：reference/matcher/value 仅 train，
transitions 仅 train/val_development/val_calibration，calibration 仅 val_calibration，
evaluation 可按已授权阶段声明 val_development/val_locked。缺失或越界声明拒绝预留与启动；
val_locked 还必须有独立、已提交且哈希匹配的 locked_validation gate。

初始 episode 或 gallery 遇到完全不可用的视频时，结算所有已发生的请求和真实 CLIP 编码，
发布版本化 `evidence-input-failure-v1` 及受影响查询集合的数量和哈希，然后使整个 run 失败。
不会产生默认预测、过滤该查询或报告部分总体指标；未完成运行不能支持完整分母的结论。
