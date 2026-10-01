# B 阶段实现验证

范围：新 semantic_retrieval 系统、合成 CPU 验证。不启动科学实验，不触碰 sealed test。

## B1

原始 episode manifest 显式绑定视频路径、内容 SHA-256、帧数、预处理容量及审核来源。未知标签为 -1，canonical alias 冲突拒绝；候选存在与全库存在分开审核。加载及 collate 均检查内容身份，推理只接收 QueryInputs。

验证：首次缺失模块测试 13 failures；实现后 13 passed。跨 split 的 canonical、source、caption 泄漏分别独立验证。A 基础回归 72 passed。

审核 provenance 是数据提供方的声明；本 loader 不代替人工标注审查。

## B2–B7

显式 CLIP 本地权重 + 时序 Transformer + 独立粗投影组成新模型。requested pair 共享投影同时供匹配、dustbin attention 和候选存在头使用。分别用 pair/existence 实际 optimizer step 验证排序输出改变。索引预先 canonical 去重，缓存存完整帧/pooled/mask，版本和引用哈希不符即拒绝。策略的 K 严格递增且有预算上限。

Trainer 绑定完整 manifest 配置及输入哈希；生产 AdamW 与显式 scheduler；全 unknown 跳过 step。新 checkpoint 保存模型、optimizer、scheduler、逐 rank RNG，二进制 SHA 及配置身份严格校验；超过 1 MiB 的数值张量外置 NPY 引用并校验引用哈希。单进程恢复后的下一步与原训练完全一致。双 rank Gloo 将未审核跨 rank pair 保持 unknown，各目标按全局有效计数归一化；以生产 AdamW 路径的完整梯度核对单进程全局参考，避免用接近零梯度的 Adam 参数差作为归一化证据。该验证不代替 GPU 四卡资格。

已验证：B2 4、B3 4、B4 5、B5 2、B6 6 个测试；B7 与 A runtime 回归 17 passed。首次双 rank 测试定位 DDP forward 外粗投影，已纳入统一 forward 修复。所有数值仅合成 CPU fixture。

## B8–B10 与独立审阅

基础链路包含冻结模型/索引/校准的单 query 服务，仅新增候选计算 pair，完整候选集合重算存在。validation 校准排除 unknown，单类拒绝；评测明确 event 分子/分母与未完整标注计数。语义分组与 OT 已做合成算子及实际共享排序更新验证，默认关闭；这不是科学效果或生产启用结论。接口与原始数据合同见 [系统说明](../agent/semantic-retrieval-system.md)。

一次独立审阅覆盖 `f0ae616..bc7d601`，发现 6 个 Important：

| 问题 | 修复与回归 |
| --- | --- |
| CLI 传入归档 API 不接受的 split | 使用既有 publish_metrics 合同，真实 train→calibrate→evaluate main() 临时登记仓库闭环 |
| 预处理身份包含无关 source commit | 改为预处理参数与实际 video/tokenizer/BPE 实现哈希，跨独立 reservation 源提交消费校准 |
| DDP RNG world count 阻挡单进程部署 | 独立 model-only checkpoint loader；训练恢复仍严格匹配 world size |
| rank 候选数不等触发 gather shape 拒绝 | 长度收集、padding 可微 gather、去 padding；不等候选数及单 rank 空候选梯度验证 |
| 实际模型/配置/optimizer 可偏离 manifest | 每步与保存时检查有效配置、模型、参数覆盖、标准 AdamW 与 scheduler 策略；篡改拒绝 |
| 同 canonical 的 episode/gallery 内容不一致 | 校准及评测前检查重叠 ID 的内容 SHA 和 frame count，冲突拒绝 |

各修复均观察了失败回归，再通过修复后的同一测试。补充的大 checkpoint 张量外置回归验证 NPY 引用及损坏拒绝，满足 AGENTS.md 的 1 MiB 数据面规则。临时还原原版本确认跨 phase 校准原本因 identity mismatch 失败。额外修复各 rank scalar loss 报告的全局归一化；逐 rank 报告一致并与单进程全局参考比较。疑似 unknown batch 跳过 backward 会破坏 DDP 的问题没有复现，保留原跳步行为；known→unknown→known CPU Gloo 验证通过。

Deferred minor：极端概率恰为 1 且接受阈值为 1 时，SemanticSet 的 inverse-logit clamp 会改变描述性 supports/entropy；动作使用的真实校准概率不受影响。

## 执行中的裁决及代价

- Ruling: Numeric Task 1–10 headings retain B labels to match bundled task tools — mechanical tooling requirement — cost if wrong: revert title formatting.
- Ruling: Reuse the existing tested environment and create a new task-owned worktree through git fallback because no native creator is available — same approved isolated workflow as A — cost if wrong: fresh dependency installation is not validated and integration remains separate.
- Ruling: Episode raw input processing is declared in each input manifest (text length, frame budget, image size), and collate tokenizes/decodes current raw content; no inline video tensors or implicit production preprocessing defaults — plan omits the raw input wire format — cost if wrong: dataset adapters must emit this declared schema.
- Task 6: Ruling: SemanticSet.build accepts optional explicit support_threshold; entropy is undefined when omitted — spec requires accepted groups only and no unregistered threshold default — cost if wrong: consumers must supply the frozen threshold to request entropy.
- Task 7: Ruling: add explicit DataConfig to SystemConfig for episode path/hash/gallery/split — CLI needs a reserved raw-data source absent from plan fields — cost if wrong: configuration schema needs a migration; no implicit dataset defaults.
- Task 7: Ruling: checkpoint config digest excludes runtime run identity but includes model/train/policy/data — audited child-run resume needs new immutable run identity — cost if wrong: source lineage must still be enforced by manifest/controller, not this digest alone.
- Task 8: Ruling: explicit DeploymentConfig reserves checkpoint, raw gallery inventory and calibration inputs — reconstructing a full gallery from episode GT would falsify coverage and lacked a CLI wire contract — cost if wrong: adapters must emit the documented gallery schema and reserve these inputs.
- Final: Ruling: scientific effectiveness/annotation quality/GPU qualification/sealed real datasets remain outside CPU verification — approved B scope explicitly excludes them — cost if wrong: no real-data effectiveness or deployment claim is justified.
- Final: Ruling: flow/diffusion/ANN/video-to-text/conflict classifier remain deferred — approved first version excludes them — cost if wrong: a later scoped design and implementation is required.
- Final: Ruling: arbitrary caller-provided index tensors are trusted under declared identity; CLI builds them using the frozen current model — constructor has no raw inputs from which to independently prove feature provenance — cost if wrong: external callers can attach false metadata to stale or fabricated embeddings.
- Final: Ruling: suspected all-unknown DDP reducer failure was not reproduced: removing zero-backward still passes known→unknown→known test with standard find_unused_parameters=True — retain the original backward/optimizer/scheduler skip rather than extra collectives — cost if wrong: other distributed graph/topology cases still need qualification.

## 最终验收

- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q -p no:cacheprovider tests`：**302 passed，1 warning，59.16s**。唯一 warning 是 PyTorch TypedStorage 弃用提示。
- 修复后的 semantic 子集先验收 **128 passed**；随后新增的大 checkpoint 外置回归已包含在上述最终全套。
- `ruff check src/semantic_retrieval scripts/{train,evaluate}_semantic_retrieval.py tests/semantic_retrieval`：通过；`git diff --check`：通过。
- B1–B10 的计划勾选与逐任务提交完成；一次独立审阅发现的 6 个 Important 均以失败回归和最终绿色全套修复，另补全局 scalar metrics 与大张量归档。
- 仅 CPU synthetic/tmp_path 仓库；未预留或启动真实实验，未读取真实 test，未获得 GPU 资格或方法效果结论。原 master 的既有未提交文档与论文保持独立。
