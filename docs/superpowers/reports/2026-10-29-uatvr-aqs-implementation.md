# UATVR + AQS 开发交付报告

日期：2026-10-29。状态：实施计划中的 CPU 软件开发与验证已完成。没有启动科学训练、正式容量 CUDA 资格或 test 评估，不报告检索精度收益。

## 源码与工作区

- 实施基点：`976f815`，包含已经提交的设计与实施文档。
- 独立分支：`codex/uatvr-aqs-v1`。
- 工作区：`/home/xujie/.codex/worktrees/uatvr-aqs-v1/UATVR`，由 Codex 原生 worktree 工具创建并附加本 chat。
- 被验证的训练代码提交：`6a27bf3`；分布式修正提交：`16abe06`。
- 开始时主目录处于合并中；本次没有切换、重置或合并主目录分支，没有提交其暂存改动。

## 已实现

新增 `src/semantic_retrieval/aqs/`，包含严格配置、canonical ID 验证、纯帧池化、确定性贪心选择、T2V 辅助损失、UATVR core 扩展、训练适配、优化 step 与严格恢复包装。

完整 batch 的原有双向检索损失及已加权 MIL/KL 保留。辅助损失使用原缩放 logits；选择使用停止梯度的原始 WTI 分数和时序帧池化表示。没有新语义 query、检测器、温度、局部片段损失、跨 batch 队列或推理重排序。

唯一旧模型文件改动是 `full_data/models.py` 提取默认 UATVR core 类型工厂。vendor 源码/PROVENANCE、Laya 模块及其 study 没有变更。新 core 显式扩展原有方法；AST 回归去除观察和布局扩展后与原方法一致。

严格恢复沿用原 checkpoint schema、optimizer、各 rank RNG、epoch/update/batch cursor 和数组外置合同。派生 recipe 绑定真实模型结构、冻结参数集合、AQS 配置、ID 映射和扩展源文件哈希；另出版不可覆盖的 `.aqs.json` 明文配方，恢复前验证身份及内容。修改负例数、权重、映射、source、world size、recipe、batching 或配方文件均被拒绝，失败测试确认模型、optimizer 和 RNG 未改变。

`aqs_step` 验证 optimizer 正好绑定当前可训练模型，避免传入另一模型的 optimizer 形成无效训练。选中负例的辅助梯度到达文本/视觉投影、时序参数和原 logit scale；原概率分支仍有真实梯度。

## 实施中修订

### CPU Gloo 梯度布局

最初双 rank 检查中，前向 logits、WTI 最大匹配索引与负例选择一致，但视觉参数和编码帧的反向梯度不一致，文本投影梯度一致。两个 rank 的最终参数梯度彼此一致，因此不是简单的 DDP world-size 缩放问题。

本地 PyTorch 2.1.2 的 Gloo all-gather 反向经由 all-to-all/scatter 模拟 reduce-scatter；接收张量使用 `empty_like(grad)` 保留布局。辅助损失单独反传产生的非连续梯度会触发布局问题。active 组件在 gather 结果后加入 `_ContiguousBackward`，只在反向返回 `.contiguous()`；不改数值、collective 数量或缩放，不修改共享 runtime/vendor，也不使用 hook/monkey-patch。

该修正前的真实模型梯度测试失败；修正后在原定 `atol=2e-6, rtol=2e-4` 下通过。测试另核对布局节点不改梯度值、全局选择一致、局部帧梯度缩放正确，以及完整双 rank 训练/恢复。关闭 AQS 时不执行节点。

### 配方和保存事务

除 identity 哈希外保存显式 `.aqs.json`，满足 AQS 配置/provenance 可检查要求。checkpoint 与配方文件均不可覆盖；若保存中断留下部分文件，需要正式归档层按 verify-run 流程处理，不能再次用同名保存来覆盖。本次未实现新的科学 controller 或事务恢复授权。

## 最终验证

环境：已有 `tvr` 环境，Python 3.10、PyTorch `2.1.2+cu121`。所有命令显式隐藏 CUDA，输入为合成数据、真实 tiny UATVR/CLIP 结构与随机初始化权重，没有读取研究数据或 Laya 权重。

实际执行命令（退出码 0）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest tests/semantic_retrieval/aqs tests/semantic_retrieval/full_data/test_models_training.py tests/semantic_retrieval/full_data/test_baseline.py tests/semantic_retrieval/full_data/test_factors.py tests/semantic_retrieval/full_data/test_audit.py -q --tb=short
```

结果：**91 passed, 1 warning in 43.83s**。警告为既有 PyTorch TypedStorage 弃用提示。新增恢复/配方及双 rank 测试的单独回归另有 **20 passed**；最终 91 项已包含这些测试。

覆盖：

- 配置、非有限输入、padding/聚合 token 排除、唯一身份、全同难度、并列、候选不足和 B=1。
- 难度优先后多样性选择、身份置换一致、无重复负例、选择无梯度/无 RNG 消耗。
- 正例及所选负例的 logits 梯度；真实编码器参数梯度及冻结参数保持。
- disabled 与 weight=0 的原有损失、logits、梯度和 RNG 精确一致。
- 固定权重/批处理路径下原有推理等价，不增加推理 AQS 操作。
- 两 rank CPU Gloo 的全局选择、辅助梯度和同步拒绝：跨 rank 重复 ID、配置/映射不一致、单 rank 非法输入。
- 单进程及双 rank 连续训练对比新模型/optimizer 精确恢复后的继续训练；参数、optimizer、RNG 和日志一致。
- 配方字段/文件篡改在恢复前拒绝，不修改活跃状态；完整 caption 覆盖与确定性 sampler cursor。
- fresh subprocess 拦截 Laya/QueryMatch/外部 reranker 模块和文件访问，仍完成 active/disabled 训练及推理；子进程写入限于 pytest scratch。

`git diff --check` 通过。相对基点检查 vendor、Laya 源码和 Laya study 的 diff 为空。

## 交付边界

接口说明见 `docs/agent/uatvr-aqs.md`。本轮按计划单 agent 执行并自审，没有声称独立 agent 审查。

未实施新的生产 manifest/controller/worker、科学预算、GPU 拓扑或 test gate；没有向旧 baseline/Laya 入口添加绕过冻结配置的 AQS 开关。正式研究执行需要独立 study 和实际适用资格；旧 study 的双卡/test 授权不外溢。

固定输入批处理路径的推理等价不等于任意 batch 形状间等价。原有 singleton 文本池化 mask 别名限制没有在本任务中修改。软件通过也不代表 CUDA 性能、正式容量数值、语义假负例处理或检索收益已获验证。
