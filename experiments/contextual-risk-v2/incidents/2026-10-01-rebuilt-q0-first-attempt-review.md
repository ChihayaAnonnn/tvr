# 四卡代码验收首次尝试：未通过，待人工处理归档

run：`synthetic-baseline-rebuilt-prod-e2e-seed713-20261001T064151Z-4426f97b5a7f`。
source：`4426f97b5a7f1446ae7ee541544fd89385eb3dfe`。
manifest：`36c362b687c86fadf00d08e7e1b2a7a040cc56db250f5b28437271d76f39e806`。

物理卡 0、2、3、4，均为 A800 80GB；CUDA UUID、四 rank NCCL、FP32 与正式 ViT-B/32 容量经过入口检查。所有科学 test 仍封存。输入均来自已完成、已提交终态记录的独立合成 prep run。worker 实际退出码为 1，全部进程已退出，没有进行第二次 GPU 尝试。

## 验收失败

日志在 `gpu_qualification.py` 的 `bounded expansion failed` 断言处结束。按原始 source 的执行顺序，此前训练、known 计数、梯度/跨 rank 参数、unknown 跳步、checkpoint/RNG/优化器恢复和损坏副本拒绝检查已执行；但未发布完整 qualification report，不能据此宣称 Stage 0 PASS，也不能补填未经记录的参数差值。

校准概率确实出现 FP32 饱和：保留的 pair 数组有 1 个概率恰为 1，presence 数组有 1 个概率恰为 1。两组 NPY 的逻辑数组哈希已校验。阈值为 1 不能保证扩检；策略可以合法直接 return。这个错误位于验收控制条件，并非已有证据显示产品检索策略出错。

pair 数组：`arrays/sha256/be/be806bdf38eceed8ad25bf31b5f8d98b1dc91f693024b16242d2309f7814ba02.npy`。
presence 数组：`arrays/sha256/e7/e7b338a97db39ca13048c0a189a1aec1497835b074e06d745fc2707d04cf71cc.npy`。

修复：扩检/零预算的分支检查使用单独、显式记录的控制 calibration，其有限 FP32 logits 对应概率始终低于阈值。实际拟合 calibration 仍保存，并用于 locked 评估。新增概率饱和回归测试，先复现同一断言失败，再通过。另在部署前发布训练检查的独立证据文件，并保留校准数值数组引用，避免后续失败丢失阶段证据。

## 归档事故

执行者错误地给 `running → validating` 传入 `exit_code=1`。现有 transition writer 写入 status 和已提交 audit，reader 随后按非终态必须为 null 的约束拒绝读取。因此当前状态是 **无效的 validating**，不是有效 failed 终态；registry 仅有原 reserved 记录，不能执行 finalize-phase。

`verify-run` 明确失败：`non-terminal run status requires a null exit_code`。这是已提交 transition 的元数据错误，不是单个待提交 transition 的可恢复窗口。没有修改或截断原 status/audit，也没有伪造 failed/completed 记录。源码、manifest 与 reserved 记录的验证通过，127 个原文件的独立字节哈希保存在 [forensic inventory](2026-10-01-rebuilt-q0-first-attempt-forensic.json)。

原 status SHA-256：`d3ca0b4cd574574f843168669df8a86f9a80d61c31827e3863bade1e430497bc`。
原 audit SHA-256：`9d3620f981c7f93ea01e365a97093bc1fa8afa320d0772867b0172c33dbfca30`。

修复 writer：非终态携带任何非 null exit_code 时，在写 status 或 audit 前拒绝；新增两种过渡的回归检查，保证拒绝后两份文件的 bytes 完全不变。修复不自动改写这次事故的历史记录。

修复验证：窄测 4 项通过（饱和/非饱和链路、两种非终态过渡）；完整命令 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q` → 324 passed，79.13 秒，1 个 PyTorch TypedStorage 弃用警告。修复仅完成 CPU 验证，未进行第二次 GPU run。

独立修复审阅：无 Critical/Important 问题；另独立执行 4 项窄测通过，全部 127 个归档字节哈希匹配。仅有一项已记录、暂缓的 Minor：饱和回归目前未直接断言 locked 指标来自实际拟合 calibration；当前实现明确重建 fitted service，行为正确，可后续补充该断言。

## 人工审查与后续授权

本报告仅提供事实和修复，不代表人工批准。按 AGENTS.md，本事故不满足自动恢复条件，必须先人工处理归档异常；任何后续 GPU 验收须得到人工审查与授权，在后续 phase 使用全新 run ID，并以 `parent_run_ids` 和引用受跟踪报告的 `selection_lineage` 保留本次尝试的关系。不得覆盖、原地重启或静默忽略此 run，不能把其状态标记为 PASS。

待审查事项：确认按本报告处理归档异常、封存原证据，并授权修复后的新 run 四卡复验。正式 benchmark 适配器、minibatch 科学训练及 test 解封仍未授权。
