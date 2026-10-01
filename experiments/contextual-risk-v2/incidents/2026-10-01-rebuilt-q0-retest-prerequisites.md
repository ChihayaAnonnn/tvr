# 四卡复验前置证据

用户在本会话审阅首次事故及人工处理/新 run 授权请求后，指示“进行四卡复验”。
原失败 worker 不恢复；控制面通过同 run correction 封存，原 status/audit 和 127 个文件哈希保持不变。
人工封存与适用范围见 [manual seal](2026-10-01-rebuilt-q0-manual-seal.json)。

修复源码 `d6b55f43eb6d8eeedd1d304069c2a0edeeddee47` 为合成代码验收增加已批准 predecessor
校验；原 completed prep 的完整七个输入仍是唯一模型/数据输入，旧 checkpoint 不作为输入。
新 run 以 `parent_run_ids` 和已提交审查报告保存重试关系，torchrun 明确 `max_restarts=0`。

CPU 验证命令：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q`。
最终结果 **333 passed，80.13 秒**，1 个 PyTorch TypedStorage 弃用警告。
首次完整回归为 332 passed/1 failed：新测试将控制器锁文件误纳入内容清单；修正测试夹具后重跑全部通过。
新 guard 的窄测 9 passed；独立审阅者再次运行 9 项通过，复核原 127 个文件哈希一致。
`git diff --check` 通过；独立审阅未发现 Critical/Important/Minor，Ready to merge: Yes。

审阅未替代 GPU 验收、实际新 phase 的归档核验或科学效果评估。本次仅验证正式 ViT-B/32
容量、随机初始化、合成视频上的 FP32 四 rank 训练/恢复/部署链路；默认关闭语义分组/OT。
所有科学 test 保持 sealed，既有失败 study 不因这次复验恢复。
