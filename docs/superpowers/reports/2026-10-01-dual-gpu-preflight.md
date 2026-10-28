# 双卡梯度通路与断点恢复预检（2026-10-01）

**数值检查通过，整体运行 FAILED。** GPU 3、4、两个 CUDA rank、FP32/TF32 off，正式 ViT-B/32 容量随机权重与129个合成视频。仅软件预检，不支持方法效果声明，不替代四卡 Q0，不读取 test。

## 实际测量

统一优化器是 **AdamW**，由 `runtime/trainer.py:build_optimizer` 创建。本预检 learning_rate=1e-4、weight_decay=0.01、betas=(0.9,0.999)、eps=1e-8。

| 检查 | 实际结果 |
|---|---|
| A2 存在损失对 matcher 梯度范数 | 0 |
| A3 存在损失对 matcher 梯度范数 | 1.7612532377 |
| A4 存在损失对 matcher 梯度范数 | 0 |
| 三个臂匹配损失对 matcher 梯度范数 | 1.4525841475 |
| A3 匹配/存在梯度 cosine | 0.0190109853 |
| 三个臂 matcher/head 联合目标更新 | 均实际更新；两 rank 参数一致 |
| 连续两步 vs 第一步后恢复并重放第二步：参数最大绝对差 | 0 |
| AdamW moments/step 最大绝对差 | 0 |
| AdamW paramgroups、scheduler、sampler 与 query 顺序 | 一致 |
| 两 rank 的 Python/NumPy/Torch CPU/CUDA RNG | 恢复一致 |

固定全局16/每rank8查询、K=4 present；不充当四卡全K/present/absent调度验收。每 rank 核心测量约39.71/40.05秒，峰值allocated显存约1.93/1.91 GiB，后者不含CUDA驱动和allocator reserved。以上为原始run软件测量，整体操作状态标 **CAVEATED**。

## 运行与事故

- run：`synthetic-baseline-gradient-resume-seed000-20261001T110306Z-d99b8454fce7`
- source SHA：`d99b8454fce73e7a21bb5d0ef0fc35959729ceda`
- manifest SHA-256：`d8d0529563c01809f2b6f53aba3d053ee5a3652b2d2ac4e706638d06f913957e`
- reserved 先提交 `3b1d98f`；failed/exit1 与摘要提交 `f94b553`。
- [终态摘要](../../../experiments/dual-gpu-preflight-v1/summaries/preflight.json)。105个输出文件当前哈希与终态库存完全一致；checkpoint模型/配置/NPY哈希、step1、sampler iteration1和两组rank RNG状态已独立校验。

临时监控脚本在 `Popen` 后错误地读取 verifier 返回值中不存在的 `command_sha256`，控制器实际退出1。原 worker 未被重启，完成了全部检查并到 validating，随后原进程消失，启动器退出码未被捕获。因此 **launcher_exit_code=null**，不能把独立产物验证的0冒充启动器退出0。原始零字节 `controller-launch.json`、两个监控异常、报告、日志、checkpoint与独立校验记录均保留；无静默重试或覆盖。

## 修复与复验提案

新增正式 `scripts/run_dual_gpu_preflight.py` 控制器：spawn前从manifest绑定命令哈希；spawn后记录PID的操作包含在try/finally内；异常时仅终止并wait自己启动的进程组，保留真实退出码；仅允许新reserved run，拒绝原地启动failed/cancelled。真实子进程退出0/7及spawn后元数据异常均经RED→GREEN验证。

若用户审核后授权复验：使用同一合成输入、配置和GPU3/4，在后续phase预留新的唯一run ID；`parent_run_ids`引用本failed run及Q0P，`selection_lineage`引用本受跟踪报告/commit并记录已审核的控制器修复理由。当前未创建或预留重试run；既有四卡Q0保持原状态。

软件回归：原实现全仓库547项通过；最终范围守卫41项通过。控制器修复与双卡/范围守卫最终针对性验证44项通过（9.25秒），Ruff与diff检查通过；独立审阅3项控制器回归及超时清理检查通过，无剩余Important问题。
