# 双卡梯度与恢复软件预检

用户批准的独立软件验收。GPU 3、4，两个 CUDA rank，FP32/TF32 off；正式 ViT-B/32 容量随机权重和129个合成视频来自已完成的 pilot Q0P。不读取 benchmark/test，不替代新版四卡 Q0，不进入论文效果比较。

固定全局16查询/每卡8，K=4 present。保留正式 PilotSampler 的全局查询表、RNG/游标/调度状态；预检独立切成两份，科学四 rank sampler合同保持不变。AdamW 复用统一 `build_optimizer`：learning_rate=1e-4、weight_decay=0.01、betas=(0.9,0.999)、eps=1e-8。

- 连续更新两步，与第一步后的真实 checkpoint 恢复并重放第二步比较；核对参数、AdamW moments/step/paramgroups、scheduler、各 rank Python/NumPy/Torch CPU/CUDA RNG、sampler 与 query顺序。CUDA 最大绝对差容差1e-6。
- A2/A3/A4 从同一 P1 matcher状态初始化，真实DDP包裹后探测梯度并执行一次联合目标 backward/AdamW update。三个臂 matching梯度必须正且有限；存在损失对matcher梯度仅A3为正，A2/A4为0；matcher/head实际更新且各rank参数一致。
- 输出 run-local `provenance/dual-resume.json`、`dual-gpu-preflight.json`、`hardware.json`、checkpoint/NPY及日志。每个rank独立检查；失败保留证据，不自动重试。

`study.yaml`冻结作用域、GPU UUID、配置/命令和Q0P父摘要，`phases/preflight.json`绑定完整输入。执行前先 `reserve-phase` 并提交整批reserved记录，从干净源码运行sealed argv。worker只到validating；关闭所有writer后再终态/finalize-phase并提交摘要。预算≤2 GPUh，控制器墙钟上限3600秒。既有四卡预留位于另一工作树，不修改它。
