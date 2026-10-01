# 四卡重建代码验收

本次授权范围是合成数据上的软件验收，沿用 Scheme A 的控制面与数据面；不恢复 contextual-risk-v2 科学研究，不授权 MSRVTT/MSVD test，不启动论文实验。选择物理卡 0、2、3、4（A800 80GB，2↔4 NVLink）；启动前重新检查占用。

1. 补齐独立资格入口与 CPU 回归测试。CPU 预备 worker 在已预留目录生成随机初始化的正式 ViT-B/32 权重、合成原始视频和 train/calibration/locked episodes。GPU manifest 绑定这些输入的全部哈希、正式模型配置和源码。
2. 四 rank NCCL 验收：原始视频解码/tokenizer→CLIP→temporal/coarse/matcher/answerability→loss/backward/optimizer。验证已知样本计数、梯度有限且非零、参数跨 rank 一致、unknown 批次不更新；checkpoint 保存与加载、各 rank Python/NumPy/CPU/CUDA RNG、连续两步与恢复后下一步一致；仅在验证集校准；部署缓存/index/预算扩展/拒答/结果评估；损坏副本与错误配置拒绝。正式权重容量，合成随机权重，无科学效果解释。
3. 先 CPU 窄测及完整相关回归、独立代码审阅并本地集成；再依次提交 prep phase、reserved 批次、运行、terminal 与摘要；GPU phase 同样操作。worker 期间不编辑受跟踪文件。预期故障只对独立副本测试，成功验收 run 内加载 checkpoint，不制造失败 run 或绕过人工重试授权。意外失败保留并停止，不自动另起 GPU run。

验收上限 8 GPU 小时。报告记录源码/输入/manifest 哈希、UUID/rank 对应关系、耗时和峰值显存。PASS 只证明覆盖的代码链路；不证明正式预训练权重效果、完整数据集 minibatch 调度、科学 gate 或分组/OT 可选实验有效。真实已终态失败任务的跨 run CLI 恢复继续要求人工审批，本次不伪造该审批。

Review focus: reserved 输入与合成来源约束、GPU/world/rank/precision 绑定、DDP 局部全 unknown 的 reducer 行为、checkpoint rank RNG 和精确比较、校准/锁定集与 test 封存、输出仅位于 canonical run、意外失败保留与无静默重试。

实现与审阅记录：CPU 初始窄测 16 项通过；新增四 rank CPU 链路和 FP32 检查通过；完整初次回归 314 项通过。独立审阅发现的验证文件未强制预留、缩减训练模块仍可能声称完整 PASS、数值 CUDA ordinal 不能证明 UUID 映射均修复并有 RED→GREEN 回归。GPU 入口逐 rank 向 CUDA driver 核验实际 UUID，visibility 使用完整 UUID。新增随机输入权重使用 NPZ，保留已有 PT/JIT 读取。核心配置严格匹配已完成并归档的 prep 模板。CPU 恢复比较要求 bitwise，GPU 参数及优化器 moment 最大绝对差要求 ≤1e-6（报告实际差值），各 rank 同一步参数仍要求完全一致。

最终源码回归：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q` → 321 passed，80.12 秒，1 个 PyTorch TypedStorage 弃用警告。没有未处理的审阅问题；GPU 验收记录另在受跟踪 phase/registry/summary 中归档。
