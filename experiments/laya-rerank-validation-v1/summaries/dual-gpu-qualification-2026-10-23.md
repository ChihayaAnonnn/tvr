# Laya 双卡软件资格通过 — 2026-10-23

范围：用户明确授权本 study 使用双卡；GPU 1、3，两个真实 CUDA rank。无科学 checkpoint、真实 validation 查询或 test 数据输入。原始设计的四卡要求仅在本 study 由双卡修订替代。

成功 run：`synthetic-baseline-laya-q0-dual-retry1-seed713-20261023T082432Z-6f70cc5f3471`。
Source SHA：`6f70cc5f3471897375db2c6c6f347e9f120f0b00`。
Manifest SHA-256：`be595953189b852a58a40132dffff67fd4fe5a9dff57cddeceaf745a8c2bae2f`。
Qualification report SHA-256：`40ec00066853ad13f18452bb164606f1bc7208d4ec22a2c6a443f3b8abb81787`。

正式容量随机初始化 UATVR + Laya（201,161,347 参数）；原生 processor、FP32、真实合成视频解码和视觉缓存路径。**8/8 检查通过**：真实推理、原始 UATVR 推理、视觉缓存、有限 FP32、微批1/8、一卡全量与双 rank 合并、缓存/未缓存、强制退出后全新进程恢复。恢复后的完整排列及 logits/概率数组与连续运行逐项一致；分片数值误差断言仍为1e-5且argmax/排序一致。

微批1/8 logits 最大绝对误差 `4.47034835815e-07`；缓存/未缓存最大绝对误差 `0.0`。控制器累计 `79.698` 秒、两卡 `0.044276510 GPU·小时`（软件预算8 GPU·小时）。此前一次控制器环境检查在 worker 启动前取消并归档，计0 GPU·小时；成功后继有已提交诊断链。原失败目录不复用、不覆盖。

终态 `completed`，215 个输出哈希已逐项复核。所有拥有的进程及独立 rank session 已退出，GPU 1、3 无本 run 遗留计算进程。外部进程未停止或修改。

此为软件资格证据，不是精度、科学假设或在线效率结论。完整 validation 精度评估与正式性能测量尚未启动。`science_authorized=false`、`test_authorized=false` 保持；科学执行器的预算/恢复封装与正式测量须独立落实后再启动，不能凭软件通过宣称科学收益。当前资格绑定本 source，后续可执行代码变更须重新验证资格适用性。
