# B1 development validation 对照

64 unique source components, one query each, 64-video gallery and frozen top8 candidates; not official test benchmark

| matcher | 初始正确/64 | 最终正确/64 | corrected | worsened | 帧请求 | CLIP帧 | temporal | MC元素 | rich NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | 16 | 14 | 1.0 | 3.0 | 4096.0 | 4096.0 | 27648.0 | 4608.0 | undefined |
| fixed-scale | 18 | 18 | 1.0 | 1.0 | 4096.0 | 4096.0 | 27648.0 | 36864.0 | 0.9322144571512565 |
| learned-temperature | 16 | 14 | 1.0 | 3.0 | 4096.0 | 4096.0 | 27648.0 | 4608.0 | undefined |
| standard-probabilistic | 16 | 15 | 1.0 | 2.0 | 4096.0 | 4096.0 | 27648.0 | 36864.0 | -0.28139390562901767 |
| gaussian | 16 | 17 | 2.0 | 1.0 | 4096.0 | 4096.0 | 27648.0 | 36864.0 | -0.578366819947636 |

完整 SHA-256 引用、组件配对区间、索引离线成本与实际运行时间见同名 JSON。
不能把相同资源上限说成实际成本相同。训练额外监督差异不支持单独的 uncertainty 因果归因。
五个并发 worker 共享 CPU/I/O，延迟不用于孤立方法速度比较。
本表不执行模型选择；按预注册仍使用 Gaussian 和 deterministic 编译转移。
test sealed。
