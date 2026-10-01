# C1 真实 train-only 数据验收

2026-10-01：D1S 与 D1 均 `completed / exit_code=0`，整批 reserved 与 terminal registry 先后提交；worker 期间受跟踪源码无漂移。此结果只证明数据与合同链路就绪，不代表科学训练或新四卡资格已通过。

| 分区 | 官方视频 | canonical 视频 | 查询 | 来源组件 |
|---|---:|---:|---:|---:|
| train | 7000 | 7000 | 7000 | 7000 |
| val_development | 500 | 500 | 500 | 500 |
| val_calibration | 500 | 500 | 500 | 500 |
| val_locked | 1000 | 1000 | 1000 | 1000 |

输入为 hash-pinned MSRVTT v2 与 CLIP4Clip 发布的 9k ID-only CSV。D1S 按允许 ID 先选择记录，再仅解码训练 caption；训练关系指纹与已提交的独立核验报告一致：`bc2c06cbbe833d4e504700430a8a96c23bb7a79f2a5418e1a137ab0fe32b123d`。导出180000条训练 caption、9000条按最小稳定 caption ID 选择的查询，排除20000条记录的 caption 解码数为0。本次导出授权来自用户明确要求完成独立快照的 Task 7，不沿用此前只读比较的许可。

9000个视频均核验完整解码帧数和8个均匀采样帧：总计3667517帧，帧数范围60–900，中位数360。精确内容 SHA 全部不同。D1 重新核验视频、证明完整训练记录投影、验证选中 caption 属于投影，并检查四分区的来源/组件互斥及 GT 所属库。各查询恰有一个 canonical 标注正例，数量门全部通过，未产生语义全库无答案标签。

D1 只读取独立 train-only 快照、ID-only CSV 与视频，不将混合 v2 标注列入其输入。MSRVTT/MSVD test 保持 sealed。此划分来自9k训练来源，不是官方1K-A test，不能直接与其 SOTA 表比较。

## 可复核记录

- D1S source `5add32a296d9`，manifest `166f30b6ecc172d100a8d8ead5240924ec2460f9ba5b9782b9e5feaa77a10018`；归档提交 `fa189d1`，12个内容输出文件。
- D1 source `bbaa8ea7fa8a`，manifest `f3d3adeb1c997d9153c9a3db1e365698436f4b25f5ea6d3882c4853d0907647c`；归档提交 `8fe109a`，18个内容输出文件。
- [D1S摘要](../../../experiments/semantic-answerability-pilot-v1/summaries/D1S.json)、[D1摘要](../../../experiments/semantic-answerability-pilot-v1/summaries/D1.json)、[append-only registry](../../../experiments/semantic-answerability-pilot-v1/registry.jsonl) 保存全部输出 SHA；原文件位于共同仓库的数据面，各运行的 `provenance/preparation-result.json` 提供不可变输入/manifest 引用。

9000个视频的逐文件身份通过10片已绑定输入索引保留，manifest 自身保持小于1MiB。C1 编译的129项候选为按 canonical ID 排序的软件合同库存，科学排序仍由 E0 的固定 CLIP 粗检索生成。后续训练/分析只消费对应的新 SHA，不能把此库存当作真实粗检索结果。
