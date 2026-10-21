# 已完成采帧与 batch 对照：MSRVTT 1k test

2026-10-22。UATVR ViT-B/32，seed713，FP32，DSL=false，完整1000视频×1000查询。训练集为来源隔离后的8246视频。仅使用此前 validation 选定的 checkpoint；原始基线复用已归档 test，不重新评估。

| 设置 | validation 选定 epoch | T2V R@1 | ΔT2V（百分点） | V2T R@1 | ΔV2T（百分点） |
|---|---:|---:|---:|---:|---:|
| 原始基线 | 5 | 42.9% | +0.0 | 42.2% | +0.0 |
| 采帧时间偏移 | 5 | 42.5% | -0.4 | 42.5% | +0.3 |
| caption 随机组 batch | 3 | 42.2% | -0.7 | 41.9% | -0.3 |

两项改动均未提升 T2V R@1。采帧组 V2T 小幅上升，batch 组两个方向均下降。这是单种子、已有 test 暴露后的描述性结果，不支持稳定增益或新不确定性方法成立的主张。batch 的 epoch3 来自 validation 选模；此前固定 epoch5 validation 终点仍是单独口径，不能混用。失败/未完成的 CLIP4Clip 对照未评 test。

## 验证与归档

243项受影响回归通过（475.61秒）；独立只读源码审查无阻断问题。真实正式容量双CUDA Q0的9项检查与factor链路资格全部通过。三个结果的registry/manifest/终态/提交摘要绑定、指标文件SHA-256、外置数组的内容引用哈希和NPY文件哈希均核验。原始与本轮control的选定模型哈希一致：`f12bd3aa6dc18e03badc85ba48defd786fe5667c33464aa597188e3a12e982a4`。

本次新增GPU用量（分配卡数×wall time，非GPU忙碌时间）：Q0 0.3286 GPUh；两个test合计 1.9650 GPUh；总计 2.2935 GPUh。设备0、3，无重试。

实现提交 `25be352`；Q0终态 `bdfb00e`，gate `3763516`；test预留 `245591d`，整批终态 `9d7d1da`。完整run身份、source SHA、哈希、指标和耗时见同目录 `factor-jsfusion-test-seed713.json`。

- 原始基线：`/data2/hxj/project/UATVR/artifacts/full-data-retrieval-v1/msrvtt/final/msrvtt-final-test-uatvr-seed713-20261012T143236Z-ca9fc5567fc6`
- 采帧时间偏移：`/data2/hxj/project/UATVR/artifacts/full-data-retrieval-v1/msrvtt/final/msrvtt-final-test-uatvr-temporal_offset-seed713-20261021T162044Z-150ecb2b3d85`
- caption 随机组 batch：`/data2/hxj/project/UATVR/artifacts/full-data-retrieval-v1/msrvtt/final/msrvtt-final-test-uatvr-caption_shuffle-seed713-20261021T162044Z-150ecb2b3d85`

最终独立数字复核：三条指标文件与报告完全一致，epoch同时匹配manifest，所有百分点差值与GPU用量计算正确；未发现过度声称。另从已核验的1000项one-based rank数组独立重算R@1，与三个结果逐项一致。
