# MSRVTT train/validation 来源审计（2026-10-10）

这是只读输入审计，不是科学实验、训练授权或新的检索结果。审计遵循当前会话授权；没有解码 test 的 URL/时间字段或 caption，没有打开 test 视频；未修改官方数据和 trusted split。

## 输入身份和方法

- 混合注释：`/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json`，SHA-256 `8342df0ed6f8212ec8f4363e7415ce9a98d6ddf89e25ed2fb72b364f97c65d33`。
- 发布的 9k 训练 allowlist：`/data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv`，SHA-256 `0a7ece2418e1f7ac5680555a04ddfef8950293ac3c4bdd6584fb0fa6cc0ff7d3`。
- trusted-v1：`dataloaders/splits/msrvtt_trusted_v1_seed0.json`，SHA-256 `a0e5a82dddec34fa76753c2323cf560c2afc36f3683b231ccd24db2c268be882`；8500 train / 500 validation，互斥且并集等于发布的 9000 个训练 ID。

复现：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python scripts/audit_evidence_sources.py
```

解析器只先解码视频 ID，命中 9000-ID allowlist 后才解码 URL/start/end。其它 root 值与不在 allowlist 中的元数据值保持编码字节。所有 caption 均不解码。对允许的 9000 个文件逐一 SHA-256，不以文件名代替内容；8 个 I/O 线程，CPU-only，stdout-only。

另外进行了内存中的只读发布来源核对：下载 CLIP4Clip 作者发布 ZIP，先验证既有来源确认中的 SHA，再从 ZIP 内读取 annotation，复用同一 allowlist-first 解析器。发布 URL：<https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip>。ZIP SHA-256 `93958a5f1c322b32349d0948eba0cb24c25a34d54f45bd5e037027d266ea6e56`；成员 `msrvtt_data/MSRVTT_data.json` SHA-256 `3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014`。9000/9000 训练记录的 video_id、URL、start time、end time 与本地一致；不同记录 0；两侧解码的排除元数据值和 caption 值均为 0。按 video_id 排序、sort_keys=True 的紧凑 JSON 训练来源元数据 SHA-256 为 `c0e9badac5e2d73d37faa8ca584cd478de5742aecf9217a2b983117b25e577d6`。该比较不宣称其它元数据全部一致。

## 实际结果

| 检查 | 数量 |
|---|---:|
| 允许的视频 | 9000 |
| 不同原始 YouTube 来源 | 6636 |
| 包含多个片段的来源 | 1986 |
| 跨 trusted train / validation 的来源 | 202 |
| 上述来源中的训练视频 | 254 |
| 上述来源中的验证视频 | 203 |
| 同来源且标注时间区间有正长度交集的跨分区片段对 | 31 |
| 字节完全相同的文件组 | 0 |
| 排除的 test 元数据记录 | 1000 |
| 解码的 test 元数据非身份字段 / 所有 caption 值 | 0 / 0 |

视频字节总量：2,908,513,591。按 video_id 排序的 `[video_id, sha256, byte_size]` 紧凑 JSON 清单 SHA-256：`96d6e6a6aa81bcf2fcb31a2c9750de27fbe3dacd28d1c5ee33ac479f007f2fed`。这里只保存清单摘要，不保存逐查询数值载荷。

时间重叠示例（train / validation / 秒）：video236 / video4382 / 28.48；video6129 / video1513 / 18.39；video974 / video6275 / 17.33。它们是发布来源元数据的重叠证据，尚未被解释为逐帧内容完全相同或性能泄漏量；未计算任何检索指标。

## 对本研究的影响与已批准修订

原设计 §5.4 要求来源组件不跨五个角色。当前 trusted split 的视频 ID 互斥不能证明该要求；已确认存在来源交叉，科学 gate 继续为 false。

建议的具体修订：

1. 用相同原始 YouTube ID 或相同视频 SHA-256 建边，取连通分量；同源但时间不重叠的片段也保守合并。caption、视图、转移继承来源组件。
2. 保留原 trusted validation 的 500 个视频成员；从本 study 的训练可用集合排除与 validation 同源的全部 254 个训练视频，得到 8246 个训练可用视频。原数据文件和 trusted split 不改写，排除名单写入新 study 的独立角色 manifest。所有方法使用同一子集。
3. 在剩余训练来源组件内划分 T_match/T_value；在原 validation 来源组件内划分 V_development/V_calibration/V_locked。按组件划分、无标签/结果参与，最后再次验证成员资格、组件/内容/caption/转移互斥。
4. 明确把这个有限的操作定义写入批准后的设计修订。它可消除已知原始来源交叉和字节重复；不能保证不同 YouTube ID 的重传、剪辑拼接或未知视觉近重复全部被排除。这个限制需保留在研究报告中。

**状态：用户于 2026-10-10 明确批准上述来源定义与训练子集修订。** 本报告不是通过的 split gate；没有生成“已审计通过”的五角色 manifest，没有修改科学授权。若用户要求跨 URL 的视觉近重复也必须审计，需先明确可审核的检测与确认方案，不能静默把相似语义当重复。

## 验证

新增行为测试先失败（缺少模块），实现后 `test_source_audit.py`：7 passed。覆盖 allowlist-first 值解码隔离、来源与重叠/字节重复区分、缺失成员、分区交叉、非有限时间、错误 URL 和重复 ID。真实 pinned 数据审计成功完成，结果如上。
