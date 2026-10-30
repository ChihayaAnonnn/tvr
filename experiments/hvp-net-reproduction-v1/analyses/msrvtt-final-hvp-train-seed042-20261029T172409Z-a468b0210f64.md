# HVP-Net 修正版复现结果

- Run: `msrvtt-final-hvp-train-seed042-20261029T172409Z-a468b0210f64`
- 状态：completed，5 epochs，训练耗时 05h16min53s；GPU 0/3 已释放。
- 配置：B/32、双卡、global batch32、seed42、作者 FP16、9k train / JSFUSION 1k test。
- 按用户要求，沿用 test T2V R@1 选点，无独立 validation。
- 最佳：第5个 epoch（文件编号4），T2V R@1 32.3；结果、日志哈希与终态 inventory 一致。

| Epoch | T2V R@1 | R@5 | R@10 | V2T R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 29.8 | 57.8 | 70.4 | 28.3 | 56.1 | 68.3 |
| 2 | 30.4 | 59.5 | 70.6 | 28.4 | 58.1 | 69.8 |
| 3 | 32.2 | 61.1 | 73.8 | 32.0 | 60.3 | 71.8 |
| 4 | 31.9 | 61.8 | 73.8 | 31.4 | 62.0 | 73.1 |
| 5 | 32.3 | 61.1 | 74.1 | 31.0 | 61.5 | 72.4 |

## 最佳 checkpoint 最终重评

| 方向 | R@1 | R@5 | R@10 | MdR | MnR |
|---|---:|---:|---:|---:|---:|
| T2V | 32.3 | 61.1 | 74.1 | 3.0 | 18.6 |
| V2T | 31.0 | 61.6 | 72.5 | 3.0 | 17.5 |

- 权重：`/data2/hxj/project/UATVR/artifacts/hvp-net-reproduction-v1/msrvtt/final/msrvtt-final-hvp-train-seed042-20261029T172409Z-a468b0210f64/checkpoints/best.pth`
- 日志：`/data2/hxj/project/UATVR/artifacts/hvp-net-reproduction-v1/msrvtt/final/msrvtt-final-hvp-train-seed042-20261029T172409Z-a468b0210f64/logs/train.log`

## 结果解释与限制

- 论文报告 T2V 56.7/83.9/90.6，本次最终为32.3/61.1/74.1，分别低24.4/22.8/16.5个百分点。未复现论文报告性能。
- 本次包含查询词权重、双卡梯度与评估等修正，且环境、seed及双卡执行与作者设置可能存在差异；不能把差距全部归因于某项修复，也不能据此断言论文结果无效。
- 同一最佳权重，第5轮评估V2T R@5/R@10为61.5/72.4，最终重评为61.6/72.5，存在微小重复评估波动；表中最终指标使用最终重评。
- 日志 Non-finite/Traceback/RuntimeError 计数均为0；未找到匹配诊断TSV。
- 与此前UATVR+AQS局部分支42.6的T2V R@1相比低10.3个百分点，但数据协议/选点方式不同，不是受控消融比较。

报告保存在控制面，未向已经终态并校验哈希的数据面添加或改写文件。

## 修复影响排查（CPU，只读检查；未新增训练或 test 评估）

- 只有组合修正版的一次训练，不能给出各项修复的因果贡献。
- 查询词权重从 `abw,bw->ab` 改为 `abw,aw->ab` 同时改变训练和推理，是最直接改变方法语义的修改。原版依赖候选视频的配对 caption，因而恢复原版高分也不能自动证明标准视频检索性能。
- AllGather 反向新增跨 rank 梯度求和，改变双卡训练的梯度，尚无相同条件的原版/修正版训练对照；不能视为绝对无影响。
- CPU/Gloo 通信、index 排序去重、strict checkpoint 格式修复在成功执行且样本顺序相同的条件下，本身不应改变对应 pair 的模型分数；固定 MPP 随机数可改变具体抽样。
- 另有非新增修复的数值风险：CLIP lr=1e-4×1e-3=1e-7，BertAdam 的 FP32 update 被转回参数 dtype 后原地相加，没有 FP32 master weights。
- 对最终 best.pth 与已绑定官方 CLIP 权重作 CPU 逐元素比较：visual.conv1.weight（FP16）完全一致；visual.transformer.resblocks.0.attn.in_proj_weight 仅1.349%元素变化；visual.proj仅1.190%；text_projection仅1.226%。FP32 token_embedding.weight 有41.029%元素变化。因此不能称整个CLIP冻结，但部分半精度层的有效更新极小。该现象与量化舍入风险一致，不能单独证明其造成24.4个百分点差距。
- 上游7743b598的提交说明称视频侧caption加权评估将某E0模型R@1由60.8改为94.6；这是作者提交说明中的未经独立验证数字，不是本次实测，也不是论文56.7的复现证据。
