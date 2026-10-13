# 官方实现差异与后续判断（2026-10-13）

本次审计完成了固定源码检查、正式 B/32 两样本数值对照及六次训练轨迹核验。计算模块通过所声明的数值检查，但现有链路仍不是论文训练配方的逐项复现；尚不能把论文指标差距定位为某一个因素造成。

固定来源：UATVR `db0062a77d9e7ab6eed801fa5e27c3085939d182`，CLIP4Clip `508ffa3de39ba0563a03199c440ab602a72e9b6f`。成功运行 provenance/official-sources 留存实际读取源码，终态清单绑定文件哈希。UATVR vendored PROVENANCE 的 10 个原始与适配后 SHA-256 均重新核对通过，实际 diff 与登记的命名空间、unused import、DDP、barrier、mask 和空白变更相符。

| 项目 | 官方源码证据 | 当前实现/审计判断 |
|---|---|---|
| 所有 caption | UATVR train.sh:11；CLIP4Clip README.md:107 均启用 expand_msrvtt_sentences | 当前也是所有 caption；旧文档“官方随机 caption”已更正 |
| batch 组成 | 两者 dataloaders/data_dataloaders.py 的 DistributedSampler，drop_last=True | 当前每个全局 batch 视频唯一、保留最后 batch；负例分布与更新数不同 |
| UATVR 活动输入 | dataloader_msrvtt_retrieval.py:415 调用 _get_rawframes；rawframes_util.py 的 30fps JPEG、训练随机偏移 | 当前 MP4 确定性 1fps 采样；本次数值验证仅覆盖官方非活动 rawvideo 分支 |
| UATVR 上游可运行性 | rawframes_util.py:101、111 有活动 pdb.set_trace | 没有删除这些语句并冒充原样完整训练；活动帧链路尚未完成端到端一致性验证 |
| backbone | UATVR train.sh:16 默认 B/16；CLIP4Clip README 的 MSRVTT 命令 B/32 | 当前两者 B/32；不能把 UATVR 默认脚本当作 B/32 论文行的完全配方 |
| UATVR token/采样 | 公开代码 2/2 extra token、7 Gaussian samples | 当前匹配；论文 token ablation 的 3/2 配置另有区别，未宣称逐参数复现该行 |
| LR | UATVR 5e-5、CLIP4Clip 1e-4，coef_lr=1e-3 | 当前匹配；CLIP 峰值为 5e-8/1e-7，不能把小学习率直接判作实现错误 |
| optimizer/冻结 | 官方 BertAdam、warmup_cosine、freeze_layer_num=0 | 两步实际参数更新及独立冻结规则核验通过 |
| DDP | 官方 AllGather backward 返回本 rank 切片 | 当前 autograd collective 修正全局梯度；真实两 rank/serial 回归通过，但不宣称与官方多卡轨迹相同 |
| 数据协议 | 论文常用 9k train / 1k test | 当前 8246 train、500 validation，来源隔离与选模为用户已批准协议；不能在审计中擅改 |
| ranking | 官方 UATVR 推理 WTI；Gaussian 分支主要参与训练 | 当前匹配；不得把 baseline 提升解释为新 evidence-conditioned 方法有效 |

## 当前可做的决策

1. 已排除所测条件下的核心 forward/loss/backward/optimizer 移植不一致；优先核对数据、时间采样与 batch 分布，不要同时换 backbone、损失与学习率。
2. 如进入下一轮研究，最有依据的受控因素是 UATVR 训练时间偏移与输入帧链路，以及官方 batch 组成。需要各自独立、validation-only 的预注册对照，不能通过修改本次只读 CPU study 启动训练。
3. UATVR 后期训练损失下降而 validation 平台，不支持直接加 epoch；CLIP4Clip 也仅有很小的末轮收益。更大的 CLIP LR 仅可作为待验证假设，官方一致性审计没有提供“应立即增大”的证据。
4. 历史没有梯度范数、裁剪频率、各 loss 梯度贡献。后续新身份实验应先补这些测量，不能用 loss 数值比例替代梯度诊断。

本次没有访问 test、没有修改历史模型/checkpoint、没有启动科学 GPU 实验，也没有重开已失败的 evidence-conditioned pilot gate。论文差距的因果解释仍待受控实验；数值一致不等于创新性或研究假设成立。

详见 [数值报告](numerical-audit.md)、[30 轮轨迹](training-trajectories.md)；成功 phase 为 A2，失败 A1 与修复记录均保留在本 study registry。
