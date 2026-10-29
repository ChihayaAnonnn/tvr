# 已下载论文索引

2026-09-30 整理时核心目录有 7 篇论文、`paper/` 中有 4 份参考 PDF；
该历史快照见 [文件清单与哈希](../docs/maintenance/2026-09-30-papers.tsv)。
本次精读新增 Holmes 与 GeoFlowVLM，当前核心目录共 9 篇、参考 PDF 共 13 份。
新资料的版本与哈希见各自 SOURCE，原文件名与路径保留，以免破坏研究证据引用。

本页整理已有资料，不代表重新核验出版状态或复现结果。版本依据本地 SOURCE/NOTES；
论文数字的引用状态请查 [evidence ledger](../research/01-evidence-ledger.md)。

| 工作 | 本地 PDF / 文本 | 来源记录 | 在本仓库中的阅读入口 |
|---|---|---|---|
| UATVR，ICCV 2023 | [PDF](uatvr_iccv2023/paper.pdf) / [文本](uatvr_iccv2023/paper.txt) | [SOURCE](uatvr_iccv2023/SOURCE.txt)，2301.06309v2 | 历史基线与相关工作；概率组件增益普查 |
| DiffusionRet，ICCV 2023 | [PDF](diffusionret_iccv2023/paper.pdf) / [文本](diffusionret_iccv2023/paper.txt) | [SOURCE](diffusionret_iccv2023/SOURCE.txt)，2303.09867v2 | 扩散检索相关工作；不要将其已有的候选分布生成作为新点 |
| PAU，NeurIPS 2023 | [PDF](pau_neurips2023/paper.pdf) / [文本](pau_neurips2023/paper.txt) | [SOURCE](pau_neurips2023/SOURCE.txt)，2309.17093v3 | Prototype-based Aleatoric Uncertainty Quantification |
| T-MASS，CVPR 2024 | [PDF](tmass_cvpr2024/paper.pdf) / [文本](tmass_cvpr2024/paper.txt) | [SOURCE](tmass_cvpr2024/SOURCE.txt)，2403.17998v1 | 文本随机嵌入；概率组件增益普查 |
| ProTA，2024 预印本版本 | [PDF](prota_2024/paper.pdf) / [文本](prota_2024/paper.txt) | [SOURCE](prota_2024/SOURCE.txt)，2404.12216v2 | Probabilistic Token Aggregation |
| DUQ，IJCAI 2025 | [PDF](duq_ijcai2025/duq.pdf) | PDF 首页及 ledger E-81；无独立 SOURCE 文件 | Dual Uncertainty Quantification；暂无本地提取文本 |
| DAB，作者记录为 ECCV 2026 | [PDF](dab_eccv2026/dab.pdf) / [文本](dab_eccv2026/dab.txt) | [NOTES](dab_eccv2026/NOTES.md)，2607.20984v1 | KL/candidate-only 项与 hubness 假说的来源；笔记中的代码发布状态是历史快照 |
| Holmes，作者记录为 ICML 2026 | [PDF](holmes_icml2026/paper.pdf) / [文本](holmes_icml2026/paper.txt) | [SOURCE](holmes_icml2026/SOURCE.txt)，2605.06083v1 | [精读笔记](holmes_icml2026/NOTES.md)：训练监督、部署边界、论文与代码表达式差异 |
| GeoFlowVLM，2026 预印本 | [PDF](geoflowvlm_2026/paper.pdf) / [文本](geoflowvlm_2026/paper.txt) | [SOURCE](geoflowvlm_2026/SOURCE.txt)，2605.13352v2 | [精读笔记](geoflowvlm_2026/NOTES.md)：条件熵、density 代理量、风险下界与 gallery 语义 |

跨论文比较已有 [OQ1 Tier −1 增益形状普查](../research/oq1/tier-minus1-gain-shapes.md)。
它是文献报告值的分析，不能等同于本仓库复现。

## `paper/` 中的补充资料

该目录被 Git 忽略，以下文件仅存在于本地，不随普通 clone 分发。
标题根据文件名或 PDF 首页/摘要辨认；不补造下载日期、版本或来源 URL。

| 文件 | 识别结果 | 处理 |
|---|---|---|
| [14394-LiuH.pdf](../paper/14394-LiuH.pdf) | Uncertainty-Aware Contrastive Learning With Hard Negative Sampling for Code Search Tasks，AAAI 2025 | 代码检索旁支；保留 |
| [Fang 等 - 2023 - UATVR Uncertainty-adaptive text-video retrieval.pdf](<../paper/Fang 等 - 2023 - UATVR Uncertainty-adaptive text-video retrieval.pdf>) | UATVR 另一份 PDF；与核心目录版本哈希不同，未做内容差异比对 | 主引用优先使用有 SOURCE 的核心版本；两份均保留 |
| [paper1.pdf](../paper/paper1.pdf) | 面向代码搜索和文本分类的不确定性感知对比学习方法研究（据内页页眉、摘要） | 泛名文件已在本索引识别，保留原名；来源未溯源 |
| [基于不确定性感知与时空解耦建模的交通流量预测研究.pdf](../paper/基于不确定性感知与时空解耦建模的交通流量预测研究.pdf) | 交通预测旁支（按文件名） | 保留；来源未溯源 |

另有 [代码检索参考实现](../paper/Uncertainty-Aware-Contrastive-Learning-With-Hard-Negative-Sampling-for-Code-Search-Tasks/README.md)，
不属于 UATVR 训练代码；`paper/figs/` 是旧稿图表及生成脚本，不是下载的参考论文。

## 新资料归档

新下载的核心论文统一放到 `related_work/<方法_年份>/`，附 PDF、可选文本和 SOURCE
（完整标题、来源 URL、版本、获取日期）。阅读结论放 NOTES，区分原文事实与研究猜想。
聊天中检索过的论文不自动视为已经下载；新增下载以本索引与对应 SOURCE 为准。

## 2026-10-30 下载与仓库核对

- DAB：复用并校验 [PDF](dab_eccv2026/dab.pdf)；与新下载的 arXiv 2607.20984v1 完全一致。[来源记录](dab_eccv2026/SOURCE.txt)。本次未找到可确认的官方代码仓库。
- HVP-Net：[PDF](hvp_net_2026/paper.pdf)、[文本](hvp_net_2026/paper.txt)、[来源记录](hvp_net_2026/SOURCE.txt)。[作者仓库](https://github.com/boyun-zhang/HVP-Net) 已克隆至本地 `research_refs/HVP-Net/`，固定 commit `7743b5981e12e598a0fdf1aa0f98f98500f089e6`；README 仅含标题，但有模型和训练代码。
