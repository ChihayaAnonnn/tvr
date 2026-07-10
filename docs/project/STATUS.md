# UATVR 当前状态

> 更新时间：2026-07-10

## 当前判断

历史 MSRVTT 数字受到 JSFusion test 逐 epoch 选模、同视频描述错误负例和 WTI padding 三项混杂影响，只作为历史档案，不再直接用于新模型决策。

## 已终止方向

Hard negative、UACL、置信度加权和 query-conditioned SAP 不再作为训练主线；其结果保留为负结果与边界分析。

## 唯一实验协议

- `trusted-v1`：8500 train / 500 internal val / JSFusion 1K blind test；
- val 使用每视频全部 20 条官方描述；
- checkpoint 只按 val T2V R@1 选择；
- 主损失为按精确 `video_id` 定义的双向多正例 InfoNCE；
- hygiene 只执行 WTI 路径。

## 下一步

先重跑 OpenAI CLIP trusted hygiene WTI-only 基线；基线建立后，才在完全相同协议下评估 EVA02-CLIP-B/16 adapter。

完整问题、证据和停止条件见
[`RESEARCH_ISSUES_AND_ROADMAP.md`](RESEARCH_ISSUES_AND_ROADMAP.md)。
