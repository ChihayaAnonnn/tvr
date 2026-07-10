# UATVR 当前状态

> 更新时间：2026-07-10
> 科研决策主文档：[`RESEARCH_ISSUES_AND_ROADMAP.md`](RESEARCH_ISSUES_AND_ROADMAP.md)

## 当前阶段

项目已结束 Hard Negative、UACL、置信度加权和 query-conditioned SAP 的主线探索，当前进入 **可信实验基座建设**。后续 MSRVTT 结论只接受 `trusted-v1` 协议；现有 48.2、49.3、50.0 等结果均来自 legacy 协议，只作历史边界，不能作为新协议基线。

## 可信指标与边界

| 结果 | T2V R@1 | 当前解释 |
|------|---------:|----------|
| B1-only v2 repeat1 | 49.3 | legacy 协议下最可靠的主线参照 |
| Exp1 历史高点 | 50.0 | 未稳定复现；repeat2 为 49.1 |
| legacy hygiene（loss-zero） | 48.2 | 仅关闭辅助 loss，不是真正的纯 WTI 前向 |
| Hard Negative | explicit 49.4；model-mined 48.6 | fixed/regressed 不占优，主线终止 |
| UACL epoch-4 四组 | 49.3 / 49.2 / 49.4 / 49.0 | 单次 49.4 为噪声级，路线冻结 |

新协议的首要指标仍为 internal val T2V R@1；JSFusion 1K test 只允许在 checkpoint 确定后独立盲测。EVA 的判断依据是相对完全匹配的 OpenAI CLIP 对照变化，而不是直接超过上述 legacy 数字。

## 路线状态

- **已终止**：Hard Negative 主线、置信度加权、global probability score、AnchorWTI、query-conditioned SAP、semantic soft target/伪标签。
- **已冻结**：UACL；不再追加 epoch、sweep 或同机制 repeat。
- **进行中**：`trusted-v1` 实施准备，以及后续 OpenAI CLIP / EVA02-CLIP-B/16 公平配对基线设计。

## 实现状态

- `trusted-v1` 已确认但**尚待代码实施**：seed 42 固定 8500 train / 500 internal val，val 每视频 20 条官方描述；训练不得构造 test dataloader；正例只按精确 `video_id` 定义，双向多正例 InfoNCE 是唯一主检索损失。
- 当前 `EXPERIMENT_PROFILE=hygiene` 只完成辅助 loss/开关归零，**尚未证明是真正的 WTI-only 前向**。目标实现必须从 forward 路径绕过 SpatialEnhancer、SAP、视频概率分支、文本 PIENet、不确定性头及概率采样/中间张量构造，并由 spy/mock 测试验证未调用。
- 当前主排序仍为 `weighted_logits = wti_logits`；HN/UACL 代码仅保留为默认关闭的消融与诊断入口。
- EVA02-CLIP-B/16 adapter 仍属于独立的未提交代码工作树，不属于本次文档状态归档。

## 唯一 P0

**先实施并验证完整 `trusted-v1` 可信实验基座，包括固定 split、train/val/test 隔离、精确 `video_id` 双向多正例 InfoNCE、WTI padding 修复和真正绕过全部概率辅助模块的 hygiene WTI-only 前向。**

P0 完成后，先建立 OpenAI CLIP trusted hygiene 基线，再以相同 seed、GPU、batch、gradient accumulation、每次 forward 全局对比 batch、optimizer steps 和 checkpoint-selection 指标运行 EVA02-CLIP-B/16 配对对照。

## 权威链接

- 科研问题、证据等级、停止条件与路线：[`RESEARCH_ISSUES_AND_ROADMAP.md`](RESEARCH_ISSUES_AND_ROADMAP.md)
- `trusted-v1` 设计规格：[`../superpowers/specs/2026-07-10-trusted-experiment-foundation-design.md`](../superpowers/specs/2026-07-10-trusted-experiment-foundation-design.md)
- 实施计划：[`../superpowers/plans/2026-07-10-trusted-experiment-foundation.md`](../superpowers/plans/2026-07-10-trusted-experiment-foundation.md)
- HN/UACL 边界归档：[`UACL_HARD_NEG_PLAN.md`](UACL_HARD_NEG_PLAN.md)
- UACL epoch-4 最终状态：[`../../logs/20260704/uacl_running_status_20260705.md`](../../logs/20260704/uacl_running_status_20260705.md)
- 日志索引：[`../logs/README.md`](../logs/README.md)
