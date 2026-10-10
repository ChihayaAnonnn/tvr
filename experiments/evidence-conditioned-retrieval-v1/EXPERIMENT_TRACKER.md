# 实验跟踪

| 区块 | 状态 | 证据与后继条件 |
|---|---|---|
| W0 官方预训练输入 | completed | summaries/W0.json；真实 CPU B/32 text/frame forward |
| 来源定义 | approved | reports/source-audit-2026-10-10.md；用户批准隔离254训练视频 |
| D0 五角色与解码 | pending | phases/D0.json，64组件/角色 |
| 最终源码 Q0-R1 四卡 | pending | 必须重新资格，旧 Q0 不代替 |
| B1 参考+5 matcher | gated | D0/W0/Q0/预注册全通过 |
| B2 两 matcher 三角色转移 | gated | B1完成；Vdev变化门槛8 corrected组件/4 worsened组件 |
| B3 收益/校准/选择 | gated | B2变化gate |
| B4 锁定 validation | gated | 冻结全部选择；MSRVTT test始终sealed |
| 多seed/扩展 | gated | 预注册正向统计与成本gate，否则停止 |
