# 实验跟踪

| 区块 | 状态 | 证据与后继条件 |
|---|---|---|
| W0 官方预训练输入 | completed | summaries/W0.json；真实 CPU B/32 text/frame forward |
| 来源定义 | approved | reports/source-audit-2026-10-10.md；隔离254训练视频，剩8246 |
| D0 五角色与解码 | completed | summaries/D0.json；64组件/角色，全部320真实解码 |
| Q0-R1/R2/R3/R4 四卡 | completed | 最新R4：13/13检查、578引用、四CUDA rank；最终产品源码f80918b |
| S1 reference | completed | summaries/S1.json；128访问、110更新 |
| S2 五matcher | completed | summaries/S2.json；全部五臂，报告保留损失定义差异 |
| B1 development评价 | completed | summaries/B1-R2.json；五臂全部归档，reports/B1-development-comparison.md/json；保留Gauss/det不按结果挑选 |
| B2 两matcher三角色转移 | gated | B1完成；Vdev变化门槛8 corrected组件/4 worsened组件 |
| B3 收益/校准/选择 | gated | B2变化gate |
| B4 锁定validation | gated | 冻结全部选择；MSRVTT/MSVD test始终sealed |
| 多seed/扩展 | gated | 预注册正向统计与成本gate，否则停止 |
