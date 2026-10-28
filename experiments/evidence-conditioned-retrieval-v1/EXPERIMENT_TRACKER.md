# 实验跟踪

| 区块 | 状态 | 证据与后继条件 |
|---|---|---|
| W0 官方预训练输入 | completed | summaries/W0.json；真实 CPU B/32 text/frame forward |
| 来源定义 | approved | reports/source-audit-2026-10-10.md；隔离254训练视频，剩8246 |
| D0 五角色与解码 | completed | summaries/D0.json；64组件/角色，全部320真实解码 |
| Q0-R1/R2/R3/R4/R5 四卡 | completed | 最新R5：13/13检查、578引用、四CUDA rank；最终产品源码1d16e57，690项回归通过 |
| S1 reference | completed | summaries/S1.json；128访问、110更新 |
| S2 五matcher | completed | summaries/S2.json；全部五臂，报告保留损失定义差异 |
| B1 development评价 | completed | summaries/B1-R2.json；五臂全部归档，reports/B1-development-comparison.md/json；保留Gauss/det不按结果挑选 |
| B2 两matcher三角色转移 | completed; gate insufficient | 六臂完整归档；Gaussian Vdev corrected=4、worsened=4，门槛8/4，未通过；reports/B2-transition-gate.json |
| B3 收益/校准/选择 | not run; stopped | B2预注册可检验性门槛不足；不是软件未实现 |
| B4 锁定validation | not run; stopped | B2未通过；V_locked未用于模型评价或选择；test sealed |
| 多seed/扩展 | stopped | 本冻结pilot不充分；不得改参追逐过关 |
