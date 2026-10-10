# B2 冻结转移可检验性 gate

passed=False；Gaussian Vdev corrected组件=4，worsened组件=4；预注册门槛8/4。

| matcher | role | 完整组件 | 候选漏召回 | 转移 | corrected组件 | worsened组件 |
|---|---|---:|---:|---:|---:|---:|
| gaussian | T_value | 64 | 6 | 1526 | 4 | 1 |
| gaussian | V_development | 64 | 4 | 1526 | 4 | 4 |
| gaussian | V_calibration | 64 | 5 | 1526 | 1 | 0 |
| deterministic | T_value | 64 | 6 | 1526 | 2 | 2 |
| deterministic | V_development | 64 | 4 | 1526 | 2 | 3 |
| deterministic | V_calibration | 64 | 5 | 1526 | 0 | 2 |

Insufficient transition variation in this frozen pilot; stop B3/B4 and scaling. This is not a universal refutation of evidence-conditioned retrieval.

全部六臂完成后判定；哈希验证完整 cohort、实际 top-K、unknown、零动作组件。V_locked 未做模型评价，test sealed。
