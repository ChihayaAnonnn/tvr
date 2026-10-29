# UATVR + AQS 局部 Query 开发交付记录

日期：2026-10-29。依据 [实施计划](../plans/2026-10-29-uatvr-aqs-local-query-implementation.md)；初始代码 `201bd78`。

## 实际交付

- 严格 LocalQueryConfig 和9个重叠3×3窗口；一层8-head attention，跨帧共享query参数，非对象跟踪。
- 同一次原CLIP视觉前向暴露CLS和49个patch，不改vendor/旧AQS/KL实现。
- 词到局部token的FP32软匹配、矩形分块与反向重计算；保持mask和梯度语义。
- 新模型工厂、融合分数、完整双向CE、视频级AQS、矩形推理和原MIL/legacy VIB。
- inactive路径直接返回旧AQS；新KL关闭且η=0，Laya/QueryMatch不可用时可完成训练与评分。
- 软件step、不可变配方/身份绑定、旧AQS parent初始化、新架构checkpoint与全rank只读恢复预检。
- 独立local_checkpoint.py承载只读staging，避免导入旧KL评分模块；没有新科学launcher。

## 验证证据

所有测试均为CPU合成fixture，不访问benchmark test。公共测试前缀：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest
```

新增测试目录 `tests/semantic_retrieval/aqs/local_query`：**51 passed，1条已有PyTorch TypedStorage弃用警告，65.35秒**。覆盖：

- 窗口交集/覆盖、词mask、窗口外输入隔离、共享参数和有限非零梯度。
- 原CLS值和梯度严格一致、视觉主模块各调用一次、公共参数keys/冻结不变。
- 逐词FP64 oracle、分块前后分数和梯度、scale只用一次、padding扩展不变、重计算存储检查。
- inactive参数/RNG/损失/梯度严格兼容；AQS关闭仍训练局部模块；融合分数改变选中视频且选中loss保留梯度。
- 双rank Gloo完整loss和全部参数梯度对单进程global-batch参考；配置/ID/shape/空词/非有限输入共同拒绝。
- 单/双rank精确恢复、单rank读取失败后全rank无状态变化、优化器组参数顺序不符拒绝。
- 错误β/τ/窗口/架构/源码/ID/优化器/RNG、损坏外置NPY、KL sidecar拒绝；parent不恢复旧优化器/RNG/步数。
- 离线严格生产配方、禁止fixture/test授权、互斥parent/resume、新进程屏蔽Laya/QueryMatch/新KL运行模块仍可训练和评分。

完整旧AQS/KL与baseline回归和最终独立审查结果将在最终文档提交中补记。

## 实施裁定

1. FP32温度下界1e-38，过小温度明确拒绝；先减最大值再除温度。
2. 新增local_checkpoint.py拆出只读payload验证；保留旧源码。
3. 优化器身份额外绑定组内参数名称/顺序，防止相同整数state索引掩盖错绑。
4. β=0配方不得含local_queries参数；直接使用旧架构训练/恢复API。
5. 允许合法零梯度，例如global batch只有一个视频；有限但为零与分支断开产生None明确区分。

## 未验证范围

没有启动GPU科学训练，没有增加实验预留，没有解封test。尚未验证正式容量CUDA数值、显存/吞吐或检索增益；CPU Gloo不替代对应study的CUDA资格。新局部模块没有新增Gaussian/KL融合、局部时序、对象身份约束或邻窗一致性目标。
