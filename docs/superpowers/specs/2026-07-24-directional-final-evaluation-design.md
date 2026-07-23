# 定向最终检索评测设计

## 目标

最终测试中，T2V 验证最优 checkpoint 仅报告 T2V，V2T 验证最优 checkpoint
仅报告 V2T；训练期间的验证仍同时报告两个方向并独立选择最佳 checkpoint。

## 接口

`eval_epoch` 新增关键字参数 `directions=("t2v", "v2t")`。可选值为
`"t2v"` 和 `"v2t"`，重复值去重；空集合和非法值均抛出 `ValueError`。

返回字典只包含请求方向。默认值维持所有现有调用方的双方向行为。

## 数据流

评测仍统一提取文本/视频特征并生成相似度矩阵。随后按 `directions`：

- T2V：从文本到视频矩阵计算指标；
- V2T：从视频到文本矩阵计算指标；
- 仅请求一个方向时，不调用另一方向的指标函数，也不记录其日志。

RSPR 的方向性矩阵仅在该方向被请求时构建，以避免单方向最终测试额外执行另一方向的 reranking。

## 最终测试

`--run_final_test` 遍历验证选择方向，但每次调用
`eval_epoch(..., directions=(direction,))`。日志说明 checkpoint 的选择方向与
报告方向；`final_test.json` 的每个 selection 仅含与其 key 相同的
`test_metrics` 子项。

## 测试

测试方向验证与默认兼容性，覆盖单 T2V、单 V2T 及最终测试调用会传入对应单方向。
