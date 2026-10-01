# B 阶段实现验证

范围：新 semantic_retrieval 系统、合成 CPU 验证。不启动科学实验，不触碰 sealed test。

## B1

原始 episode manifest 显式绑定视频路径、内容 SHA-256、帧数、预处理容量及审核来源。未知标签为 -1，canonical alias 冲突拒绝；候选存在与全库存在分开审核。加载及 collate 均检查内容身份，推理只接收 QueryInputs。

验证：首次缺失模块测试 13 failures；实现后 13 passed。跨 split 的 canonical、source、caption 泄漏分别独立验证。A 基础回归 72 passed。

审核 provenance 是数据提供方的声明；本 loader 不代替人工标注审查。
