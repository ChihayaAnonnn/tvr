# B 阶段实现验证

范围：新 semantic_retrieval 系统、合成 CPU 验证。不启动科学实验，不触碰 sealed test。

## B1

原始 episode manifest 显式绑定视频路径、内容 SHA-256、帧数、预处理容量及审核来源。未知标签为 -1，canonical alias 冲突拒绝；候选存在与全库存在分开审核。加载及 collate 均检查内容身份，推理只接收 QueryInputs。

验证：首次缺失模块测试 13 failures；实现后 13 passed。跨 split 的 canonical、source、caption 泄漏分别独立验证。A 基础回归 72 passed。

审核 provenance 是数据提供方的声明；本 loader 不代替人工标注审查。

## B2–B7

显式 CLIP 本地权重 + 时序 Transformer + 独立粗投影组成新模型。requested pair 共享投影同时供匹配、dustbin attention 和候选存在头使用。分别用 pair/existence 实际 optimizer step 验证排序输出改变。索引预先 canonical 去重，缓存存完整帧/pooled/mask，版本和引用哈希不符即拒绝。策略的 K 严格递增且有预算上限。

Trainer 绑定完整 manifest 配置及输入哈希；生产 AdamW 与显式 scheduler；全 unknown 跳过 step。新 checkpoint 保存模型、optimizer、scheduler、逐 rank RNG，二进制 SHA 及配置身份严格校验。单进程恢复后的下一步与原训练完全一致。双 rank Gloo 将未审核跨 rank pair 保持 unknown，各目标按全局有效计数归一化；用 SGD 参数差核对单进程全局参考，避免 Adam 对接近零的浮点梯度放大。该验证不代替 GPU 四卡资格。

已验证：B2 4、B3 4、B4 5、B5 2、B6 6 个测试；B7 与 A runtime 回归 17 passed。首次双 rank 测试定位 DDP forward 外粗投影，已纳入统一 forward 修复。所有数值仅合成 CPU fixture。
