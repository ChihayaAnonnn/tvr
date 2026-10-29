# Local query dual-GPU training

用户在2026-10-29明确要求双卡启动当前局部query训练，并确认β=0.2、τ=0.1。
本study单独登记双卡拓扑及资格门，不继承其他study的例外，不改变其他study的四卡规则。
先在GPU0、3执行正式ViT-B/32容量、随机权重、global batch64、12帧、32文本token的FP32资格，检查新query梯度、更新、环境kernel witness、跨rank一致及fresh-model精确恢复。
通过并归档后才预留科学训练；使用原CLIP预训练权重，M16、λ0.2、seed713、5epochs、batch64、lr5e-5。
输入只包含已认证train8246×20和val500×20；不访问test、不启用新KL或Laya。

控制器：`scripts/local_query_campaign.py prepare --mode qualify|train`；随后对返回run目录执行 `execute --run-dir ...`。
prepare冻结phase并通过reserve-phase预留和提交registry；execute验证精确预留、启动sealed command、等待结束，然后关闭日志writer再归档终态/摘要。
任何失败不原地重试；本study不授权自动successor。新输出均在独立数据面，资格及训练的日志分别为run-local `logs/qualification.log`、`logs/train.log`。

环境复用`.aris/compute/local.md`所列tvr与原env-spec，未安装或改动依赖；每rank启动先核验包版本并执行固定seed矩阵乘witness。
