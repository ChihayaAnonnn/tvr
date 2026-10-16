# 单因素 validation 对照：2026-10-16 冻结协议

用户授权“采帧策略与 batch 组成的单因素 validation 对照”。这是现有 full-data-retrieval-v1 baseline 诊断扩展，使用其已批准双卡、固定全局 batch 和来源隔离协议；不重开失败研究、不外借其他 study 授权。独立登记新 phase/run，不修改历史模型或运行记录。此前已观察六 baseline 的 test；本轮为后续 validation 探索，不能包装成未见过 test 的确认性研究。

两方法 UATVR/CLIP4Clip × 三臂 control/temporal_offset/caption_shuffle，先固定 seed713，全部五轮。重新训练同源码 control，旧结果仅背景参考。每方法两项单因素效应；不运行组合臂，不改损失、学习率、模型、精度、全局 batch64/128 或源隔离的8246视频/164920caption；validation保持500视频/10000caption、确定性采帧、FP32、DSL=false。每个epoch完整覆盖训练caption，保留相同尾批，更新数与LR调度一致。

- control：当前 unique-video caption batch＋确定性原生FPS采帧。
- temporal_offset：仅训练采帧增加时间相位。独立SHA256(seed,epoch,query_id)派生NumPy RNG，取0..28离散相位，映射为 floor(phase * floor(native_fps)/30) 帧，在原有每秒采样索引上平移，末端夹到最后有效帧。保留原有采样数量和12帧上限。按query独立、同query同epoch可精确重放，不依赖rank、worker调度或模型RNG。此为受控时间偏移实验，**不是官方30fps JPEG链路完整复现**；不同时改变解码、JPEG压缩或短视频帧数。
- caption_shuffle：仅把 unique-video轮转改为全caption随机排列再按相同batch大小分块。允许同视频不同caption成为官方对角loss的负例；这是被测因素，故保留对角CE/MIL定义，不改成多正例loss。不是DistributedSampler逐位复刻：固定尾批保留及分rank规则以隔离组批因素。

接口：FullDataSettings.factor_arm 为严格枚举；缺省保持旧manifest字节结构和原行为；arm进入manifest/checkpoint recipe身份，旧checkpoint不得混用。augmentation仅可用于train；验证拒绝启用。解码失败、非有限loss/gradient、输入身份或恢复不一致均失败关闭。每epoch重建loader前设置epoch，恢复按已校验epoch游标重放。

验证：先行为测试，真实视频解码测试；真实双Gloo rank重复视频梯度对serial；全full_data回归；匹配新增源码的正式容量双CUDA Q0（重复视频、真实前后向、精确恢复、数值/梯度与排名检查）通过并归档后才训练。新增采帧索引与真实解码由CPU行为测试验证；GPU Q0用合成数据且不声称验证真实数据预处理。

预算沿用已注册科学每run86400秒、study累计512 GPU小时、软件每run1800秒累计8 GPU小时；本轮最多六个成功科学配置（max_scientific_runs计成功配置数；失败/取消尝试仍全部计入累计预算），初始最坏预留288 GPU小时，按controller全历史累计守卫；不自动扩seed或调参。软件失败最多两个诊断后新身份successor，历史保留。资源等待最多21600秒，共享两卡保留60GiB上限＋8GiB余量，绝不终止外部作业。新代码先完整测试/提交，再Q0，再按逻辑phase预留提交。worker期间无跟踪文件修改。

主要终点：固定epoch5的validation T2V R@1变化，V2T/R@5/R@10为次要；earliest-best validation T2V checkpoint作为次要选模结果，两者均报告。各臂训练轨迹、失败、资源完整披露。单seed只作机制诊断，不证明跨seed稳健或论文主表提升。后续确认需另行预注册，不根据本轮结果静默扩展配置。

实现顺序：枚举/采帧/组批及必要测试 → typed manifest/拒绝守卫/训练集成 →同源码控制与有限campaign → 独立审查 →提交并Q0 →六run验证与归档 →逐方法paired对照报告。主线由本会话原生执行，独立代理仅最终审查。

## 启动前验证记录

2026-10-16：full_data 回归61 passed（73.45秒）；独立审查发现的并发重复预留和报告身份认证缺口修复后，受影响factor/authorization/campaign测试18 passed（2.70秒）。独立只读复核无遗留阻断项；成功配置上限文义已澄清。命令：`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. /home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/semantic_retrieval/full_data`；修复后窄回归为该目录的test_factors.py、test_authorization.py、test_campaign.py。

原生入口：`/home/xujie/.conda/envs/tvr/bin/python scripts/run_validation_factors.py`。入口持全程互斥锁，逐phase预留、提交、执行、终态归档；保留已成功配置并拒绝重复，失败需要定位后新身份恢复。
