# Laya Vision 冻结重排序：validation 精度与效率验证设计

## 1. 目标与固定边界

在原始 UATVR 后接本地 Laya Vision，验证完整 MSRVTT validation T2V R@1 收益及在线延迟、显存、离线缓存成本。只使用认证 validation：500 视频、10,000 查询；始终输出完整500候选排列。冻结两模型，eval/no-grad，FP32、DSL关闭，原有预处理。使用已归档且由 validation 选择的原始 UATVR seed713 epoch05 checkpoint；对照不能换采帧或 batch 版本。单 checkpoint 可行性结果不是独立测试结论。不训练、不搜索、不校准、不访问 test；OpenJev、蒸馏、选择性调用不在本轮。

## 2. 固定模型与重排序规则

原始 UATVR 全库排序 → 固定 Top-10 → 每视频最多4帧与查询输入 Laya → 按 P(match) 重排10项 → 原样接回其余490项。Laya 本地201M权重、原生 processor、FP32、checkpoint现有温度、option_attention=block；一次固定选项排列，无排列集成。

从 UATVR 原有有效采样帧列表，按 floor(linspace(0,n−1,min(4,n))) 取帧；输入对应原始 RGB，不能反归一化224像素输入代替。少于4个不同有效帧使用全部，不复制 padding；按时间顺序通过 images 输入，并说明来自同一视频的时间有序采样。记录实际帧数及重复帧去除。不输入人工caption、视频ID、文件名、排名、分数，查询是唯一文本语义条件。

固定 choice 问题：

> Based only on the sampled frames, does this video match the following retrieval description: {query}?

固定选项顺序：

- match: The sampled visual evidence supports the description.
- not_supported: The sampled visual evidence does not establish the description.

不支持只表示采样证据不足。按 P(match) 降序，完全同分保留原次序；Top-10外顺序不变。不融合、不阈值。非有限、缺失选项、查询截断、完全无法解码使运行失败，不能回退或跳过。只报告T2V，不能转置局部结果声称V2T收益。

## 3. 模块与缓存

从 codex/full-data-baselines 建独立分支，不改变原始评分。增加视觉证据适配器（认证视频、原采样索引→RGB/时间/数量/身份）、Laya评审器（查询/证据→logits/概率/截断/身份）、重排序评估器（全库排序→完整排列/配对指标/资源）。独立配置及manifest驱动入口，不向训练入口加隐式分支。

只持久缓存查询无关视觉token，键绑定视频内容、帧索引、processor、权重及精度哈希。数据面持久存储，在线测量置CPU内存，CPU→GPU传输计时。关闭跨查询语言前缀缓存。查询输出键同时绑定模型、查询、视频；恢复验证完整块身份/哈希，完整块复用，不完整块重算；覆盖10,000查询。隔离并锁定依赖，外部代码/权重内容清单；不修改下载目录，不联网自动换权重。

## 4. 验证与测量

先窄范围CPU测试，再真实通路：跨查询隔离、请求乱序映射；完整候选集合和尾部不变；短视频/padding/解码/截断/非有限/同分；缓存与无缓存FP32 logits最大绝对误差≤1e-5且argmax/排序一致；batch1/8及单进程/分布式分片一致，不放宽断言；中断恢复等同连续执行且无重复；拒绝train/test，不向模型暴露标签。

两臂固定为原始UATVR与Top10/最多4帧/P(match)重排。报告R@1/5/10、MRR、正确→错误及错误→正确。R@10必须一致。来源组配对bootstrap，seed713，10,000次，95%区间；基线与归档核对，不明差异停止生成结论。

- 正向可行性：ΔR@1≥0.5个百分点且区间下界>0。
- 未决：点估计>0但未满足上述条件。
- 无提升/负结果：点估计≤0。

只决定研究价值，不授权后续实验，负结果照常归档。

在线单查询已建库，查询batch1、候选微batch8。同GPU同环境，各臂新进程。query ID哈希固定抽100条validation，各预热5条、3轮交替；CUDA同步计时，报中位数/P95/吞吐/相对倍数。预建原始特征及Laya视觉缓存，各报构建耗时/磁盘/CPU内存。增强计入查询编码、检索、读取/传输、前向及排序。报峰值allocated/reserved、加载后常驻及增量。精度可共享GPU，正式性能须无无关计算，不停止他人进程；无法稳定测量标记未完成。无虚构延迟或显存目标，不凭参数量声称高效。

## 5. 归档、资格门与交付

独立study laya-rerank-validation-v1，experiments控制面及artifacts/<study>/<dataset>/<stage>/<run-id>数据面；遵循reserve→提交→verify→执行→终态归档。旧双卡资格不外溢。新增链路四卡合成资格：正式容量、随机权重、真实推理/缓存、分片合并/数值一致/中断恢复，无优化器或虚假梯度验收。通过并归档前禁止科学GPU；卡数不足等待，不降gate。

软件资格累计≤8 GPU小时、每次≤30分钟；科学/性能累计≤24 GPU小时、每次≤8小时。每类软件失败最多2次有诊断后继；身份改变新run。预算或重复失败终止保留产物，不按R@1调参重跑。

交付代码、测试、固定配置、模型/输入身份清单、资格报告及完整精度/资源报告。逐查询结果/排序/bootstrap外置NPY/NPZ并哈希。软件正确性与科学假设分别报告。

2026-10-23 执行授权：先保存设计，再开发代码、CPU测试及固定配置；四卡资格通过并归档前禁止科学GPU评估。
