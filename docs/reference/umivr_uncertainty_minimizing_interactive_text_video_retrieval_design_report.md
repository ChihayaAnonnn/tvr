# 方法设计报告：UMIVR（Uncertainty-Minimizing Interactive Text-to-Video Retrieval）

本报告结合 UMIVR 的 [官方论文](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Quantifying_and_Narrowing_the_Unknown_Interactive_Text-to-Video_Retrieval_via_Uncertainty_ICCV_2025_paper.html)、[项目说明](/data2/hxj/project/tvr/research_refs/umivr/README.md:1) 与随附实现编写。这里的“轮”指初始查询后的交互轮次；实验数字均为论文/项目报告值，并非本次本地运行结果。

## 1. 目标问题

- **任务与输入输出**：UMIVR 面向交互式 Text-to-Video Retrieval。输入为初始文本查询 \(x_0\) 和包含 \(N\) 个候选视频的视频库；系统在每轮生成澄清问题、接收回答并形成精化查询 \(x_r\)，输出该轮文本到全部候选视频的相似度矩阵 \(S_r\in\mathbb{R}^{N\times N}\)、视频排名及跨轮指标。
- **具体缺口**：单句文本可能指向多种语义事件（文本歧义）；多个相似视频可能与同一文本有接近分数（文本—视频映射不确定性）；低清晰度或重复帧会使视频表征遗漏关键视觉内容（帧质量不确定性）。一次性的双塔排序无法把这三类信号转换为后续该问什么、如何补足查询的动作。
- **设计目标**：UMIVR 以 Text Ambiguity Score（TAS）量化查询歧义、以 Mapping Uncertainty Score（MUS）量化候选难分性、以 Temporal Quality-guided Frame Sampling（TQFS）准备清晰且语义多样的视频帧；再由三层问题策略驱动问答和查询精化。它是建立在预训练 Video-LLaVA/LanguageBind 上的 training-free 推理框架，而非新增检索训练方法。

## 2. 总体解决思路

~~~text
离线准备
视频库 ──→ TQFS 帧索引 / 视频像素 ──→ LanguageBind 视频编码 ──→ V: N × D
   │                                         │
   └─→ Video-LLaVA 元数据生成 ──→ caption、objects、scene ──→ C、E_C: N × D

在线第 r 轮
x_r ──→ LanguageBind 文本编码 ──→ t_r: 1 × D ──→ 与 V 的全库检索
                                                │
                         C、E_C ──→ TAS ────────┤
                         top-5 分数 ──→ MUS ────┘
                                                ↓
                      Level 0 / 1 / 2 问题生成 → VideoQA 回答 → 查询精化 x_(r+1)
                                                ↓
                      收集 T: N × (R+1) × D → S_r=T_rV^T → 排名、Recall、Hit、BRI
~~~

实现从 [UMIVR 配置](/data2/hxj/project/tvr/research_refs/umivr/retrieval_config/umivr_msrvtt.json:8) 读取 8 帧、10 轮、\(\alpha=0.5\) 的 TAS 阈值和 \(\beta=0.2\) 的 MUS/JSD 阈值；[注册表](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/registry.py:10) 将 <code>ua_multiturn</code> 路由到主循环。该循环先预计算全库视频表征，再对每条文本依次执行“检测不确定性 → 生成问题 → 回答 → 更新查询”，最后才对全部 \(N\) 个视频进行完整排名；用于控制问题的 top-5 候选不是最终结果的截断集合。

## 3. 核心模块设计

### 3.1 预训练双塔检索与视频语义证据库

- **作用**：用同一 Video-LLaVA/LanguageBind 骨干同时建立可检索的文本—视频公共空间，以及供 TAS 和差异性提问使用的全库文本证据。
- **输入与输出**：视频像素批为 \(B\times C\times M\times H\times W\)，其中当前配置 \(M=8\)；视频侧输出并拼接为 \(V\in\mathbb{R}^{N\times D}\)。一条或一批查询输出 \(t\in\mathbb{R}^{B\times D}\)。每个视频还生成 <code>caption</code>、<code>main_objects</code>、<code>scene_type</code>，全部 caption 再编码为 \(E_C\in\mathbb{R}^{N\times D}\)。
- **核心计算**：视频塔的 <code>pooler_output</code> 经 <code>retrieval_video_proj</code> 投影、L2 归一化并以 \(e^{\mathrm{logit\_scale}}\) 缩放；文本塔以 token id 为输入，经 <code>video_tower_text_encoder</code> 和 <code>retrieval_text_proj</code> 得到同维且归一化的表示。当前查询与全库视频的点积给出候选分数；Video-LLaVA 的视觉生成接口为每个视频填充三项文字元数据。
- **阶段与依赖**：视频特征、元数据、caption 语料及其嵌入在交互前准备；文本特征在每一轮重算。\(V\) 交给候选检索和最终全库打分，\(C,E_C\) 交给 TAS，候选元数据交给 Level-1 问题生成。
- **核心代码**：[get_video_features](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_utils.py:569) 产生并缓存 \(V\) 与视频像素；[get_text_features](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:22) 产生 \(t\)；[generate_video_metadata](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:173) 和 [embedding_text_corpus](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:214) 构造证据库；Video-LLaVA 的视频塔、文本塔及两侧检索投影在 [languagebind 初始化](/data2/hxj/project/tvr/research_refs/umivr/videollava/model/multimodal_encoder/languagebind/__init__.py:202) 中接入。
- **模块结果**：该共享空间构成 round 0 的非交互起点。在 MSR-VTT-1k 上，round 0 的 R@1/R@5/R@10 为 43.1/66.1/75.8，Median Rank 为 22.4；后续模块只通过更新查询来改变同一视频库上的文本侧表示。

### 3.2 TQFS：时间质量引导的帧选择

- **作用**：从长视频中选出时间上覆盖事件、视觉上较清晰且语义上分散的帧，为视频表征和视觉问答准备更有效的视觉内容。
- **输入与输出**：输入是原视频与目标帧数 \(K\)（配置为 8）；输出是按时间排序的 \(K\) 个原始帧索引及 IQA JSON 缓存。质量评分后的候选帧还会生成图像嵌入，供语义聚类使用。
- **核心计算**：先以 5 FPS 提取帧，以 Laplacian 方差作为无参考清晰度 \(Q(F_i)\)。把时间轴划为区间 \(I_m\)，各区间保留

  \[
  F_m^*=\arg\max_{F_i\in I_m}Q(F_i).
  \]

  再以图像塔嵌入对候选帧执行 KMeans；每个簇保留质量最高者，最后按原时间顺序输出。因此质量筛选避免模糊帧，聚类避免把多个近似帧一同选入。
- **阶段与依赖**：这是交互前的帧索引构建，不参与参数学习。启动入口会生成或读取 IQA 缓存；在随附 MSR-VTT 配置中 <code>using_llava_preprocess=true</code>，在线数据项将视频路径交给 LanguageBind 的 video processor 形成 8 帧张量，该 processor 以均匀索引解码。因而实现中同时保留“可缓存的 TQFS 帧索引”和“Video-LLaVA 在线视频张量”两个准备产物。
- **核心代码**：[frame_selection](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/dataset.py:107) 串联 5 FPS 候选、质量、图像嵌入和 KMeans；[do_iqa_geration](/data2/hxj/project/tvr/research_refs/umivr/interact_ivr.py:181) 负责缓存；[VideoDataset.__getitem__](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/dataset.py:260) 选择 LLaVA 预处理分支，[LanguageBindVideoProcessor](/data2/hxj/project/tvr/research_refs/umivr/videollava/model/multimodal_encoder/languagebind/video/processing_video.py:72) 完成当前在线解码。
- **模块结果**：组件组合中，TAS+MUS 的 round-5 R@1/Hit@10 为 64.2/94.4，加入 TQFS 后为 65.0/94.8；论文还将 TQFS 接到 VideoCLIP，Text-to-Video 的 R@1/R@10/Mean Rank 从 30.7/71.0/18.1 变为 31.1/72.4/16.1，表明该帧准备策略可服务于不同检索骨干。

### 3.3 TAS：以 caption 语义簇量化文本歧义

- **作用**：区分“文本还描述得过于宽泛”与“文本已足够明确但候选仍难区分”这两种情况，使系统在前一种情况优先提出开放式澄清问题。
- **输入与输出**：输入为当前查询 \(x_r\)、全库 caption 语料 \(C\) 和 caption 嵌入 \(E_C\in\mathbb{R}^{N\times D}\)；输出为标量 TAS。实现取最相似的 top-5 caption，并使用 0.65 的 caption 余弦相似度阈值进行贪心聚类。
- **核心计算**：令各语义簇的相似度质量为 \(p(c_j\mid x_r)\)，论文的语义熵为

  \[
  SE(x_r)=-\sum_jp(c_j\mid x_r)\log p(c_j\mid x_r).
  \]

  熵经归一化后成为 TAS：多个语义簇拥有相近质量时分数更高。随附代码先对 top-5 检索分数 softmax，再汇聚簇概率，计算并 sigmoid 归一化语义熵；默认还以 \(\delta=0.2\) 与查询特异性分数混合，使短、泛的查询得到更高的不确定性。
- **阶段与依赖**：每轮推理都在问题生成前计算；TAS 与 MUS 一起送入层级状态机，其结果决定使用粗粒度、候选差异或细粒度补充问题。
- **核心代码**：[compute_semantic_uncertainty](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/qcs.py:122) 完成查询嵌入、检索、聚类和熵；[get_query_uncertainty](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:279) 在管理器中调用它。
- **模块结果**：TAS 是消融中的基本控制器。仅有 TAS 时，MSR-VTT 消融的 round 1/3/5 R@1 为 51.6/61.0/63.4，Hit@1 为 56.6/67.0/73.0；其作用应与后续 MUS、TQFS 的组合结果一起理解。

### 3.4 MUS 与三层不确定性路由

- **作用**：衡量当前 top-5 视频分数是否集中到一个明确候选；若文本已较具体但多个候选仍彼此接近，就转而提出能够区分候选元数据的提问。
- **输入与输出**：输入是当前查询对全库检索后排名最高的分数 \(s_{1:5}\)；输出为重标定候选分布 \(p\)、归一化 MUS/JSD 标量及离散的 <code>level</code>。只有这 5 项参与 MUS，最终检索仍对全部 \(N\) 个视频打分。
- **核心计算**：代码以 top-5 平均分 \(\bar{s}\) 为截断点，计算

  \[
  p_i=\frac{\max(s_i-\bar{s},0)^2}
            {\sum_j\max(s_j-\bar{s},0)^2}.
  \]

  令 \(q\) 为最大分数位置的 one-hot “确定匹配”分布，计算 \(JSD(p\parallel q)\)，再以理论最大 JSD 归一化为 MUS。数值越高，表示当前候选分布越偏离单一确定候选。

  状态机采用 \(\alpha=0.5,\beta=0.2\)：首轮或 \(\mathrm{TAS}\ge\alpha\) 为 Level 0；\(\mathrm{TAS}<\alpha\) 且 \(\mathrm{MUS}\ge\beta\) 为 Level 1；两者均低为 Level 2。代码还规定 Level 1 的下一轮转入 Level 2，以把候选区分后续接为细节补充。
- **阶段与依赖**：每轮先对 \(t_rV^\top\) 取 top-5，随后计算 MUS；状态会将筛选后候选元数据交给对应的问题生成器。
- **核心代码**：[get_cur_retrieval](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_utils.py:30) 返回 top-5 索引和分数；[matching_score](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/qcs.py:288) 产生 \(p\) 与归一化 JSD；[set_uncertainty_level](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:284) 执行 TAS/MUS 路由。
- **模块结果**：在 TAS 基础上加入 MUS 后，round 1/3/5 R@1 从 51.6/61.0/63.4 变为 52.2/62.1/64.2，Hit@10 从 86.1/91.1/93.7 变为 86.4/92.5/94.4；这是 TAS+MUS 组合的结果。

### 3.5 层级问题、视觉回答与查询精化

- **作用**：把连续的 TAS/MUS 数值变成一次能够增加检索信息的交互，并把回答写回文本查询空间。
- **输入与输出**：输入为当前查询、路由等级及候选元数据；输出为问题 \(q_r\)。视觉回答模块输入 \(q_r\) 和对应视频像素，输出答案 \(a_r\)；查询精化模块输出下一轮文本 \(x_{r+1}\)。中粒度分支还输入由 \(p_i\ge0.02\) 过滤后的候选 <code>caption</code>、<code>main_objects</code>、<code>scene_type</code>。
- **核心计算**：
  - **Level 0** 仅依据宽泛查询提出开放式问题，收集主体、活动或事件；
  - **Level 1** 比较多个候选的元数据，生成 WHAT/WHERE/WHO 等区分性问题；
  - **Level 2** 在已有文本上追问颜色、对象、动作等视觉细节。

  Video-LLaVA 以问题和视频像素生成答案。测试调用将与当前文本同序的配对视频提供给视觉回答函数；该回答同上一查询用 <code>and</code> 拼接。若可用检索 token 少于 10，则先用生成模型压缩已有查询，再拼入本轮答案。
- **阶段与依赖**：仅在交互推理中执行；它依赖 TAS/MUS 产生的等级、视频语义证据库和当前目标视频的像素，输出的 \(x_{r+1}\) 会重新经过文本塔并进入下一轮。
- **核心代码**：[do_caption_generation](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:304) 编排“检索—MUS—问题—回答—更新”；[do_question_generation](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:436) 路由三类问题，[middle_grained_question_generation](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:521)、[coarse_grained_question_generation](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:561) 与 [fine_grained_question_generation](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:598) 生成问题；[generate_answewr](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:489) 和 [get_final_query](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:336) 产生答案和精化查询。
- **模块结果**：完整 TAS+MUS+TQFS 组合在 MSR-VTT 消融中，round 1/3/5 的 R@1 为 52.5/61.3/65.0，Hit@1 为 57.1/68.9/73.9，Hit@10 为 86.7/92.7/94.8。该结果反映的是三种不确定性控制和多轮问答协同后的表现。

### 3.6 全库分数矩阵与跨轮排名

- **作用**：将各轮精化查询转换为可比较的完整 Text-to-Video 排名，并同时衡量单轮检索与交互累计收益。
- **输入与输出**：对 \(N\) 条测试文本收集初始及 \(R\) 轮查询嵌入，得到 \(T\in\mathbb{R}^{N\times(R+1)\times D}\)；结合 \(V\in\mathbb{R}^{N\times D}\)，第 \(r\) 轮输出 \(S_r\in\mathbb{R}^{N\times N}\)。
- **核心计算**：每一轮提取 \(T_r\in\mathbb{R}^{N\times D}\)，计算

  \[
  S_r=T_rV^\top.
  \]

  每一行按降序排序，以同序文本—视频配对的对角元素名次计算 R@1/R@5/R@10、Median Rank、Mean Rank。Hit@k 维护截至当前轮的历史最佳名次；BRI 则编码多轮过程中的名次改进。
- **阶段与依赖**：该模块在一批所有文本完成多轮对话后执行，依赖 3.1 的视频库表示和 3.5 产生的每轮文本表示；它不反向更新任何参数。
- **核心代码**：[ua_multiturn](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:61) 收集 \(N\times(R+1)\times D\) 的文本特征；[compute_metrics_multiturn](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_utils.py:46) 建立 \(S_r\)；[compute_recall_metrics](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_utils.py:80)、[compute_hit_metrics](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_utils.py:99) 和 [compute_bri_metrics](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_utils.py:137) 计算指标。
- **模块结果**：MSR-VTT-1k 上，R@1 随交互从 round 0 的 43.1 提升到 round 3 的 61.3、round 8 的 67.3；项目说明报告 round 10 达到 69.2。

## 4. 端到端训练与推理

### 训练

1. **UMIVR 不新增训练阶段。** 主入口只构造测试数据加载器、加载已训练的 <code>LanguageBind/Video-LLaVA-7B</code>，并进入检索调度。[interact_ivr.py](/data2/hxj/project/tvr/research_refs/umivr/interact_ivr.py:105)
2. 可在交互前准备两类缓存：TQFS 帧索引，以及由 Video-LLaVA 生成的视频元数据、caption 语料和 \(E_C\)。这些都是预训练模型的前向计算结果。
3. 交互主线的特征提取、候选检索、TAS/MUS、问题生成、视觉回答和查询压缩均在推理上下文中执行；配置指定 <code>dialogue_type=ua_multiturn</code>、<code>total_turn=10</code>。[umivr_msrvtt.json](/data2/hxj/project/tvr/research_refs/umivr/retrieval_config/umivr_msrvtt.json:23)
4. **总损失：无。** UMIVR 本身没有优化器、反向传播或新增的检索损失；它复用已训练骨干，通过多轮查询更新改善 \(S_r\)。

### 推理

1. 启动程序加载配置、Video-LLaVA、检索 tokenizer 与视频预处理器，构建视频/文本测试数据加载器，并路由到 <code>ua_multiturn</code>。[interact_ivr.py](/data2/hxj/project/tvr/research_refs/umivr/interact_ivr.py:148)
2. 预计算 \(V\)，准备 caption 证据库 \(C,E_C\)，并令初始查询 \(x_0\) 得到 \(t_0\)。
3. 对第 \(r\) 轮：以当前查询检索 top-5，计算 TAS 与 MUS，选定 Level 0/1/2，生成问题和视觉答案，形成 \(x_{r+1}\)，再编码为 \(t_{r+1}\)。
4. 所有文本完成 \(R\) 轮后，对每轮计算 \(S_r=T_rV^\top\)，输出 R@k、Median/Mean Rank 以及跨轮 Hit@k、BRI。

## 5. 核心代码实现映射

| 设计模块 | 代码入口 | 核心对象/函数 | 输入 → 输出 | 直接职责 |
| --- | --- | --- | --- | --- |
| 配置与调度 | [interact_ivr.py](/data2/hxj/project/tvr/research_refs/umivr/interact_ivr.py:105) | <code>main</code>、<code>assemble_parameters_ivr</code> | JSON、测试集 → UMIVR 参数 | 加载骨干并选择 <code>ua_multiturn</code> |
| 视频/文本检索编码 | [ivr_utils.py](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_utils.py:569) | <code>get_video_features</code>、<code>get_text_features</code> | 视频像素/文本 → \(V:N\times D\)、\(t:B\times D\) | 建立公共检索空间 |
| 视频元数据与 caption 语料 | [ivr_base.py](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:173) | <code>uaAutoDialogueManager</code>、<code>generate_video_metadata</code> | 视频像素 → 三类文字元数据、\(C,E_C\) | 给 TAS 与差异性提问提供证据 |
| TQFS | [dataset.py](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/dataset.py:107) | <code>frame_selection</code> | 视频 → 清晰、多样的 K 帧索引 | 构建帧质量缓存 |
| TAS | [qcs.py](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/qcs.py:122) | <code>compute_semantic_uncertainty</code> | \(x_r,C,E_C\) → TAS 标量 | 量化查询语义歧义 |
| MUS 与状态机 | [qcs.py](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/qcs.py:288) | <code>matching_score</code>、<code>set_uncertainty_level</code> | top-5 分数 → \(p\)、MUS、Level | 量化候选难分性并选择提问粒度 |
| 问答与查询精化 | [ivr_base.py](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_base.py:304) | <code>do_caption_generation</code>、<code>generate_answewr</code> | 查询/元数据/视频 → \(q_r,a_r,x_{r+1}\) | 把交互信息写回检索查询 |
| 全库评价 | [ivr_utils.py](/data2/hxj/project/tvr/research_refs/umivr/retrievalmodel/ivr_utils.py:46) | <code>compute_metrics_multiturn</code> | \(T,V\) → \(S_r:N\times N\) 与排名指标 | 产生完整检索结果和跨轮增益 |

## 6. 实现结果

### 6.1 功能性结果

- **检索表示与分数**：视频库被缓存为 \(V\in\mathbb{R}^{N\times D}\)，每条文本在初始及精化后得到 \(t_r\in\mathbb{R}^{1\times D}\)。所有测试文本聚合为 \(T_r\) 后，\(S_r=T_rV^\top\) 是实际用于完整排序的检索分数；它覆盖全库而非只覆盖控制模块的 top-5。
- **不确定性控制信号**：TAS 是查询—caption 语料的语义簇熵/特异性混合标量，MUS 是 top-5 分数相对 one-hot 确定匹配的归一化 JSD；二者不直接替代检索分数，而是决定下一轮的问题粒度。
- **交互产物**：每轮保存问题、答案和精化查询；这些字符串经过同一文本检索编码器后成为新的 \(t_r\)。TQFS、元数据及 caption 嵌入是为上述流程准备的缓存或视觉输入。

### 6.2 实验结果

- **主结果（MSR-VTT-1k）**：

  | 轮次 | R@1 | R@5 | R@10 | Median Rank | Hit@1 | Hit@10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0 | 43.1 | 66.1 | 75.8 | 22.4 | 43.1 | 75.8 |
  | 3 | 61.3 | 84.1 | 89.0 | 8.1 | 68.9 | 92.7 |
  | 8 | 67.3 | 88.3 | 92.8 | 5.7 | 78.9 | 96.5 |

  项目说明还报告第 10 轮 R@1 为 69.2。[README.md](/data2/hxj/project/tvr/research_refs/umivr/README.md:3)

- **组件消融（MSR-VTT，round 1/3/5）**：表中应按组合解释，不把完整成绩归因于单一模块。

  | 组合 | R@1（1/3/5） | Hit@1（1/3/5） | Hit@10（1/3/5） | BRI@5 |
  | --- | --- | --- | --- | ---: |
  | TAS | 51.6 / 61.0 / 63.4 | 56.6 / 67.0 / 73.0 | 86.1 / 91.1 / 93.7 | 0.69 |
  | TAS + MUS | 52.2 / 62.1 / 64.2 | 57.4 / 68.6 / 72.8 | 86.4 / 92.5 / 94.4 | 0.67 |
  | TAS + MUS + TQFS | 52.5 / 61.3 / 65.0 | 57.1 / 68.9 / 73.9 | 86.7 / 92.7 / 94.8 | 0.67 |

- **关键设计选择**：论文在阈值消融中选择 \((\alpha,\beta)=(0.5,0.2)\)，该组合对应完整模型的 R@1 52.5/61.3/65.0（round 1/3/5），与仓库的 [MSR-VTT 配置](/data2/hxj/project/tvr/research_refs/umivr/retrieval_config/umivr_msrvtt.json:31) 一致。
- **跨数据集结果**：AVSD 的 round 6 R@1/R@10/Hit@1/Hit@10/BRI 为 49.9/82.2/63.3/88.2/1.02；MSVD 的 round 5 为 69.7/94.8/79.3/96.7/0.49；ActivityNet 的 round 5 为 41.8/80.7/50.1/84.1/1.13。

## 7. 设计闭环总结

- **文本语义过宽** → **TAS 对 caption 语义簇分布和查询特异性评分，Level 0 提出开放式问题** → 查询在前几轮获得主体、活动和事件信息；MSR-VTT 上 TAS 基础消融的 round-5 R@1 为 63.4。
- **具体文本仍对应多个相似候选** → **MUS 用 top-5 分数分布选择 Level 1 的候选差异性问题，再进入 Level 2 细节补充** → 问答把候选间的对象、场景和动作差异写回查询；TAS+MUS 的 round-5 R@1 为 64.2。
- **视频关键内容受帧质量与冗余影响** → **TQFS 用时间质量筛选和语义聚类准备视觉内容，并由双塔全库重排** → 完整 TAS+MUS+TQFS 组合的 round-5 R@1/Hit@10 达到 65.0/94.8，主实验的多轮 R@1 进一步从 round 0 的 43.1 提升至 round 8 的 67.3。
