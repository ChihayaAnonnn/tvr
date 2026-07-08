可以，结合你现在基于 **UATVR + CLIP** 做不确定性建模和对比学习，我建议你把这些 backbone 分成两类看：**SigLIP/EVA-CLIP 是“CLIP-like 双塔替换”**，迁移成本低；**InternVideo2 是“视频 foundation backbone 替换”**，收益潜力大，但代码改动明显更大。

UATVR 本身是把 text-video lookup 建模成 **distribution matching**，并通过 learnable tokens 聚合多粒度语义，再在概率嵌入空间做采样和 multi-instance contrastive learning；它的官方论文和仓库都在公开版本中给出。([CVF开放获取][1])

---

## 1. SigLIP / SigLIP2 代表作与仓库

| 类别       | 论文 / 项目                                                                                                                          | 仓库 / 权重                                                                         | 和你替换 UATVR-CLIP 的关系                                                                                                                                                            |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 原始模型     | **Sigmoid Loss for Language Image Pre-Training**, ICCV 2023 Oral                                                                 | GitHub: **google-research/big_vision**；HF: **google/siglip-base-patch16-224** 等 | 最像 CLIP 的替换对象。仍是 image-text 双塔，但训练 loss 从 CLIP 的 batch softmax contrastive 换成 pairwise sigmoid loss。([arXiv][2])                                                               |
| 新一代      | **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features**, 2025 | GitHub: **big_vision / README_siglip2**；HF: **google/siglip2 collection**       | 更值得优先试。SigLIP2 官方说在 zero-shot classification、image-text retrieval、VLM visual representation transfer 上都优于对应 SigLIP，并发布 B/L/So400m/g 四档模型。([arXiv][3])                          |
| 视频检索相关代表 | **Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval**, CVPR 2025                                        | GitHub: **yogesh-iitj/Video-ColBERT**                                           | 很适合你参考，因为它也是在 text-video retrieval 上做细粒度 token interaction，而且论文明确把 CLIP/SigLIP 这种 image-text dual encoder 适配到 T2VR。它还引入 dual sigmoid loss，和你做不确定性 + 对比学习的方向比较契合。([CVF开放获取][4]) |

**我的判断：如果你想先做最小改动，SigLIP2 是第一优先级。**
它和 CLIP 一样是 image/text encoder 分离的结构，HF Transformers 文档也说明 SigLIP 使用独立的 image encoder 和 text encoder 产生两个模态表示，只是训练目标变成 pairwise sigmoid loss。([Hugging Face][5])

不过，SigLIP 替换 CLIP 时要注意一个坑：**不要只换 image encoder，最好 text encoder 也一起换**。因为 SigLIP/SigLIP2 的视觉空间和文本空间是成对训练出来的，你如果用 SigLIP image encoder + 原 CLIP text encoder，很可能跨模态空间不一致，导致检索性能不稳定。

---

## 2. EVA-CLIP / EVA-CLIP-18B 代表作与仓库

| 类别     | 论文 / 项目                                                                    | 仓库 / 权重                                                           | 和你替换 UATVR-CLIP 的关系                                                                                      |
| ------ | -------------------------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 原始模型   | **EVA-CLIP: Improved Training Techniques for CLIP at Scale**, 2023         | GitHub: **baaivision/EVA / EVA-CLIP**；HF: **QuanSun/EVA-CLIP**    | 最容易作为 CLIP 的“强 backbone”平替。它仍是 CLIP-style dual encoder，重点是更强的视觉表征、更高效的大规模训练。([arXiv][6])                 |
| 大模型版本  | **EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters**, 2024              | HF: **BAAI/EVA-CLIP-18B**；GitHub 仍在 BAAI EVA 系列中                  | 代表 EVA 系列的极限扩展，但对你做 UATVR 替换未必实用，因为 18B 太重。更现实的是 EVA02-CLIP-B/L/E。([Hugging Face][7])                    |
| 相关增强方向 | **LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation** | GitHub: **microsoft/LLM2CLIP**；HF 有 EVA02 / SigLIP2 相关 checkpoint | 如果你的文本 query 很长、很复杂，这条线值得看。它关注用 LLM 改善 CLIP-style 模型的文本侧表达，官方介绍强调改善长文本、复杂 caption 的跨模态任务能力。([GitHub][8]) |

**我的判断：EVA-CLIP 是最稳妥的第二选择，尤其适合做“backbone-only ablation”。**
EVA-CLIP 官方权重里比较适合你的应该先看：

| 推荐顺序 | 模型                             | 理由                                                       |
| ---- | ------------------------------ | -------------------------------------------------------- |
| 1    | **EVA02_CLIP_L_336_psz14_s6B** | 性能强，参数量仍可控；HF 表中给出 428M 参数、336 分辨率版本。([Hugging Face][9]) |
| 2    | **EVA02_CLIP_B_psz16_s8B**     | 更轻，适合先跑通 pipeline。([Hugging Face][9])                    |
| 3    | **EVA02_CLIP_E_psz14 / plus**  | 性能更强但很重，适合服务器资源充足时做上限实验。([Hugging Face][9])              |

EVA-CLIP 的优势是：**UATVR 原来基于 CLIP 的 frame-token / text-token 逻辑基本不用推倒重写**。你主要改 encoder loading、projection dim、preprocess、tokenizer 和 feature normalization。

---

## 3. InternVideo2 代表作与仓库

| 类别      | 论文 / 项目                                                                                                    | 仓库 / 权重                                                                                         | 和你替换 UATVR-CLIP 的关系                                                                                                                                                                                  |
| ------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 主论文     | **InternVideo2: Scaling Foundation Models for Multimodal Video Understanding**, ECCV 2024 technical report | GitHub: **OpenGVLab/InternVideo / InternVideo2**；HF: **OpenGVLab/InternVideo2 collection**      | 这是视频文本检索里最值得看的 video foundation backbone 之一。它不是简单的 frame-wise CLIP，而是视频模型；论文说它统一 masked video modeling、cross-modal contrastive learning 和 next token prediction，并把 video encoder 扩到 6B。([arXiv][10]) |
| 检索可用模型  | **InternVideo2 Stage2 / InternVideo2-CLIP variants**                                                       | HF: **InternVideo2-Stage2_1B-224p-f4**, **InternVideo2-Stage2_6B**, **InternVideo2-CLIP-1B/6B** | 推荐你优先试 Stage2_1B 或 Stage2_6B 的 frozen feature 方案。官方 model zoo 给了 zero-shot video-text retrieval 脚本和 MSR-VTT、LSMDC、DiDeMo、MSVD、ActivityNet、VATEX 等结果。([GitHub][11])                                   |
| 下游视频大模型 | **InternVideo2-Chat-8B / HD**                                                                              | HF: **OpenGVLab/InternVideo2-Chat-8B**                                                          | 不一定适合直接做 retrieval backbone，但能看 InternVideo2 作为 video encoder 接 LLM/VideoBLIP 的方式。([Hugging Face][12])                                                                                               |
| 延伸版本    | **InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling**, 2025                       | GitHub: **InternVideo2.5**；HF: **OpenGVLab/InternVideo2_5_Chat_8B**                             | 适合长视频理解，不是你第一阶段替换 UATVR 的首选，但如果后面做长视频检索/事件级检索，可以跟进。([Hugging Face][13])                                                                                                                              |

**InternVideo2 官方 model zoo 里最值得注意的一点：Stage2 系列的 zero-shot video-text retrieval 很强。**
例如官方表中，InternVideo2_s2-1B 在 MSR-VTT 上给出 T2V 51.9、V2T 50.9；InternVideo2_s2-6B 给出 T2V 55.9、V2T 53.7。([GitHub][11]) 这说明它作为视频检索 backbone 的起点，比普通 frame-wise CLIP 更有潜力。

但它和 SigLIP/EVA-CLIP 最大区别是：**它不是简单替换 image encoder，而是替换整个 video representation pipeline**。所以我建议你不要一上来 end-to-end 改 UATVR，而是先 frozen feature。

---

## 4. 针对你当前 UATVR 代码，推荐替换路线

我会按这个顺序做，比较稳：

### 路线 A：EVA-CLIP 替换 CLIP，最低风险

先把 UATVR 里的 CLIP visual/text encoder 换成 **EVA02-CLIP-L/14 或 EVA02-CLIP-L/14-336**。

保留：

```text
frame sampling
text tokenization
DSA / learnable tokens
DUA uncertainty distribution
multi-instance contrastive loss
```

只改：

```text
CLIP encoder -> EVA-CLIP encoder
CLIP preprocess -> EVA preprocess
CLIP tokenizer -> EVA/OpenCLIP tokenizer
projection dim -> 对齐 UATVR hidden dim
```

这条路线最适合写论文 ablation：

```text
UATVR-CLIP
UATVR-EVA-B
UATVR-EVA-L
UATVR-EVA-L-336
```

这样能清楚证明：你的不确定性模块不是只吃 CLIP 红利，换强 backbone 后仍然有效。

---

### 路线 B：SigLIP2 替换 CLIP，论文味更强

SigLIP2 比 EVA-CLIP 更适合你做“loss / uncertainty / contrastive learning”方面的文章，因为它本身就是从 loss 角度改 CLIP。原始 SigLIP 论文提出 pairwise sigmoid loss，不需要对 batch 内所有 image-text pair 做全局 softmax normalization。([arXiv][2])

你的 UATVR 可以做两个版本：

```text
版本 1：
SigLIP2 backbone + UATVR 原 multi-instance contrastive loss

版本 2：
SigLIP2 backbone + multi-instance sigmoid loss / BCE matching loss
```

第二个版本更有研究点。你可以把 UATVR 的概率原型采样从：

```text
sampled text prototypes vs sampled video prototypes
↓
InfoNCE / softmax contrastive
```

改成：

```text
sampled text prototypes vs sampled video prototypes
↓
pairwise sigmoid matching
```

这会自然形成一个题目方向：

```text
Uncertainty-aware Sigmoid Distribution Matching for Text-Video Retrieval
```

---

### 路线 C：InternVideo2 作为 frozen video-text feature，再接 UATVR 不确定性头

这是收益潜力最大的一条，但工程风险也最大。

第一阶段不要动 InternVideo2 内部参数，先做：

```text
video -> InternVideo2 video embedding / token features
text  -> InternVideo2 text embedding / token features
↓
projection layer
↓
UATVR DSA / DUA
↓
distribution matching + contrastive learning
```

也就是把 UATVR 变成：

```text
Backbone: InternVideo2 frozen
Head: UATVR uncertainty-adaptive matching
```

这样好处是：

```text
1. 训练成本低
2. 不容易把 InternVideo2 预训练空间训坏
3. 更容易定位你的不确定性模块是否仍然有效
```

等 frozen 版本有收益后，再做：

```text
unfreeze projection
unfreeze temporal adapter
unfreeze last N layers
LoRA / adapter tuning
```

---

## 5. 我建议你最终实验表这样设计

你现在做的是“不确定性 + 对比学习”，所以实验表不要只报 backbone，更要证明 uncertainty module 在强 backbone 上仍然有效。

| 实验组      | Backbone                  | 是否保留 UATVR 不确定性 | Loss                        | 目的                             |
| -------- | ------------------------- | --------------: | --------------------------- | ------------------------------ |
| Baseline | CLIP                      |               否 | InfoNCE                     | 原始普通对比学习                       |
| UATVR    | CLIP                      |               是 | MI contrastive              | 复现原论文                          |
| Ours-A   | EVA-CLIP-L                |               否 | InfoNCE                     | 看 backbone 单独提升                |
| Ours-B   | EVA-CLIP-L                |               是 | MI contrastive              | 看不确定性是否叠加有效                    |
| Ours-C   | SigLIP2-L/SO400M          |               是 | MI contrastive              | 看 SigLIP2 表征是否更强               |
| Ours-D   | SigLIP2-L/SO400M          |               是 | MI sigmoid                  | 你的方法创新点                        |
| Ours-E   | InternVideo2-s2-1B frozen |               否 | InfoNCE                     | 视频 foundation feature baseline |
| Ours-F   | InternVideo2-s2-1B frozen |               是 | MI contrastive / MI sigmoid | 最强版本候选                         |

如果资源有限，我会先做：

```text
CLIP-B/32
EVA02-CLIP-L/14-336
SigLIP2-L/384
InternVideo2-s2-1B frozen
```

---

## 6. 最推荐你先读/拉的清单

**第一批必读：**

1. **UATVR: Uncertainty-Adaptive Text-Video Retrieval**，你的当前基线。([CVF开放获取][1])
2. **Sigmoid Loss for Language Image Pre-Training / SigLIP**，理解 sigmoid loss 如何替代 CLIP softmax contrastive。([arXiv][2])
3. **SigLIP 2**，当前更值得实际替换的 SigLIP 版本。([arXiv][3])
4. **EVA-CLIP**，最稳的 CLIP-like backbone 替换。([arXiv][6])
5. **InternVideo2**，如果你想从 frame-wise CLIP 升级到真正 video foundation model。([arXiv][10])
6. **Video-ColBERT**，很适合参考它如何把 CLIP/SigLIP 这种双塔模型扩展到细粒度 text-video retrieval。([CVF开放获取][4])

**第一批该拉的仓库 / 模型：**

| 用途                    | 名称                                                              |
| --------------------- | --------------------------------------------------------------- |
| UATVR 原始代码            | **bofang98/UATVR** ([GitHub][14])                               |
| SigLIP / SigLIP2 官方代码 | **google-research/big_vision** ([GitHub][15])                   |
| SigLIP2 权重            | **google/siglip2 Hugging Face collection** ([Hugging Face][16]) |
| EVA-CLIP 官方代码         | **baaivision/EVA / EVA-CLIP** ([GitHub][17])                    |
| EVA-CLIP 权重           | **QuanSun/EVA-CLIP**, **BAAI/EVA-CLIP-18B** ([Hugging Face][9]) |
| InternVideo2 官方代码     | **OpenGVLab/InternVideo** ([GitHub][18])                        |
| InternVideo2 权重       | **OpenGVLab/InternVideo2 collection** ([Hugging Face][19])      |
| InternVideo2 检索/模型表   | **InternVideo2/multi_modality/MODEL_ZOO.md** ([GitHub][11])     |
| Video-ColBERT 参考实现    | **yogesh-iitj/Video-ColBERT** ([GitHub][20])                    |

---

我的建议很明确：**先 EVA-CLIP-L/336 跑通，再 SigLIP2-L/384 做方法创新，最后 InternVideo2-s2-1B frozen 做强 backbone 对照。**
这样你的实验会非常清楚：从“CLIP baseline”到“强 CLIP-like backbone”再到“video foundation backbone”，每一步都能解释收益来源。

[1]: https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_UATVR_Uncertainty-Adaptive_Text-Video_Retrieval_ICCV_2023_paper.pdf "UATVR: Uncertainty-Adaptive Text-Video Retrieval"
[2]: https://arxiv.org/abs/2303.15343 "[2303.15343] Sigmoid Loss for Language Image Pre-Training"
[3]: https://arxiv.org/abs/2502.14786 "[2502.14786] SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features"
[4]: https://openaccess.thecvf.com/content/CVPR2025/papers/Reddy_Video-ColBERT_Contextualized_Late_Interaction_for_Text-to-Video_Retrieval_CVPR_2025_paper.pdf "Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval"
[5]: https://huggingface.co/docs/transformers/en/model_doc/siglip "SigLIP · Hugging Face"
[6]: https://arxiv.org/abs/2303.15389 "[2303.15389] EVA-CLIP: Improved Training Techniques for CLIP at Scale"
[7]: https://huggingface.co/BAAI/EVA-CLIP-18B "BAAI/EVA-CLIP-18B · Hugging Face"
[8]: https://github.com/microsoft/LLM2CLIP "GitHub - microsoft/LLM2CLIP: LLM2CLIP significantly improves already state-of-the-art CLIP models. · GitHub"
[9]: https://huggingface.co/QuanSun/EVA-CLIP "QuanSun/EVA-CLIP · Hugging Face"
[10]: https://arxiv.org/abs/2403.15377 "[2403.15377] InternVideo2: Scaling Foundation Models for Multimodal Video Understanding"
[11]: https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/MODEL_ZOO.md "InternVideo/InternVideo2/multi_modality/MODEL_ZOO.md at main · OpenGVLab/InternVideo · GitHub"
[12]: https://huggingface.co/OpenGVLab/InternVideo2-Chat-8B "OpenGVLab/InternVideo2-Chat-8B · Hugging Face"
[13]: https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B "OpenGVLab/InternVideo2_5_Chat_8B · Hugging Face"
[14]: https://github.com/bofang98/UATVR "GitHub - bofang98/UATVR: [ICCV'23] UATVR: Uncertainty-Adaptive Text-Video Retrieval · GitHub"
[15]: https://github.com/google-research/big_vision "GitHub - google-research/big_vision: Official codebase used to develop Vision Transformer, SigLIP, MLP-Mixer, LiT and more. · GitHub"
[16]: https://huggingface.co/collections/google/siglip2 "SigLIP2 - a google Collection"
[17]: https://github.com/baaivision/EVA/blob/master/EVA-CLIP/README.md "EVA/EVA-CLIP/README.md at master · baaivision/EVA · GitHub"
[18]: https://github.com/opengvlab/internvideo "GitHub - OpenGVLab/InternVideo: [ECCV2024] Video Foundation Models & Data for Multimodal Understanding · GitHub"
[19]: https://huggingface.co/collections/OpenGVLab/internvideo2 "InternVideo2 - a OpenGVLab Collection"
[20]: https://github.com/yogesh-iitj/Video-ColBERT "GitHub - yogesh-iitj/Video-ColBERT: Implementation of Video-ColBERT · GitHub"
