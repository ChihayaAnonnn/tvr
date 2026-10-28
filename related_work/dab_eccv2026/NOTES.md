# DAB — Distribution-Alignment Bridge for Uncertainty-Aware Text-to-Video Retrieval

- arXiv **2607.20984v1**, submitted 2026-07-23, comments field says **ECCV 2026**
- Kyeongmo Chae, Jihoon Lee, Sangtae Ahn (no affiliation on the arXiv page)
- PDF: `dab.pdf` / extracted text: `dab.txt`
- **Code: does not exist.** No GitHub link on the abs page, and no URL of any kind
  anywhere in the paper body (`grep -oE "https?://" dab.txt` → empty).

## Relationship to UATVR: it is NOT built on UATVR

DAB **builds on X-Pool** (CVPR'22). Implementation Details, §4.1: *"We build on X-Pool [7]."*

UATVR appears in exactly two roles:

1. **A criticized prior method** (§2.3): *"UATVR [6] extends this idea to TVR, allowing
   variance to capture semantic diversity. However, UATVR relies on sampled prototypes
   to compute similarity, which partially collapses probabilistic representations into
   deterministic comparisons."*
   → DAB's entire selling point is being **sampling-free**.
2. **A comparison row** in Tables 1/2/3 (cited as ICCV'23, i.e. the original Fang et al.
   paper, not this repo's fork).

§4.2: *"Compared to probabilistic UATVR [6] and generative DiffusionRet [13], which rely
on stochastic sampling, DAB's deterministic, sampling-free distribution alignment produces
both stable convergence and strong generalization."*

## Method

Backbone: CLIP ViT-B/32 and ViT-B/16, F=12 frames, d=512.

1. **Text-conditioned frame pooling** (from X-Pool): text is Query, frames are Keys/Values
   → `v_c ∈ R^{B×B×D}`, where `v_c^{(i,j)}` = video j pooled conditioned on text i.
2. **Probabilistic Head** (shared by both modalities): L2-normalize, then two
   LayerNorm+Linear heads produce `μ` and `log σ²`. `log σ²` bias init **-2.0**
   (low initial uncertainty), clipped to **[-6, 2]**.
3. **Token-Based Uncertainty Head** (video only): frame tokens → single-head self-attention
   → mean-pool → two linear projections of the pooled feature and of `v_c`, residual sum
   → `log σ²_tok`. Final video variance = average of base and token variance.
4. **Distribution Bridge**: sinusoidal time embedding + `DriftDistMLP` taking
   `[μ_t, σ_t, μ_v, σ_v, e_t]` → predicts `Δμ_t, Δlog σ²_t`. Output heads **zero-initialized**.
   Update `μ_{t-1} = μ_t − χ_t·Δμ_t` (same for log σ²). Step size `χ_t` from a linear
   β-schedule, β ∈ [1e-4, 2e-2], normalized to **[0.2, 0.6]**. N=1000 total steps,
   **truncated to T'=32**. No noise injected — fully deterministic.
5. **PAC loss**: cost matrix `C_ij = D_KL(Q_{v_j} ‖ P̂_{t_i})` (closed-form diagonal-Gaussian KL),
   bidirectional softmax over rows and columns. Explicitly **not** symmetric KL or JS.

Training: AdamW, wd 0.02, warmup 0.1, lr **1e-6** (CLIP) / **1e-4** (prob + bridge modules),
batch **32**, **10 epochs**, seed 24, dropout 0.3 (MSRVTT/VATEX) 0.4 (MSVD),
**single NVIDIA A6000**. Inference chunk sizes 128/512.

## Reported results (T2V)

MSR-VTT 1K-A, ViT-B/32 — DAB **56.2 / 85.4 / 92.4**, MnR **4.1**, RSum 234.0
vs DITS (NeurIPS'24) 51.9 / 75.7 / 84.6, MnR 11.6 ; UATVR (ICCV'23) 47.5 / 73.9 / 83.5, MnR 12.9

MSVD ViT-B/32 — 54.5 / 87.8 / 93.9, MnR 3.4 (UATVR 46.0 / 76.3 / 85.1)
VATEX ViT-B/16 — 70.7 / 95.7 / 98.3, MnR 1.9 (UATVR B/32 61.3 / 91.0 / 95.6)

They explicitly deny post-processing: *"These gains are achieved without post-processing
techniques such as inverted softmax or dual normalization."*

## Critical reading — three things that do not add up

### 1. The gain shape is wrong for a "better model"
R@1 +4.3 over DITS, but R@5 **+9.7**, R@10 **+7.8**, and MnR 11.6 → 4.1 (**−64%**).
A genuinely stronger matcher usually moves R@1 at least as much as R@5/R@10. This shape —
small R@1 gain, huge R@5/R@10 gain, collapsed MnR — is the signature of a change in the
**global score distribution**, not of better per-pair matching. They rule out DSL/inverted
softmax explicitly, so the mechanism must be inside the score function itself.

### 2. The ablation says each component alone HURTS R@1 (Table 5a, 5 epochs)

| variant            | R@1  | R@5  | R@10 | MnR  |
|--------------------|------|------|------|------|
| Baseline (X-Pool)  | 46.9 | 72.8 | 82.2 | 14.3 |
| w/ Prob. only      | **44.3** | 76.6 | 87.2 | 7.4  |
| w/ Bridge only     | **43.2** | 76.6 | 86.3 | 8.5  |
| DAB (both)         | 52.6 | 82.7 | 90.8 | 5.3  |

Both components **individually reduce R@1 below baseline** (46.9 → 44.3 / 43.2), yet together
give +5.7. Strongly non-additive in a way the paper does not explain.

Note what happens the moment the probabilistic head is added: R@1 **down**, but R@5 +3.8,
R@10 +5.0, MnR **halved** (14.3 → 7.4). That is not "better matching" — that is the
ranking becoming globally smoother.

### 3. Hypothesis: the gain is largely implicit hubness correction, not uncertainty
The Gaussian KL cost contains variance terms (`σ_P²/σ_Q²`, `log σ_Q²/σ_P²`) that depend on the
**candidate alone**, independent of the query. That is a per-candidate additive bias on every
score — mathematically the same family of correction as inverted softmax / QB-Norm / Sinkhorn
hubness reduction. It would suppress hub videos and promote never-retrieved ones, which is
exactly what crushes MnR and lifts R@5/R@10 while doing little for R@1.

If true, the honest description of DAB is "a learned hubness correction parameterized as a
variance head", not "uncertainty-aware retrieval". **This is testable**: ablate the variance
terms out of the KL (keep only the Mahalanobis mean term), or measure the k-occurrence
distribution / hubness skewness before and after.

### Also: not complexity-comparable to the dual-tower baselines
The drift network takes `μ_v, σ_v` as input (Eq. 7), so refining the text distribution requires
knowing **which candidate** it is being scored against. Scoring is therefore per-(query,
candidate) pair over T'=32 refinement steps — cross-encoder-class cost, compared against
dual-tower rows (CLIP4Clip, X-Pool, UATVR) in the same table. Confirmed by Table 4: FLOPs
scale linearly with T' (69.7T → 557.9T for T'=8 → 64). No retrieval-latency-vs-baseline
comparison is reported.

## What is actually useful to us (§4.4)

This is the most relevant section and it **independently corroborates our SAP negative result**.

They use the **top1−top2 KL margin** as a confidence signal for selective prediction:
- R@1 rises 56.2 → **85.6** at 25% coverage, → **94.0** at 10% coverage
- AURC 0.429 → 0.256 (R@1); 0.144 → 0.082 (R@5)

And critically, their own sentence: *"DAB's uncertainty signal emerges from the bridge-induced
distributional gap rather than **from text variance alone**."*

That is the same finding as ours from the other side: the **learned variance head by itself is
not the useful signal — the score-geometry margin is.** We showed encoder-internal evidential
uncertainty (SAP `u_mode`, Dirichlet evidence) is at chance for failure prediction (AUROC
0.46–0.54) while score geometry gives 0.72–0.88. An ECCV paper arguing the opposite thesis
quietly reports the same thing.

Table 6 (variance asymmetry: video σ² = 5.87× text σ²) is the only place the learned variance
is shown to be meaningful on its own, and it is a descriptive statistic, not a predictive result.
