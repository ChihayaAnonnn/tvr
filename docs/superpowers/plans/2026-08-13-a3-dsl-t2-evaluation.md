# A3 DSL T=2 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the validation-selected A3 epoch-index 3 checkpoint once on MSRVTT JSFUSION with fixed Dual Softmax temperature 2.0 and determine whether T2V R@1 exceeds 52.6.

**Architecture:** Run the existing eval-only path in a fresh output directory with the selected checkpoint and all A1/A3 modules instantiated. DSL changes only the directional retrieval matrices; A1/A3 continue to consume detached raw WTI evidence. Preserve the registered no-DSL result and publish this run separately as a post-hoc exploratory result.

**Tech Stack:** Bash, PyTorch 2.1.2, existing UATVR evaluator, four visible A800 GPUs through one evaluation process, MSRVTT JSFUSION.

## Global Constraints

- Use `ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/pytorch_model.bin.3` with expected SHA-256 `2e6f662bc69dda9b880bf882195247cdd1394de6ecb6865c48148cdfb0d85a8c`.
- Set `--DSL True --dsl_temperature 2.0`; do not search temperatures.
- Set `--do_eval --eval_split test`; do not train or load an optimizer.
- Do not pass `--registered_contextual_risk_a3` or `--run_final_test`; this is an independent post-hoc evaluation, not a second registered final-test claim.
- Do not modify the original A3 output directory, candidate selection, acceptance record, or `final_test.json`.
- Write to the unique paths computed from `RUN_STAMP`: `ckpts/ckpt_msrvtt_${RUN_STAMP}_a3_dsl_t2_eval` and `logs/${RUN_DATE}/${RUN_CLOCK}_a3_dsl_t2_eval_msrvtt.log`.
- Treat 52.6 as a fixed historical comparator: greater is an improvement, equal is a reproduction, lower is a regression or evaluation discrepancy.

---

### Task 1: Preflight the immutable evaluation inputs

**Files:**
- Read: `ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/pytorch_model.bin.3`
- Read: `/data2/hxj/project/UATVR/.cache/ViT-B-16.pt`
- Read: `data/generated/msrvtt_trusted_v1/`
- Read: `dataloaders/splits/msrvtt_trusted_v1_seed0.json`

**Interfaces:**
- Consumes: selected A3 checkpoint, CLIP architecture asset, JSFUSION input files, current feature-worktree source.
- Produces: a recorded preflight with exact hashes, clean tracked status, free GPU selection, and a unique output/log pair.

- [ ] **Step 1: Verify source and immutable hashes**

Run:

```bash
git status --short --branch
sha256sum \
  ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/pytorch_model.bin.3 \
  /data2/hxj/project/UATVR/.cache/ViT-B-16.pt
```

Expected: no tracked worktree/index modification; checkpoint hash is `2e6f662bc69dda9b880bf882195247cdd1394de6ecb6865c48148cdfb0d85a8c`; CLIP hash is `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.

- [ ] **Step 2: Verify required data files**

Run:

```bash
test -f /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv
test -f /data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json
test -d /data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps
test -f dataloaders/splits/msrvtt_trusted_v1_seed0.json
test -f data/generated/msrvtt_trusted_v1/train.csv
test -f data/generated/msrvtt_trusted_v1/val.csv
```

Expected: exit status 0 with no missing path.

- [ ] **Step 3: Verify accelerator availability**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
```

Expected: GPUs selected for evaluation have no conflicting compute process and enough free memory. If 0,2,3,4 are busy, stop and wait; do not silently change the registered hardware layout.

---

### Task 2: Execute the fixed DSL evaluation

**Files:**
- Create: `logs/${RUN_DATE}/${RUN_CLOCK}_a3_dsl_t2_eval_msrvtt.log`
- Create: `ckpts/ckpt_msrvtt_${RUN_STAMP}_a3_dsl_t2_eval/log.txt`
- Read: `main_task_retrieval.py`

**Interfaces:**
- Consumes: Task 1 preflight and the existing `main_task_retrieval.py --do_eval` interface.
- Produces: one completed eval-only log containing T2V/V2T retrieval and A1/A3 risk metrics.

- [ ] **Step 1: Reserve unique paths and launch exactly one evaluation**

From the feature worktree, set one timestamp and run:

```bash
set -euo pipefail
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DATE="${RUN_STAMP%%_*}"
RUN_CLOCK="${RUN_STAMP##*_}"
OUTPUT_DIR="$(pwd)/ckpts/ckpt_msrvtt_${RUN_STAMP}_a3_dsl_t2_eval"
LOG_DIR="$(pwd)/logs/${RUN_DATE}"
LOG_FILE="${LOG_DIR}/${RUN_CLOCK}_a3_dsl_t2_eval_msrvtt.log"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0,2,3,4 \
CLIP_CACHE_DIR=/data2/hxj/project/UATVR/.cache \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/xujie/.conda/envs/tvr/bin/torchrun \
  --nproc_per_node=1 --master_addr=127.0.0.9 --master_port=29548 \
  main_task_retrieval.py \
  --do_eval --eval_split test \
  --init_model "$(pwd)/ckpts/ckpt_msrvtt_20260813_031640_contextual_risk_a3/pytorch_model.bin.3" \
  --output_dir "${OUTPUT_DIR}" \
  --use_uatvr_full --use_ada_norm \
  --use_retrieval_risk_sidecar --w_retrieval_risk 0.0 \
  --retrieval_risk_hidden_dim 32 --retrieval_risk_temperature 1.0 \
  --use_contextual_risk_sidecar \
  --contextual_risk_topk 16 \
  --contextual_risk_projection_dim 128 \
  --contextual_risk_hidden_dim 128 \
  --contextual_risk_temperature 1.0 \
  --contextual_risk_bootstrap_samples 1000 \
  --w_contextual_risk_fit 0.25 --w_contextual_risk_rank 0.25 \
  --DSL True --dsl_temperature 2.0 \
  --seed 0 --batch_size_val 16 --num_thread_reader 12 --prefetch_factor 4 \
  --train_csv "$(pwd)/data/generated/msrvtt_trusted_v1/train.csv" \
  --val_csv "$(pwd)/data/generated/msrvtt_trusted_v1/val.csv" \
  --source_train_csv /data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv \
  --test_csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv \
  --split_manifest "$(pwd)/dataloaders/splits/msrvtt_trusted_v1_seed0.json" \
  --data_path /data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json \
  --features_path /data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps/ \
  --datatype msrvtt --expand_msrvtt_sentences \
  --max_words 32 --max_frames 8 --feature_framerate 1 \
  --freeze_layer_num=-1 --slice_framepos 3 \
  --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/16 \
  --clip_layer_norm_precision fp32 \
  --extra_video_cls_num 2 --extra_text_cls_num 2 \
  --experiment_profile hygiene \
  --experiment_desc "Post-hoc A3 winner DSL T=2 evaluation" \
  2>&1 | tee "${LOG_FILE}"
```

Expected: process exits 0; log states `do_train=False`, `do_eval=True`, `eval_split=test`, `Using Dual Softmax evaluation (temperature=2.000)`, and reports both T2V and V2T metrics.

- [ ] **Step 2: Fail closed on runtime defects**

Run:

```bash
rg -n "Traceback|RuntimeError|ValueError|CUDA out of memory|nan|NaN" "${LOG_FILE}" || true
```

Expected: no runtime exception or non-finite diagnostic. On failure, preserve the log and output directory and stop; do not alter temperature or retry with changed model/data settings.

- [ ] **Step 3: Extract the comparison result**

Run:

```bash
rg -n "Using Dual Softmax|T2V \| R@1|V2T \| R@1|contextual risk| risk \|" "${LOG_FILE}"
```

Expected: one fixed-T result block. Classify T2V R@1 strictly as `> 52.6`, `= 52.6`, or `< 52.6` without rounding a lower value upward.

---

### Task 3: Validate and report the exploratory result

**Files:**
- Create: `logs/${RUN_DATE}/analysis_${RUN_CLOCK}_a3_dsl_t2_eval_msrvtt.log.md`
- Create: `docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md`
- Read: `docs/superpowers/reports/2026-08-12-contextual-retrieval-risk-v2-results.md`

**Interfaces:**
- Consumes: Task 2 log, output `log.txt`, selected checkpoint hash, and historical raw/DSL metrics.
- Produces: a log-adjacent Chinese analysis and a concise tracked scientific report.

- [ ] **Step 1: Cross-check both logs and configuration**

Run:

```bash
sha256sum "${LOG_FILE}" "${OUTPUT_DIR}/log.txt"
rg -n "do_train=False|do_eval=True|eval_split=test|Dual Softmax evaluation \(temperature=2.000\)" \
  "${LOG_FILE}" "${OUTPUT_DIR}/log.txt"
```

Expected: both logs identify eval-only JSFUSION and fixed DSL T=2; record both hashes in the report.

- [ ] **Step 2: Write the log-adjacent analysis**

Create `analysis_<log-name>.md` beside the log with these exact sections:

```markdown
# A3 DSL T=2 评估日志分析

## 结论
## 固定实验配置
## T2V/V2T 检索结果
## A1/A3 风险指标
## 与 raw 50.0 和历史 DSL 52.6 的比较
## 协议说明与局限
```

Expected: the analysis explicitly states that A3 does not generate the R@1 change and that the run is post-hoc.

- [ ] **Step 3: Write and verify the scientific report**

Create `docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md` containing checkpoint/log/output hashes, exact metrics, the 52.6 comparison, and any deviations. Then run:

```bash
rg -n "TBD|TODO|Traceback|NaN" \
  docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md \
  "${LOG_FILE}" || true
git diff --check
```

Expected: no placeholder, traceback, non-finite metric, or whitespace error.

- [ ] **Step 4: Commit only the tracked scientific report**

Run:

```bash
git add docs/superpowers/reports/2026-08-13-a3-dsl-t2-results.md
git commit -m "docs: report A3 DSL T2 evaluation"
git status --short --branch
```

Expected: the commit contains exactly one tracked report; checkpoints and logs remain untracked and preserved.
