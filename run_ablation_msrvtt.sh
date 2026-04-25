#!/usr/bin/env bash
set -e

DATA_PATH=/data2/hxj/data/MSRVTT

# Default to the best checkpoint from logs/20260111/153216_train_msrvtt.log
MODEL_PATH="${MODEL_PATH:-ckpts/ckpt_msrvtt_20260111_153216/pytorch_model.bin.3}"

# Log naming: logs/YYYYMMDD/HHMMSS_ablation_msrvtt.log
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"
LOG_DIR="logs/${RUN_DATE}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TIME}_ablation_msrvtt.log}"

echo "=== Starting MSRVTT Inference-time Ablation ===" | tee -a ${LOG_FILE}
echo "Model: ${MODEL_PATH}" | tee -a ${LOG_FILE}
echo "Val CSV: ${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv" | tee -a ${LOG_FILE}
echo "---------------------------------------------" | tee -a ${LOG_FILE}

modes=("default" "base_only" "query_only" "fixed_avg")

for mode in "${modes[@]}"; do
    echo "[Running Ablation Mode]: ${mode}" | tee -a ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=0 \
        torchrun --nproc_per_node=1 --master_addr=127.0.0.9 --master_port=29519 \
        main_task_retrieval.py \
        --do_eval \
        --output_dir ckpts/ablation_msrvtt_temp \
        --datatype msrvtt \
        --val_csv "${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv" \
        --data_path "${DATA_PATH}/annotation/MSRVTT_v2.json" \
        --features_path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/" \
        --pretrained_clip_name ViT-B/16 \
        --linear_patch 2d \
        --sim_header seqTransf \
        --strategy 2 \
        --extra_video_cls_num 2 \
        --extra_text_cls_num 2 \
        --max_words 32 \
        --max_frames 12 \
        --feature_framerate 1 \
        --batch_size_val 8 \
        --loose_type \
        --slice_framepos 2 \
        --uncertainty_text_head text \
        --log_sigma_min -6 \
        --log_sigma_max 6 \
        --eval_branch_mode ${mode} \
        --init_model ${MODEL_PATH} "$@" >>${LOG_FILE} 2>&1

    # Extract latest Text-to-Video line for quick comparison.
    echo "--- Results for ${mode} ---" | tee -a ${LOG_FILE}
    grep -A 1 "Text-to-Video:" ${LOG_FILE} | tail -n 2 | tee -a ${LOG_FILE}
    echo "---------------------------------------------" | tee -a ${LOG_FILE}
done

echo "Ablation Study Completed. Full log: ${LOG_FILE}"

