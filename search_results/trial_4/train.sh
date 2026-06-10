#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=/data2/hxj/data/MSRVTT
RUN_ID="search_4_20260530_021006"
OUTPUT_DIR="ckpts/ckpt_msrvtt_search_4_20260530_021006"

CUDA_VISIBLE_DEVICES="1,2" \
    torchrun --nproc_per_node=2 --master_addr=127.0.0.9 --master_port=34337 \
    main_task_retrieval.py \
    --do_train --num_thread_reader=8 --epochs=3 \
    --batch_size=256 --gradient_accumulation_steps=2 --n_display=50 \
    --train_csv "${DATA_PATH}/csv/MSRVTT_train.9k.csv" \
    --val_csv "${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv" \
    --data_path "${DATA_PATH}/annotation/MSRVTT_v2.json" \
    --features_path "${DATA_PATH}/videos/compressed_videos/msrvtt_224_12fps/" \
    --output_dir "${OUTPUT_DIR}" \
    --lr 1e-4 --max_words 32 --max_frames 8 --batch_size_val 16 \
    --datatype msrvtt --expand_msrvtt_sentences \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 3 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --strategy 2 \
    --pretrained_clip_name ViT-B/16 \
    --extra_video_cls_num 2 \
    --extra_text_cls_num 2 \
    --n_video_embeddings 7 \
    --n_text_embeddings 7 \
    --mamba_lr_ratio 0.1 \
    --uncertainty_text_head text \
    --log_sigma_min -1.5 \
    --log_sigma_max 4.0 \
    --w_mil 0.01 \
    --w_evidential 0.01 \
    --w_neg_reg 0.01 \
    --w_orth 0.1 \
    --w_uncertainty_reg 0.001 \
    --gate_log_interval 100 \
    --log_moe_weights \
    --fusion_mode prob_mos \
    --w_query_sim 0.5 \
    --fusion_temperature 1.5 \
    --rope_mode 2d \
    --use_ada_norm \
    --anneal_warmup_epochs 2 \
    --experiment_desc "search_trial_4"
