DATA_PATH=/data2/hxj/data/MSVD

# Log naming: logs/YYYYMMDD/HHMMSS_eval_msvd_zeroshot.log
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"
LOG_DIR="logs/${RUN_DATE}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_TIME}_eval_msvd_zeroshot.log"

CUDA_VISIBLE_DEVICES=2,3 \
    torchrun --nproc_per_node=2 --master_addr=127.0.0.9 --master_port=29509 \
    main_task_retrieval.py \
    --do_eval --num_thread_reader=8 --batch_size_val 128 \
    --data_path ${DATA_PATH}/desc_files \
    --features_path ${DATA_PATH}/YouTubeClips \
    --output_dir ckpts/eval_msvd_zeroshot \
    --datatype msvd \
    --max_words 32 --max_frames 12 \
    --feature_framerate 1 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 \
    --extra_video_cls_num 2 \
    --extra_text_cls_num 2 \
    --n_video_embeddings 7 \
    --n_text_embeddings 7 \
    --uncertainty_text_head text \
    --log_sigma_min -6 \
    --log_sigma_max 6 \
    --DSL True \
    --init_model ckpts/ckpt_msvd_20251223_153240/pytorch_model.bin.0 "$@" 2>&1 | tee ${LOG_FILE}
