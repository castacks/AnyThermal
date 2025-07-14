#!/bin/bash
python3 new_benchmark_vpr.py \
    --model_names mmdistill_frozen_dinov2_final mmdistill_frozen_dinov2_final_vlad_64 salad \
    --dataset cart\
    --top_k_vals 1 5 10 \
    --batch_size 16 \
    --save_qual \
    --qual_k 5 \
    --use_faiss_gpu \
    --use_wandb \
    --wandb_project PlaceRecBench \
    --wandb_entity parv \
    --wandb_group ClassificationModelBenchmark \
    --db_q_mode RGB_THERMAL THERMAL_THERMAL THERMAL_RGB RGB_RGB\
    --dataset_splits val train \
    --dist_thresh 15 \
    --exclude_exact_query_in_db \
    --viz_clusters