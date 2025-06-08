#!/bin/bash
# --model_names alexnet resnet18 resnet50 vgg16 squeezenet1_1 mixvpr netvlad dinov2_vits14 imagebind salad mmdistill_variable\
    # --model_names mixvpr salad dinov2_vitb14_variable dinov2_vitb14_fixed mmdistill_dinov2_variable_dinov2_vitb14 mmdistill_dinov2_fixed_dinov2_vitb14 imagebind\

python3 benchmark_vpr.py \
    --model_names netvlad_mmdistill_dinov2_cart\
    --dataset_name cart \
    --top_k_vals 1 5 10 \
    --batch_size 16 \
    --save_qual \
    --qual_k 5 \
    --output_dir ./qualitative_outputs/cart \
    --use_faiss_gpu \
    --use_wandb \
    --wandb_project PlaceRecBench \
    --wandb_entity parv \
    --wandb_group ClassificationModelBenchmark \
    --db_q_mode RGB_THERMAL \
    --exclude_exact_query_in_db \
    --seq "2022-12-20-12-16-02 2022-12-20-12-48-59 2022-12-20-13-37-37"
