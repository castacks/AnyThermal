#!/bin/bash
# --model_names alexnet resnet18 resnet50 vgg16 squeezenet1_1 mixvpr netvlad dinov2_vits14 imagebind salad mmdistill_variable\

python3 benchmark_vpr.py \
    --model_names sgm\
    --dataset_name thermal_day_night \
    --dataset_root /ocean/projects/cis220039p/shared/datasets/ms2_full \
    --top_k_vals 1 5 10 \
    --batch_size 8 \
    --save_qual \
    --qual_k 5 \
    --output_dir ./qualitative_outputs/ms2/change_ratio \
    --use_faiss_gpu \
    --use_wandb \
    --wandb_project PlaceRecBench \
    --wandb_entity parv \
    --wandb_group ClassificationModelBenchmark \
    --db_q_mode RGB_THERMAL \
    --seq "_2021-08-06-16-45-28 _2021-08-06-11-37-46 _2021-08-06-17-21-04 _2021-08-13-16-08-46 _2021-08-13-22-03-03 _2021-08-13-21-58-13"

