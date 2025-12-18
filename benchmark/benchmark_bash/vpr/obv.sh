#!/bin/bash
# sgm: Long-range UAV Thermal Geo-localization with Satellite Imagery - https://xjh19971.github.io/STGL/
# imagebind: ImageBind: One Embedding Space To Bind Them All - https://arxiv.org/abs/2305.05665
# mmdistill_frozen_dinov2:  Pretrained DINOv2 ViT-b14 
# mmdistill_anythermal: AnyThermal - without a VPR head
# salad: Optimal Transport Aggregation for Visual Place Recognition - https://github.com/serizba/salad
# vpr_mmdistill_salad_anythermal_full: SALAD VPR head + Anythermal backbone distilled and trained using all the datasets (Boson, VIVID++, STHEREO, FREIBURG, TartanRGBT)

python3 -m benchmark.benchmark_vpr \
    --model_names sgm imagebind mmdistill_frozen_dinov2 mmdistill_anythermal salad vpr_mmdistill_salad_anythermal_full \
    --dataset odombeyondvision\
    --top_k_vals 1 5 10 \
    --batch_size 16 \
    --save_qual \
    --qual_k 1 \
    --use_faiss_gpu \
    --use_wandb \
    --wandb_project PlaceRecBench \
    --wandb_entity parv \
    --wandb_group ClassificationModelBenchmark \
    --db_q_mode RGB_THERMAL\
    --exclude_exact_query_in_db \
    --viz_clusters \
    --only_same_backbone \
    --local_odom \
    --sequences benchmark/benchmark_bash/vpr/odombeyondvision_seq.txt