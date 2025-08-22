#!/bin/bash
    # --model_names  mmdistill_cart_only_contrastive_all_final_vlad_64 mmdistill_cart_only_contrastive_all_final mmdistill_ms2_only_contrastive_all_final mmdistill_ms2_only_contrastive_all_final_vlad_64 mmdistill_combine_both_cosine_all_final_vlad_64 mmdistill_combine_both_cosine_all_final mmdistill_combine_contrastive_all_final mmdistill_combine_contrastive_all_final_vlad_64 \
    # --model_names mmdistill_combine_global_contrastive_salad_backbone_50_final mmdistill_combine_global_contrastive_salad_backbone_50_final_vlad_64 mmdistill_combine_global_contrastive_salad_backbone_25_final mmdistill_combine_global_contrastive_salad_backbone_25_final_vlad_64 mmdistill_combine_global_contrastive_salad_backbone_10_final mmdistill_combine_global_contrastive_salad_backbone_10_final_vlad_64 \
    # --model_names mmdistill_frozen_dinov2_final mmdistill_frozen_dinov2_final_vlad_64 mmdistill_frozen_dinov2_final_globalvlad_64 mmdistill_frozen_salad_final mmdistill_frozen_salad_final_vlad_64 salad mmdistill_frozen_salad_final_globalvlad_64  mmdistill_combine_global_contrastive_all_equal_10_final mmdistill_combine_global_contrastive_all_equal_10_final_vlad_64 mmdistill_combine_global_contrastive_all_equal_10_final_globalvlad_64  mmdistill_combine_global_contrastive_salad_backbone_50_final mmdistill_combine_global_contrastive_salad_backbone_50_final_vlad_64 vpr_mmdistill_salad_frozen_normal_backbone vpr_mmdistill_salad_normal_salad_backbone vpr_mmdistill_salad_frozen_normal_backbone_32 vpr_mmdistill_salad_normal_salad_backbone_32\
    # --model_names salad vpr_mmdistill_salad_frozen_ms2_freiburg_sthereo_boson vpr_mmdistill_salad_ms2 vpr_mmdistill_salad_ms2_freiburg vpr_mmdistill_salad_ms2_freiburg_sthereo vpr_mmdistill_salad_ms2_freiburg_sthereo_boson vpr_mmdistill_salad_ms2_freiburg_sthereo_boson_vivid  vpr_mmdistill_salad_boson vpr_mmdistill_salad_boson_ms2  \
# 
python3 new_benchmark_vpr.py \
    --model_names salad vpr_mmdistill_salad_frozen_ms2_freiburg_sthereo_boson vpr_mmdistill_salad_ms2_freiburg_sthereo_boson\
    --dataset vivid\
    --top_k_vals 1 5 10 \
    --batch_size 16 \
    --save_qual \
    --qual_k 5 \
    --use_faiss_gpu \
    --use_wandb \
    --wandb_project PlaceRecBench \
    --wandb_entity parv \
    --wandb_group ClassificationModelBenchmark \
    --db_q_mode RGB_THERMAL\
    --dataset_splits val\
    --dist_thresh 15 \
    --exclude_exact_query_in_db \
    --viz_clusters \
    --only_same_backbone