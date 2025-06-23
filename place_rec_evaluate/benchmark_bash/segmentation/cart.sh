python3 benchmark_segmentation.py \
    --model_names  frozen_thr_non_linear_64_head_dice frozen_rgb_non_linear_128_head_dice\
    --dataset_name cart \
    --batch_size 16 \
    --use_wandb \
    --viz_outputs \
    --test_areas socal kentucky northcarolina\
    --splits train test
 
