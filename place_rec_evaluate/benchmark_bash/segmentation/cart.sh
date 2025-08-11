python3 benchmark_segmentation.py \
    --model_names  mmdistill_cart_non_linear_64 mmdistill_cart_frozen_non_linear_64 mmdistill_cart_non_linear_64_salad_init_dice\
    --dataset_name cart \
    --batch_size 16 \
    --use_wandb \
    --viz_outputs \
    --test_areas socal kentucky northcarolina\
    --splits test
 
