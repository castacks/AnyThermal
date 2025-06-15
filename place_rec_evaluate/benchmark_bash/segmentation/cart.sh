python3 benchmark_segmentation.py \
    --model_names mmdistill_dinov2_cart\
    --dataset_name cart \
    --batch_size 16 \
    --use_wandb \
    --viz_outputs \
    --test_areas socal kentucky northcarolina\
    --splits train val test
 
