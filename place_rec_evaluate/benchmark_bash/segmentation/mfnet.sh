python3 benchmark_segmentation.py \
    --model_names  mmdistill_mfnet_non_linear_64 mmdistill_mfnet_frozen_non_linear_64\
    --dataset_name mfnet \
    --batch_size 128 \
    --use_wandb \
    --viz_outputs \
    --splits train val test
 
