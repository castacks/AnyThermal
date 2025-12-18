python3 -m benchmark.benchmark_segmentation \
    --model_names mmdistill_cart_anythermal\
    --dataset_name cart_random \
    --batch_size 128 \
    --use_wandb \
    --viz_outputs \
    --splits test
 
