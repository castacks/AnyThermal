python3 -m benchmark.benchmark_segmentation \
    --model_names mmdistill_mfnet_anythermal \
    --dataset_name mfnet \
    --batch_size 128 \
    --use_wandb \
    --viz_outputs \
    --splits test
 
