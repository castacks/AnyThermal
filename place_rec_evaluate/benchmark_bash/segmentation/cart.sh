python3 benchmark_segmentation.py \
    --model_names  mmdistill_cart_non_linear_64_dropout_dice mmdistill_cart_non_linear_64_dropout_dice_frozen \
    --dataset_name cart \
    --batch_size 16 \
    --use_wandb \
    --viz_outputs \
    --splits socal_test kentucky_test northcarolina_test \
 
