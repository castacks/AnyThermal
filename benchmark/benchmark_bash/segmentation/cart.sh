    # --model_names  mmdistill_cart_non_linear_64 mmdistill_cart_non_linear_64_248 mmdistill_cart_non_linear_64_250 mmdistill_cart_frozen_non_linear_64\
    # --model_names mmdistill_cart_thermal_dinov2_scaling_boson mmdistill_cart_thermal_dinov2_scaling_boson_vivid mmdistill_cart_thermal_dinov2_scaling_boson_freiburg mmdistill_cart_thermal_dinov2_scaling_boson_freiburg_vivid mmdistill_cart_thermal_dinov2_scaling_boson_vivid_sthereo mmdistill_cart_thermal_dinov2_scaling_boson_freiburg_vivid_sthereo mmdistill_cart_thermal_dinov2_with_tartan_rgbt mmdistill_cart_frozen_rgb_dinov2\

python3 benchmark_segmentation.py \
    --model_names mmdistill_cart_thermal_dinov2_with_tartan_rgbt mmdistill_cart_frozen_rgb_dinov2\
    --dataset_name cart_random \
    --batch_size 128 \
    --use_wandb \
    --viz_outputs \
    --splits test
 
