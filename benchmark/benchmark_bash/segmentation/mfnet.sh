    # --model_names  mmdistill_mfnet_non_linear_64 mmdistill_mfnet_non_linear_64_248 mmdistill_mfnet_non_linear_64_250 mmdistill_mfnet_frozen_non_linear_64\
    # --model_names mmdistill_mfnet_thermal_dinov2_scaling_boson mmdistill_mfnet_thermal_dinov2_scaling_boson_vivid mmdistill_mfnet_thermal_dinov2_scaling_boson_freiburg mmdistill_mfnet_thermal_dinov2_scaling_boson_freiburg_vivid mmdistill_mfnet_thermal_dinov2_scaling_boson_vivid_sthereo mmdistill_mfnet_thermal_dinov2_scaling_boson_freiburg_vivid_sthereo mmdistill_mfnet_thermal_dinov2_with_tartan_rgbt\
python3 -m benchmark.benchmark_segmentation \
    --model_names mmdistill_mfnet_anythermal \
    --dataset_name mfnet \
    --batch_size 128 \
    --use_wandb \
    --viz_outputs \
    --splits test
 
