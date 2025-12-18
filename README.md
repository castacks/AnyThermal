# AnyThermal

## Outline
- Setting up environment 
- Downloading and postprocessing of datasets
- (Optional) Downloading pretrianed checkpoints
- Training backbone (AnyThermal)
- Task training and eval
    - VPR
    - Segmentation 
    - Depth

## Setting up environment 

### Docker 
```
docker pull parvmaheshwari/py310_cu123:latest
```

### Python dependencies 
```
cd <PROJECT_ROOT>
pip install -r requirements.txt -c constraints.txt .
```

## Downloading and postprocessing of datasets

- Change the dataset paths in the `/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/custom_datasets/dataset_path.yaml` file to your local paths for root folders of each of the datasets.

- Follow the instructions in the respective dataset folders to download and postprocess the datasets.

## (Optional) Downloading pretrianed checkpoints

TODO

## Training backbone

```
cd <PROJECT_ROOT>

python3 -m pretraining.distill --dataset boson sthereo freiburg vivid tartanrgbt --loss_file loss_configs/loss_config_global_contrastive_final.yaml --wandb_name anythermal_backbone_distillaltion
```

This will save the checkpoint in $PROJECT_ROOT/checkpoints/<concatenated_sorted_string_of_dataset_names>

## Cross-Modal Place Recognition 

### Training 

```
cd <PROJECT_ROOT>
python3 -m pretraining.vpr --backbone_path pretrained_checkpoints/backbone/AnyThermal_full/model20.pth
```
### Evaluation

```
cd <PROJECT_ROOT>

# For Zero-shot MS2 (urban) Evaluation
bash benchmark/benchmark_bash/vpr/ms2.sh

# For Zero-shot CART (aerial) Evaluation
bash benchmark/benchmark_bash/vpr/cart.sh

#For Zero-shot OdomBeyondVision (indoor) Evalaution
bash benchmark/benchmark_bash/vpr/obv.sh
```

The outputs are saved in 

## Thermal Segmentation (MF-Net and CART)

### Training
```
cd <PROJECT_ROOT>

# For training segmentation on MF-Net 
python3 -m pretraining.segmentation --dataset mfnet --backbone_ckpt pretrained_checkpoints/backbone/AnyThermal_full/model20.pth --augment

# For training segmentation on CART (random data split as provided by the dataset) 
python3 -m pretraining.segmentation --dataset cart_random --backbone_ckpt pretrained_checkpoints/backbone/AnyThermal_full/model20.pth --augment --thermal_segmentation_augmentation brightness_contrast gamma hflip --epochs 125
```

### Evaluation
```
cd <PROJECT_ROOT>

# For evaluating segmentation on MF-Net
bash benchmark/benchmark_bash/segmentation/mfnet.sh

# For evaluating segmentation on CART
bash benchmark/benchmark_bash/segmentation/cart.sh
```

## Monocular Thermal Depth Estimation

### Training
```
```

### Evaluation
```
```



