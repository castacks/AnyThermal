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

## (Optional) Downloading pretrianed checkpoints

## Training backbone

```
cd <PROJECT_ROOT>

python3 pretraining.distill --dataset boson sthereo freiburg vivid tartanrgbt --loss_file loss_configs/loss_config_global_contrastive_final.yaml --wandb_name anythermal_backbone_distillaltion
```

This will save the checkpoint in $PROJECT_ROOT/checkpoints/<concatenated_sorted_string_of_dataset_names>


You can also download a pretrained checkpoint - "TODO" 

We will use path to the pretrianed checkpoint as a example for all the training of task-specific heads.

## VPR 

### Training 

```
cd <PROJECT_ROOT>
python3 pretraining.vpr
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

## Segmentation Training and evaluation

## Depth Training and evaluation



