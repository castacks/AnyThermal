# AnyThermal

This code repository supports the training and evaluation of the AnyThermal model introduced in the paper "AnyThermal: Towards Learning Universal Representations for Thermal Perception," accepted at ICRA 2026.
Please feel free to create issues if you need assistance with using the checkpoints or the codebase. Will try to resolve the issues as soon as possible.  

## Outline
- Setting up the environment 
- Downloading and postprocessing of datasets
- (Optional) Downloading pre-trained checkpoints
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

### Submodules

VPR (`pretraining.vpr`, `benchmark.benchmark_vpr`) imports the SALAD aggregator from a submodule.  After cloning, pull at least the SALAD submodule:

```
git submodule update --init baselines/VPR/salad
```

The other submodules (`STHN`, `ImageBind`, `MCNet`, `BridgeMultiSpectralDepth`, `fieldscale`) are only needed if you run those specific baselines; init them on demand.

## Downloading and postprocessing of datasets

- Change the dataset paths in the `/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/custom_datasets/dataset_path.yaml` file to your local paths for root folders of each of the datasets.

- Follow the instructions in the respective dataset folders to download and postprocess the datasets.

## (Optional) Downloading pretrained checkpoints

Download [`anythermal_checkpoints.zip`](https://huggingface.co/theairlabcmu/AnyThermal/blob/main/anythermal_checkpoints.zip) from HuggingFace. The zip's top-level directory is `pretrained_checkpoints/`, so extracting from the repo root would produce a nested `pretrained_checkpoints/pretrained_checkpoints/` tree that the code paths in [`custom_models/str_to_cls.py`](custom_models/str_to_cls.py) don't expect. Extract to a scratch dir and move the contents in instead:

```bash
cd <PROJECT_ROOT>
unzip /path/to/anythermal_checkpoints.zip -d /tmp/anythermal_ckpt
mv /tmp/anythermal_ckpt/pretrained_checkpoints ./
rm -rf /tmp/anythermal_ckpt
```

The result is a flat `pretrained_checkpoints/{backbone,segmentation,vpr,depth}/...` under the repo root.

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

All the evaluation outputs are saved in benchmark/qualitative_ouptuts

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

All the evaluation outputs are saved in benchmark/qualitative_ouptuts


## Monocular Thermal Depth Estimation

Instructions for this will be added soon.

## Citation 

If you found this repo to be helpful, please give us a star and consider citing our work 

```
@misc{maheshwari2026anythermallearninguniversalrepresentations,
      title={AnyThermal: Towards Learning Universal Representations for Thermal Perception}, 
      author={Parv Maheshwari and Jay Karhade and Yogesh Chawla and Isaiah Adu and Florian Heisen and Andrew Porco and Andrew Jong and Yifei Liu and Santosh Pitla and Sebastian Scherer and Wenshan Wang},
      year={2026},
      eprint={2602.06203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.06203}, 
}
```


