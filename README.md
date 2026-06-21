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
pip install -r requirements.txt
```

`requirements.txt` targets Python 3.10 with PyTorch 2.1.2 / CUDA 12.1. The depth baseline needs
one extra package, `pytorch3d`, which has to be built against the installed PyTorch and is therefore
installed in a separate step — see [Monocular Thermal Depth Estimation](#monocular-thermal-depth-estimation).

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

We provide a monocular thermal depth estimator that puts a MiDaS depth head on top of the **frozen
AnyThermal DINOv2 backbone**, trained and evaluated on the
[MS2 dataset](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset). The depth training /
evaluation code lives in the [`BridgeMultiSpectralDepth`](https://github.com/castacks/BridgeMultiSpectralDepth)
submodule.

### 1. Pull the depth submodule

```
cd <PROJECT_ROOT>
git submodule update --init baselines/depth/BridgeMultiSpectralDepth
```

### 2. Install `pytorch3d`

In addition to `requirements.txt`, the depth baseline needs `pytorch3d`. It compiles against the
installed PyTorch, so install it **after** the base dependencies, with build isolation disabled:

```
export CUDA_HOME=/usr/local/cuda-12.2        # a CUDA toolkit matching your torch (cu12.x)
export TORCH_CUDA_ARCH_LIST="8.6"            # your GPU's compute capability (optional; speeds up the build)
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
```

### 3. Checkpoints

Download and extract the pretrained checkpoints (see
[Downloading pretrained checkpoints](#optional-downloading-pretrained-checkpoints)). Depth uses the
frozen backbone and ships trained depth checkpoints:

```
pretrained_checkpoints/
├── backbone/AnyThermal_full/model20.pth                  # frozen backbone (used for training)
└── depth/
    ├── Midas_anythermal/ckpt_epoch=28_step=145000.ckpt   # MiDaS + AnyThermal backbone
    ├── Midas_dinov2/ckpt_epoch=28_step=145000.ckpt       # MiDaS + vanilla DINOv2 backbone
    └── Midas_small/ckpt_epoch=26_step=135000.ckpt        # MiDaS-small baseline
```

The AnyThermal depth config already points at this repo-root `pretrained_checkpoints/` directory
(via a `../../../` relative path), so no copying or symlinking into the submodule is needed. Pass the
same `../../../pretrained_checkpoints/...` prefix for `--ckpt_path` when evaluating (see below).

### 4. MS2 dataset

Request access and download MS2 from the
[official page](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset). For monocular
thermal depth you need the `sync_data` (thermal stereo images + calibration), `proj_depth`
(projected / filtered GT depth), and the official `*_list.txt` split files, arranged as:

```
<MS2_ROOT>/
├── sync_data/<seq>/thr/{img_left,img_right}/*.png   (+ calib.npy per <seq>)
├── proj_depth/<seq>/thr/{depth,depth_filtered,intensity}/*.png
├── train_list.txt   val_list.txt
└── test_day_list.txt   test_night_list.txt   test_rainy_list.txt
```

Point the config at your copy by editing `dataset.MS2.dataset_dir` in
`baselines/depth/BridgeMultiSpectralDepth/configs/Base/Base_Sup_Mono_Depth.yaml`.

### 5. Training

```
cd baselines/depth/BridgeMultiSpectralDepth
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/AnyThermal/Midas_anythermal.yaml \
    --num_gpus 1 --exp_name Midas_anythermal
```

This freezes the AnyThermal backbone (`pretrained_checkpoints/backbone/AnyThermal_full/model20.pth`)
and trains the MiDaS depth head on MS2 thermal images. Checkpoints are written to
`checkpoints/<exp_name>/`.

### 6. Evaluation

Evaluate a checkpoint on each MS2 test condition (`test_day`, `test_night`, `test_rain`). You can
use the released `Midas_anythermal` checkpoint directly, without retraining:

```
cd baselines/depth/BridgeMultiSpectralDepth
for ENV in test_day test_night test_rain; do
  CUDA_VISIBLE_DEVICES=0 python test_monodepth.py \
      --config configs/AnyThermal/Midas_anythermal.yaml \
      --ckpt_path ../../../pretrained_checkpoints/depth/Midas_anythermal/ckpt_epoch=28_step=145000.ckpt \
      --modality thr --test_env ${ENV} \
      --save_dir ./results/Midas_anythermal/${ENV}
done
```

This prints standard depth metrics (Abs Rel, Sq Rel, RMSE, δ accuracies) per condition and saves
depth visualizations under `--save_dir`.

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


