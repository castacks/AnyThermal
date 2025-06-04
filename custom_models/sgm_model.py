from torchvision import models
import torch.nn as nn
from .base_model import *
from .utils import default_preprocess_tensor
import sys
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition')
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/STHN')
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/STHN/global_pipeline')
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/STHN/global_pipeline/model')

from STHN.global_pipeline.model import network
from STHN.global_pipeline.util import resume_model

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Args:
    # Generator
    G_tanh: bool = False
    G_contrast: str = "none"
    force_ce: bool = False

    # GAN
    GAN_epochs_decay: int = 0
    GAN_resize: List[int] = field(default_factory=lambda: [512, 512])
    GAN_mode: str = "lsgan"
    GAN_upsample: str = "bilinear"
    GAN_save_freq: int = 0
    GAN_norm: str = "batch"
    D_net: str = "none"
    G_net: str = "none"
    G_loss_lambda: float = 100.0

    # DANN
    DA_only_positive: bool = False
    lambda_DA: float = 0.1
    DA: bool = False

    # Other params
    use_sparse_database: int = -1
    use_extended_data: bool = False
    exclude_val_region: bool = False
    visual_all: bool = False
    prior_location_threshold: int = -1
    use_best_n: int = 1
    train_batch_size: int = 4
    infer_batch_size: int = 16
    criterion: str = "triplet"
    margin: float = 0.1
    epochs_num: int = 1000
    patience: int = 3
    lr: float = 0.00001
    lr_crn_layer: float = 5e-3
    lr_crn_net: float = 5e-4
    optim: str = "adam"
    cache_refresh_rate: int = 1000
    queries_per_epoch: int = 5000
    negs_num_per_query: int = 10
    neg_samples_num: int = 1000
    mining: str = "partial"

    # Model
    backbone: str = "resnet18conv4"
    l2: str = "before_pool"
    aggregation: str = "netvlad"
    netvlad_clusters: int = 64
    pca_dim: Optional[int] = None
    num_non_local: int = 1
    non_local: bool = False
    channel_bottleneck: int = 128
    fc_output_dim: Optional[int] = None
    unfreeze: bool = False
    pretrain: str = "imagenet"
    off_the_shelf: str = "imagenet"
    trunc_te: Optional[int] = None
    freeze_te: Optional[int] = None

    # Init
    seed: int = 0
    resume: Optional[str] = None

    # System
    device: str = "cuda"
    num_workers: int = 8
    resize: List[int] = field(default_factory=lambda: [512, 512])
    test_method: str = "hard_resize"
    majority_weight: float = 0.01
    efficient_ram_testing: bool = False
    val_positive_dist_threshold: int = 50
    train_positives_dist_threshold: int = 35
    recall_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 30, 50, 100])

    # Augmentation
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    hue: Optional[float] = None
    rand_perspective: Optional[float] = None
    horizontal_flip: bool = False
    random_resized_crop: Optional[float] = None
    random_rotation: Optional[float] = None

    # Paths
    datasets_folder: str = "/path/to/datasets"
    dataset_name: str = "foxtech_satellite"
    pca_dataset_folder: Optional[str] = None
    save_dir: str = "default"
    output_pairs: bool = False


class SGMFeatureExtractor(BaseFeatureExtractor):
    def build_model(self):
        # import pdb; pdb.set_trace()
        args = Args()
        args.resume='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/STHN/logs/global_retrieval/satellite_0_thermalmapping_135_contrast_dense_exclusion-2024-02-14_23-02-31-91400d55-5881-48e5-b6cb-cecff4f47a3f/best_model.pth'
        args.aggregation ='gem'
        args.backbone = 'resnet50conv4'
        args.fc_output_dim =4096
        self.args = args
        print("Loading SGM model from args = {}".format(self.args))
        model = network.GeoLocalizationNet(self.args)
        model = resume_model(self.args,model)

        model.eval()
        
        return model

    def preprocess(self, images, keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio,size=self.args.resize,resize=resize)
