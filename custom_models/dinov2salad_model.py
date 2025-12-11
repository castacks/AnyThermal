import sys
import torch
from .base_model import *
from .utils import default_preprocess_tensor

# Add MixVPR repo to path
import os
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SALAD_DIR = os.path.join(PROJECT_ROOT, "baselines", "VPR", "salad")

if SALAD_DIR not in sys.path:
    sys.path.insert(0, SALAD_DIR)
from baselines.VPR.salad.vpr_model import VPRModel as SaladVPRModel

class DinoV2SALADFeatureExtractor(BaseFeatureExtractor):
    def __init__(self,use_head=True,**kwargs):
        self.use_head = use_head
        super().__init__(**kwargs)


    def build_model(self):
        # Load backbone (ResNet50) and MixVPR head
        model = torch.hub.load("serizba/salad", "dinov2_salad")
        # Load pretrained weights
        model.output_dim = model.agg_config["num_clusters"] * model.agg_config["cluster_dim"] + model.agg_config["token_dim"]
        model = model.eval()
        return model

    def preprocess(self, images, keep_ratio=False,resize=True):
        size = 322
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio, size=size)

    def forward(self, images):
        """Forward pass through the model."""
        return self.model(images)