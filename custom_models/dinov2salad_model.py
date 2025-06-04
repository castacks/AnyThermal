import sys
import torch
from .base_model import *
from .utils import default_preprocess_tensor

# Add MixVPR repo to path
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition")
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/salad")

from salad.vpr_model import VPRModel as SaladVPRModel

class DinoV2SALADFeatureExtractor(BaseFeatureExtractor):
    def build_model(self):
        # Load backbone (ResNet50) and MixVPR head
        model = SaladVPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
        )
        # Load pretrained weights
        model.load_state_dict(torch.load("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/salad/pretrained_models/dino_salad.ckpt"))
        model = model.eval()
        return model

    def preprocess(self, images, keep_ratio=False,resize=True):
        size = 322
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio, size=size)

class ThermalMMDistillDinoV2SALADFeatureExtractor(DinoV2SALADFeatureExtractor):
    def build_model(self):
        # Load backbone (ResNet50) and MixVPR head
        model = SaladVPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
        )
        # Load pretrained weights
        model.load_state_dict(torch.load("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/salad/pretrained_models/dino_salad.ckpt"))
        mmdistill_state_dict = torch.load("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2/rgb_thr/2025-05-17_15-50-49/model9.pth", map_location=self.device)["student_model_state_dict"]
        # import pdb; pdb.set_trace()
        model.backbone.model.load_state_dict(mmdistill_state_dict)
        model = model.eval()
        return model

class RGBMMDistillDinoV2SALADFeatureExtractor(DinoV2SALADFeatureExtractor):
    def build_model(self):
        # Load backbone (ResNet50) and MixVPR head
        model = SaladVPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
        )
        # Load pretrained weights
        model.load_state_dict(torch.load("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/salad/pretrained_models/dino_salad.ckpt"))
        # mmdistill_state_dict = torch.load("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2/rgb_thr/2025-05-17_15-50-49/model9.pth", map_location=self.device)["student_model_state_dict"]
        # import pdb; pdb.set_trace()
        rgb_model = torch.hub.load("facebookresearch/dinov2", 'dinov2_vitb14')

        model.backbone.model.load_state_dict(rgb_model.state_dict())
        model = model.eval()
        return model
