import sys
import torch
from .base_model import *
from .utils import default_preprocess_tensor

# Add MixVPR repo to path
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition")
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/MixVPR")
from MixVPR.main import VPRModel

class MixVPRFeatureExtractor(BaseFeatureExtractor):
    def build_model(self):
        # Load backbone (ResNet50) and MixVPR head
        model = VPRModel(backbone_arch='resnet50', 
                 layers_to_crop=[4],
                 agg_arch='MixVPR',
                 agg_config={'in_channels' : 1024,
                             'in_h' : 20,
                             'in_w' : 20,
                             'out_channels' : 1024,
                             'mix_depth' : 4,
                             'mlp_ratio' : 1,
                             'out_rows' : 4},
                )
        # Load pretrained weights
        state_dict = torch.load('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/MixVPR/pretrained_models/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess(self, images, keep_ratio=False,resize=True):
        size = 320
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio, size=size)

        
