from .base_model import *
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from .dinov2_model import DINOv2FeatureExtractor
from torch.nn import functional as F
from torchvision import transforms as T
from tqdm import tqdm
class FixedThermalDistillDINOv2FeatureExtractor(DINOv2FeatureExtractor):
    """
        The difference between this and the VariableThermalDistillDINOv2FeatureExtractor is that this uses a fixed size input - 518*518a as compared to a variable one - new_H = (H // patch_size) * patch_size , new_W = (W // patch_size) * patch_size
    """
    def build_model(self):
        model = torch.hub.load("facebookresearch/dinov2", self.model_type)
        state_dict = torch.load("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints_ce/dinov2_ms2_checkpoints_thermal_global_bigger_denser_no_night/thermal4.pth", map_location=self.device)["model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        #PARV_TODO this shoudl ideally be ViT but the training was done with imagenet mean
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio,size=518,resize=resize)

class VariableThermalDistillDINOv2FeatureExtractor(FixedThermalDistillDINOv2FeatureExtractor):
    def preprocess(self, images,keep_ratio=False,resize=True):
        #PARV_TODO this shoudl ideally be ViT but the training was done with imagenet mean
        return preprocess_dinov2(images, normalise_model = 'imagenet') 


class FixedRGBDistillDINOv2FeatureExtractor(DINOv2FeatureExtractor):
    def build_model(self):
        model = torch.hub.load("facebookresearch/dinov2", self.model_type)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        #PARV_TODO this shoudl ideally be ViT but the training was done with imagenet mean
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio,size=518,resize=resize)

class VariableRGBDistillDINOv2FeatureExtractor(FixedRGBDistillDINOv2FeatureExtractor):
    def preprocess(self, images,keep_ratio=False,resize=True):
        #PARV_TODO this shoudl ideally be ViT but the training was done with imagenet mean
        return preprocess_dinov2(images, normalise_model = 'imagenet') 
