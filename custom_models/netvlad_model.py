import torch.nn as nn
from .base_model import *
from .utils import default_preprocess_tensor
from torchvision import models
import sys
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/pytorch-NetVlad")
from netvlad import NetVLAD  # from Nanne's repo or custom implementation

class NetVLADFeatureExtractor(BaseFeatureExtractor):
    def build_model(self):
        # Base CNN (e.g., VGG16 or ResNet18 without classifier)
        base_cnn = models.vgg16(pretrained=True).features[:-1]
        dim = 512  # VGG16 conv5 output channels
        vlad_layer = NetVLAD(num_clusters=64, dim=dim)

        model = nn.Sequential(
            base_cnn,
            vlad_layer  # outputs [batch_size, 64 * 512]
        )
        return model

    def preprocess(self, images, keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio)
