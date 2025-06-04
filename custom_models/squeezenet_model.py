from torchvision import models
import torch.nn as nn
from .base_model import *
from .utils import default_preprocess_tensor

class SqueezeNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, version='1_1', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.version = version
        super().__init__(device=device)

    def build_model(self):
        if self.version == '1_0':
            model = models.squeezenet1_0(pretrained=True)
        elif self.version == '1_1':
            model = models.squeezenet1_1(pretrained=True)
        else:
            raise ValueError(f"Unsupported SqueezeNet version: {self.version}")
        
        # Feature extractor for SqueezeNet:
            # - model.features: convolutional backbone for spatial feature extraction
            # - AdaptiveAvgPool2d((1, 1)): global average pooling to produce fixed-size (512, 1, 1) tensor
            # - Flatten(): converts pooled feature map into a 512-dim feature vector
            # This produces a compact, fixed-length embedding suitable for image retrieval.

        return nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def preprocess(self, images, keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio)
