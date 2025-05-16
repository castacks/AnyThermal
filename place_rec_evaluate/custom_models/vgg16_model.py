from torchvision import models
import torch.nn as nn
from .base_model import *
from .utils import default_preprocess_tensor

class VGG16FeatureExtractor(BaseFeatureExtractor):
    def build_model(self):
        model = models.vgg16(pretrained=True)
        
        # Retain only feature extractor, exclude classifier
        # Feature extractor for VGG16:
            # - model.features: stack of convolutional + ReLU + MaxPool layers; reduces spatial resolution by a factor of 32
            #   Output shape for input of size (H, W) will be [batch_size, 512, H/32, W/32]
            #   For example, input 224×224 → output [batch_size, 512, 7, 7]
            # - AdaptiveAvgPool2d((7, 7)): forces output to have fixed spatial dimensions (7×7) regardless of input size
            #   Useful when input resolution may vary, or to match VGG16’s original classifier input shape
            # - Flatten(): converts the pooled 512×7×7 feature map into a 25088-dim vector
            # This setup strips off the classifier and yields a consistent global descriptor for image retrieval or similarity tasks.

        return nn.Sequential(*list(model.features.children()), nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten())

    def preprocess(self, images, keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio)
