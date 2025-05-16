from torchvision import models
import torch.nn as nn
from .base_model import *
from .utils import default_preprocess_tensor
class ResNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, resnet_depth=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.resnet_depth = resnet_depth
        super().__init__(device=device)

    def build_model(self):
        '''
        Select and configure a ResNet backbone based on the specified depth:
        - Supports ResNet18, 34, 50, 101, and 152 — increasing in depth and representational power
        - Loads the corresponding pretrained model from torchvision
        - Removes the final fully connected classification layer (model.children()[:-1])
        - Appends Flatten() to convert the global pooled output into a 1D feature vector:
            - ResNet18 and ResNet34 output: [batch_size, 512, 1, 1] → 512-dim vector
            - ResNet50, 101, and 152 output: [batch_size, 2048, 1, 1] → 2048-dim vector
        This produces a fixed-length embedding suitable for retrieval, while discarding the classification-specific head.

        '''
        # Select the correct ResNet architecture
        if self.resnet_depth == 18:
            model = models.resnet18(pretrained=True)
        elif self.resnet_depth == 34:
            model = models.resnet34(pretrained=True)
        elif self.resnet_depth == 50:
            model = models.resnet50(pretrained=True)
        elif self.resnet_depth == 101:
            model = models.resnet101(pretrained=True)
        elif self.resnet_depth == 152:
            model = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet depth: {self.resnet_depth}")
        
        # Remove the classification head and flatten output
        return nn.Sequential(*(list(model.children())[:-1]), nn.Flatten())

    def preprocess(self, images, keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio)
