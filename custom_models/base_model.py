from abc import ABC, abstractmethod
import torch
from .utils import *


def default_preprocessing(images: torch.Tensor, keep_ratio,normalise_model,size = 224,normalise = True,resize=True):
    if keep_ratio:
        # Resize while maintaining aspect ratio
        return default_preprocess_tensor_keep_ratio_center_crop(images, normalise_model,size=size,normalise=normalise,resize=resize)
    else:

        return default_preprocess_tensor(images, normalise_model,size=size,normalise=normalise,resize=resize)


class BaseFeatureExtractor(ABC):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.build_model()
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        self.own_recall_method = False
    
    @abstractmethod
    def build_model(self):
        """Return the base CNN model without classifier head."""
        pass

    @abstractmethod
    def preprocess(self, images,keep_ratio=False,resize=True):
        """Return a preprocessed tensor from a tensor of images"""
        pass

    def forward(self, images):
        """Forward pass through the model."""
        return self.model(images)

    def extract_feature(self, images,keep_ratio=False,resize=True):
        tensor = self.preprocess(images,keep_ratio=keep_ratio,resize=resize).to(self.device)
        with torch.no_grad():
            feature = self.forward(tensor)
            assert feature is not None, "Feature extraction failed. Check the model and input."
            assert len(feature.shape) == 2, "Feature extraction should return a 2D tensor."
            assert feature.shape[0] == tensor.shape[0], "Feature shape mismatch with input batch size."
        return feature / feature.norm(p=2, dim=1, keepdim=True)  # Normalize the feature vector

class BaseSegmentationModel(ABC):
    def __init__(self, device, num_classes, **kwargs):
        self.device = device
        self.num_classes = num_classes
        self.model = self.build_model()

    @abstractmethod
    def preprocess(self, images,keep_ratio=False,resize=True):
        """Return a preprocessed tensor from a tensor of images"""
        pass

    @abstractmethod
    def build_model(self):
        """Return the segmentation model."""
        pass

    @abstractmethod
    def forward(self, images):
        """Forward pass through the model."""
        pass