import sys
import torch
from .base_model import *
from .utils import default_preprocess_tensor
import os
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
ImageBind_DIR = os.path.join(PROJECT_ROOT, "baselines", "VPR", "ImageBind")

# Add it to sys.path if not already there
if ImageBind_DIR not in sys.path:
    sys.path.insert(0, ImageBind_DIR)

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class ImageBindRGBFeatureExtractor(BaseFeatureExtractor):
    def build_model(self):
        # Load pretrained weights
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        """
        Preprocess the input images for the ImageBind model.image bind default preprocessing has keep_ratio as true (load_and_transform_vision_data in ImageBind/imagebind/data.py)
        But for consistenecy with other methods we use keep_ratio as false by default
        """
        return default_preprocessing(images, normalise_model = 'vit',keep_ratio=keep_ratio)

    def forward(self, inputs):
        """Forward pass through the model."""
        data = {ModalityType.VISION: inputs}
        return self.model(data)[ModalityType.VISION]

class ImageBindThermalFeatureExtractor(ImageBindRGBFeatureExtractor):
    def preprocess(self, images,keep_ratio=False,resize=True):
        """
        Preprocess the input thermal images for the ImageBind model based on load_and_transform_thermal_data in https://github.com/Kanazawanaoaki/ImageBind/commit/45a3fc55f9ed3f277d12e620ff807a1cafff0f98#diff-f2f5009321a954542fbbfe3f22cab437f6a546b6a89cf71ba8ce9355ad92dfbfR110
        no normalization is done for thermal images
        """

        return default_preprocessing(images, normalise_model = 'vit',keep_ratio=keep_ratio,normalise=False)
    def forward(self, inputs):
        """Forward pass through the model."""
        data = {ModalityType.THERMAL: inputs}
        # this is B,C,H,W. I wan tot keep only the first channel
        """for thermal only a single channel image is passed"""
        data[ModalityType.THERMAL] = data[ModalityType.THERMAL][:,0:1,:,:]
        return self.model(data)[ModalityType.THERMAL]
