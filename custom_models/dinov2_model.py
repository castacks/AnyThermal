from .base_model import *
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class DINOv2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model_type="dinov2_vits14", use_cls=False, use_intermediate_layers = True,backbone_path="",**kwargs):
        self.model_type = model_type
        self.use_cls = use_cls  # CLS token or patch pooling
        self.use_intermediate_layers = use_intermediate_layers
        self.backbone_path = backbone_path
        super().__init__(**kwargs)

    def build_model(self):
        model = torch.hub.load("facebookresearch/dinov2", self.model_type)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        """
        Preprocess the input images for the ImageBind model.image bind default preprocessing has keep_ratio as true (load_and_transform_vision_data in ImageBind/imagebind/data.py)
        But for consistenecy with other methods we use keep_ratio as false by default
        """
        return default_preprocessing(images, normalise_model = 'vit',keep_ratio=keep_ratio,size=518)

    def forward(self, inputs):
        if self.use_intermediate_layers:
           
            features = self.model.get_intermediate_layers(inputs, n=1)[0]

            if self.use_cls:
                desc = features[:, 0]  # [CLS] token
            else:
                desc = features[:, 1:]  # all patch tokens
                desc = desc.mean(dim=1)  # mean pooling
        else:
            desc = self.model(inputs)
        return desc

class DINOv2FeatureExtractor_Variable(DINOv2FeatureExtractor):
    """
    This is the variable size input version of the DINOv2 feature extractor. It uses the default ViT preprocessing.
    """
    def preprocess(self, images,keep_ratio=False,resize=True):
        return preprocess_dinov2(images, normalise_model = 'vit')