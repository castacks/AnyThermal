from .base_model import *
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from .dinov2_model import DINOv2FeatureExtractor
from torch.nn import functional as F
from torchvision import transforms as T
from tqdm import tqdm
from .dinov2_segmentation_model import SegmentationHead, BaseDinov2SegmentationModel,BaseDinov2SegmentationModelPreUpscaled
from .dinov2_vpr_model import NetVLADHead, BaseDinov2VPRModel
import contextlib
from abc import ABC, abstractmethod
import inspect

class MMDistillDinov2():
    def __init__(self, model_type, modality, un_frozen_layer_index):
        self.model_type = model_type
        self.modality = modality
        self.un_frozen_layer_index = un_frozen_layer_index
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_type).cuda()
        # freeze the layer by setting requires_grad to False
        for name, param in self.model.named_parameters():
            if "blocks" in name and int(name.split('.')[1]) not in self.un_frozen_layer_index:
                param.requires_grad = False
            else:
                param.requires_grad = True
            # print(f"Setting requires_grad for {name} to {param.requires_grad}")

        print(f"Using model {self.model_type} with un_frozen_layer_index {self.un_frozen_layer_index} for modality {self.modality}")
    def forward(self,x,preprocess=True,debug=False):
        #implement a forward method that uses un_frozen_layer_index and self.model
        # if self.modality =='thr':
        #     debug =True
        if debug:
            print("x.requires_grad:", x.requires_grad)

        if preprocess:
            x = self.preprocess(x)
            if debug:
                print("After preprocess, x.requires_grad:", x.requires_grad)

        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)
        if debug:
            print("After prepare_tokens_with_masks, x.requires_grad:", x.requires_grad)
        for idx,blk in enumerate(self.model.blocks):
            # with torch.no_grad() if idx not in self.un_frozen_layer_index else contextlib.nullcontext():
            if debug:
                if idx in self.un_frozen_layer_index:
                    print(f"UnFreezing layer {idx} for modality {self.modality}")
                print("x.requires_grad:", x.requires_grad)
            x = blk(x)
        
        x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.model.embed_dim)).permute(0, 3, 1, 2)
        return f, t

    
    def forward_train(self, x, preprocess=False,return_local_features=False,end_to_end=False):
        batch = x.shape[0]
        # import pdb; pdb.set_trace()
        if end_to_end:
            return self.model(x)
        f,t = self.forward(x, preprocess=preprocess)
        # temp = self.model(x.clone())
        # import pdb; pdb.set_trace()
        if return_local_features:

            return torch.cat([f.reshape((batch,-1)), t.reshape((batch,-1))],dim=-1)
        else:
            return t
        
    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    
    def parameters(self):
        return self.model.parameters()
            

    def preprocess(self, images):
        if self.modality == 'rgb':
            return preprocess_dinov2(images, normalise_model = 'imagenet')
        elif self.modality == 'thr':
            return preprocess_dinov2(images, normalise_model = 'imagenet',normalise=False)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}. Supported modalities are 'rgb' and 'thr'.")
class FixedThermalDistillDINOv2FeatureExtractor(DINOv2FeatureExtractor):
    """
        The difference between this and the VariableThermalDistillDINOv2FeatureExtractor is that this uses a fixed size input - 518*518a as compared to a variable one - new_H = (H // patch_size) * patch_size , new_W = (W // patch_size) * patch_size
    """
    def build_model(self):
        model = torch.hub.load("facebookresearch/dinov2", self.model_type)
        state_dict = torch.load(self.backbone_path, map_location=self.device)["student_model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        #PARV_TODO this shoudl ideally be ViT but the training was done with imagenet mean
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio,size=518,resize=resize,normalise=False)

class VariableThermalDistillDINOv2FeatureExtractor(FixedThermalDistillDINOv2FeatureExtractor):
    def preprocess(self, images,keep_ratio=False,resize=True):
        #PARV_TODO this shoudl ideally be ViT but the training was done with imagenet mean
        return preprocess_dinov2(images, normalise_model = 'imagenet', normalise=False) 


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


class DistillDINOv2FeatureExtractor(DINOv2FeatureExtractor):
    """
        The difference between this and the VariableThermalDistillDINOv2FeatureExtractor is that this uses a fixed size input - 518*518a as compared to a variable one - new_H = (H // patch_size) * patch_size , new_W = (W // patch_size) * patch_size
    """
    def build_model(self):
        model = torch.hub.load("facebookresearch/dinov2", self.model_type)
        state_dict = torch.load(self.backbone_path, map_location=self.device)["student_model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        #PARV_TODO this shoudl ideally be ViT but the training was done with imagenet mean
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio,size=518,resize=resize)



class MMDistillSegmentationModel(BaseSegmentationModel):
    def __init__(self, pre_upscale,model_type, frozen_backbone,frozen_head,device,num_classes,backbone_path="",model_path="",   **kwargs):
        self.model_type = model_type
        self.frozen_backbone = frozen_backbone
        self.backbone_path = backbone_path
        self.frozen_head = frozen_head
        self.model_path = model_path
        self.pre_upscale = pre_upscale
        super().__init__(device, num_classes)

    def build_model(self):
        backbone = torch.hub.load("facebookresearch/dinov2", self.model_type).to(self.device)
        head = SegmentationHead(in_channels=768, num_classes=self.num_classes).to(self.device)
        if self.backbone_path!= "" and self.model_path != "":
            raise ValueError("Both backbone_path and model_path cannot be set at the same time. Please set only one of them.")
        if self.backbone_path!= "":
            print(f"Loading backbone from {self.backbone_path}")
            state_dict = torch.load(self.backbone_path, map_location=self.device)["student_model_state_dict"]
            backbone.load_state_dict(state_dict)
        elif self.model_path != "":
            print(f"Loading backbone and head model from {self.model_path}")
            head_state_dict = torch.load(self.model_path, map_location=self.device)["seg_head"]
            backbone_path = torch.load(self.model_path, map_location=self.device)["backbone_path"]
            backbone_state_dict = torch.load(backbone_path, map_location=self.device)["student_model_state_dict"]
            backbone.load_state_dict(backbone_state_dict)
            head.load_state_dict(head_state_dict)
        if self.pre_upscale:
            model = BaseDinov2SegmentationModelPreUpscaled(backbone, head).to(self.device)
        else:
            model = BaseDinov2SegmentationModel(backbone, head).to(self.device)

        if self.frozen_backbone:
            model.backbone.eval()
        if self.frozen_head:
            model.head.eval()
        return model
    
    def preprocess(self, images, keep_ratio=False, resize=True):
        return images
    
    def forward(self, x):
        return self.model(x, frozen_backbone=self.frozen_backbone)

class MMDistillVPRModel(BaseFeatureExtractor):
    def __init__(self, frozen_backbone,frozen_head,modality,backbone_path="",model_path="", model=None,  **kwargs):
        self.frozen_backbone = frozen_backbone
        self.backbone_path = backbone_path
        self.frozen_head = frozen_head
        self.model_path = model_path
        self.modality = modality
        self.model = model
        super().__init__(**kwargs)


    def build_model(self):
        if self.model is not None:
            print("Using provided model")
            return self.model.to(self.device)

        head = NetVLADHead(mode="local_only").to(self.device) #PARV_TODO make the mode and choosing a head configurable
        if self.backbone_path!= "" and self.model_path != "":
            raise ValueError("Both backbone_path and model_path cannot be set at the same time. Please set only one of them.")
        if self.backbone_path!= "":
            print(f"Loading backbone from {self.backbone_path}")
            self.model_type = torch.load(self.backbone_path, map_location=self.device)["student_model_type"]
            backbone = torch.hub.load("facebookresearch/dinov2", self.model_type).to(self.device)
            state_dict = torch.load(self.backbone_path, map_location=self.device)["student_model_state_dict"]
            backbone.load_state_dict(state_dict)
        elif self.model_path != "":
            print(f"Loading backbone and head model from {self.model_path}")
            head_state_dict = torch.load(self.model_path, map_location=self.device)["vpr_head"]
            backbone_path = torch.load(self.model_path, map_location=self.device)["backbone_path"]
            self.model_type = torch.load(backbone_path, map_location=self.device)["student_model_type"]
            backbone_state_dict = torch.load(backbone_path, map_location=self.device)["student_model_state_dict"]
            backbone = torch.hub.load("facebookresearch/dinov2", self.model_type).to(self.device)

            backbone.load_state_dict(backbone_state_dict)
            head.load_state_dict(head_state_dict)
        
        model = BaseDinov2VPRModel(backbone, head).to(self.device)

        if self.frozen_backbone:
            model.backbone.eval()
        if self.frozen_head:
            model.head.eval()
        return model
    
    def preprocess(self, images, keep_ratio=False, resize=True):
        if self.modality == 'rgb':
            return preprocess_dinov2(images, normalise_model = 'imagenet')
        elif self.modality == 'thermal':
            return preprocess_dinov2(images, normalise_model = 'imagenet',normalise=False)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}. Supported modalities are 'rgb' and 'thermal'.")
    
    def forward(self, x):
        return self.model(x, frozen_backbone=self.frozen_backbone)

    def extract_feature(self, images,keep_ratio=False,resize=True, test=True):
        images = self.preprocess(images,keep_ratio=keep_ratio,resize=resize).to(self.device)
        with torch.no_grad() if test else contextlib.nullcontext():
            feature = self.forward(images)
            assert feature is not None, "Feature extraction failed. Check the model and input."
            assert len(feature.shape) == 2, "Feature extraction should return a 2D tensor."
            assert feature.shape[0] == images.shape[0], "Feature shape mismatch with input batch size."
        return feature / feature.norm(p=2, dim=1, keepdim=True)  # Normalize the feature vector