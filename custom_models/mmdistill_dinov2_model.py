from .base_model import *
import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
from torchvision import transforms as T
from tqdm import tqdm
# from .dinov2_vpr_model import NetVLAD,L2Norm

import contextlib
from abc import ABC, abstractmethod
import inspect
import sys
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/salad")
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition")
from salad.models.helper import get_aggregator
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import logging

global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MMDistillDinov2():
    def __init__(self, model_type, modality, un_frozen_layer_index,layers_to_hook,backbone_path="",proj_head=""):
        self.model_type = model_type
        self.modality = modality
        self.un_frozen_layer_index = un_frozen_layer_index
        self.proj_head = proj_head

        self.model = torch.hub.load("facebookresearch/dinov2", self.model_type).to(global_device)
        self.num_register_tokens = getattr(self.model, "num_register_tokens", 0)
        self.patch_size = self.model.patch_size
        assert self.patch_size == 14, f"Expected patch size to be 14, but got {self.patch_size}. Please check the model architecture."

        if self.proj_head =="linear":
            self.modality_specific_proj_head = nn.Linear(self.model.embed_dim, self.model.embed_dim//2).to(global_device)
            self.modality_shared_proj_head = nn.Linear(self.model.embed_dim, self.model.embed_dim//2).to(global_device)
        elif self.proj_head == "":
            self.modality_specific_proj_head = None
            self.modality_shared_proj_head = None
        else:
            raise ValueError(f"Unsupported proj_head: {self.proj_head}. Supported proj_heads are 'linear' and ''.")

        if backbone_path != "":
            print(f"Loading backbone from {backbone_path}")
            state_dict = torch.load(backbone_path, map_location=global_device,weights_only=True)["student_model_state_dict"]
            self.load_model(state_dict)
        if -1 in self.un_frozen_layer_index:
            temp_unfrozen_layer_index = [ i for i in self.un_frozen_layer_index if i != -1]
            self.un_frozen_layer_index = temp_unfrozen_layer_index
            self.un_frozen_layer_index += list(range(0, len(self.model.blocks)))  # Unfreeze all layers if -1 is present
        self.layers_to_hook = layers_to_hook
        print(f"Using model {self.model_type} with un_frozen_layer_index {self.un_frozen_layer_index} for modality {self.modality}")
    def forward_train(self, x, preprocess=False):
        if preprocess:
            x = self.preprocess(x)
        return self.extract_vit_tokens(x)
        
    def load_model(self, model_dict):
        """
        Loads the model state dict from a dictionary.
        """
        self.model.load_state_dict(model_dict["backbone_model_state_dict"])
        if self.proj_head != "":
            if "modality_specific_proj_head_state_dict" not in model_dict or "modality_shared_proj_head_state_dict" not in model_dict:
                raise ValueError("modality_specific_proj_head_state_dict or modality_shared_proj_head_state_dict not found in model_dict. Please check the model_dict.")
            self.modality_specific_proj_head.load_state_dict(model_dict["modality_specific_proj_head_state_dict"])
            self.modality_shared_proj_head.load_state_dict(model_dict["modality_shared_proj_head_state_dict"])
        print("Model loaded successfully")

    
    def return_model_dict_for_saving(self):
        output = {}
        output["backbone_model_state_dict"] = self.model.state_dict()
        if self.modality_specific_proj_head is not None:
            output["modality_specific_proj_head_state_dict"] = self.modality_specific_proj_head.state_dict()
            output["modality_shared_proj_head_state_dict"] = self.modality_shared_proj_head.state_dict()
        return output

    def eval(self, set_grad=False):
        self.model.eval()
        if self.modality_specific_proj_head is not None:
            self.modality_specific_proj_head.eval()
        if self.modality_shared_proj_head is not None:
            self.modality_shared_proj_head.eval()

        if set_grad:
            for param in self.model.parameters():
                param.requires_grad = False

            if self.proj_head != "":
                for param in self.modality_specific_proj_head.parameters():
                    param.requires_grad = True
                for param in self.modality_shared_proj_head.parameters():
                    param.requires_grad = True

        
    def train(self):
        self.model.train()
        if self.modality_specific_proj_head is not None:
            self.modality_specific_proj_head.train()
        if self.modality_shared_proj_head is not None:
            self.modality_shared_proj_head.train()
    
    def shared(self,x):
        """
        Returns the shared projection head output
        """
        if self.modality_shared_proj_head is None:
            return None
        if len(x.shape) ==2:
            assert x.shape[1] == self.model.embed_dim, f"Expected input shape to be (B, {self.model.embed_dim}), but got {x.shape}"
            return self.modality_shared_proj_head(x)
        elif len(x.shape) == 4:
            assert x.shape[1] == self.model.embed_dim, f"Expected input shape to be (B, {self.model.embed_dim}, H, W), but got {x.shape}"
            return self.modality_shared_proj_head(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Permute to (B, C', H, W)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected shape to be (B, C) or (B, C, H, W).")
    
    def specific(self,x):
        """
        Returns the modality specific projection head output
        """
        if self.modality_specific_proj_head is None:
            return None
        if len(x.shape) ==2:
            assert x.shape[1] == self.model.embed_dim, f"Expected input shape to be (B, {self.model.embed_dim}), but got {x.shape}"
            return self.modality_specific_proj_head(x)
        elif len(x.shape) == 4:
            assert x.shape[1] == self.model.embed_dim, f"Expected input shape to be (B, {self.model.embed_dim}, H, W), but got {x.shape}"
            return self.modality_specific_proj_head(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected shape to be (B, C) or (B, C, H, W).")
            
    
    def unfrozen_parameters(self):
        return self.unfrozen_named_parameters(include_name=False)
    
    def unfrozen_named_parameters(self,include_name=True):
        output = []
        for name, param in self.model.named_parameters():
            include_param = False
            if "blocks" in name:
                if int(name.split('.')[1]) in self.un_frozen_layer_index:
                    include_param = True
            elif "patch_embed" in name:
                if "patch_embed" in self.un_frozen_layer_index:
                    include_param = True
            elif "norm" in name:
                if "norm" in self.un_frozen_layer_index:
                    include_param = True
            elif "cls_token" in name:
                if "cls_token" in self.un_frozen_layer_index:
                    include_param = True
            elif "pos_embed" in name:
                if "pos_embed" in self.un_frozen_layer_index:
                    include_param = True
            elif "mask_token" in name:
                include_param = False
            elif "register_tokens" in name:
                if "register_tokens" in self.un_frozen_layer_index:
                    include_param = True
            else:
                print(f"Unsupported layer name: {name}. Please check the model architecture.")
                import pdb; pdb.set_trace()
            if include_param:
                if include_name:
                    output.append((name, param))
                else:
                    output.append(param)
        if self.proj_head != "":
            if "proj_head" in self.un_frozen_layer_index: 
                for name, param in self.modality_specific_proj_head.named_parameters():
                    if include_name:
                        output.append((f"specific_proj_head.{param.name}", param))
                    else:
                        output.append(param)
                for name, param in self.modality_shared_proj_head.named_parameters():
                    if include_name:
                        output.append((f"shared_proj_head.{param.name}", param))
                    else:
                        output.append(param)
            elif self.un_frozen_layer_index !=[]:
                raise ValueError(f"Unsupported un_frozen_layer_index: {self.un_frozen_layer_index} since linear proj head is present , it shoudl be unfrozen. Please check the model architecture.")
            else:
                raise ValueError(f"Unsupported un_frozen_layer_index: {self.un_frozen_layer_index}. If you want to freeze the proj head, please set proj_head to '' instead of 'linear'.")

        return output
            

    def preprocess(self, images):
        if self.modality == 'rgb':
            return preprocess_dinov2(images, normalise_model = 'vit')
        elif self.modality == 'thr':
            return preprocess_dinov2(images, normalise_model = 'vit',normalise=True) #normalising thermal also
        else:
            raise ValueError(f"Unsupported modality: {self.modality}. Supported modalities are 'rgb' and 'thr'.")

    def convert_output_to_dict(self,output,layer,B,H,W):
        # output: (B, N+1, C)
        output_dict = {}
        output_len = output.shape[1]
        # import pdb; pdb.set_trace()
        output_dict[f"block_{layer}_output"] = [output[:, 1+self.num_register_tokens:].reshape((B, H // self.patch_size, W // self.patch_size, self.model.embed_dim)).permute(0, 3, 1, 2), output[:, 0]]  # Patch tokens
        if self.num_register_tokens > 0:
            output_dict[f"block_{layer}_output"].append(output[:, 1:self.num_register_tokens+1])
        output_dict[f"block_{layer}_shared_output"] = self.shared(output_dict[f"block_{layer}_output"][0]),self.shared(output_dict[f"block_{layer}_output"][1])  # Shared projection head output
        output_dict[f"block_{layer}_specific_output"] = self.specific(output_dict[f"block_{layer}_output"][0]),self.specific(output_dict[f"block_{layer}_output"][1])  # Modality specific projection head output
        return output_dict

    def extract_vit_tokens(self,x):
        """
        Extracts CLS and patch tokens at specified intermediate transformer layers
        and final output after the norm — all from a single forward pass.

        Returns:
            tokens_dict with keys:
                block_{i}_cls, block_{i}_patch,
                final_cls, final_patch
        """
        B,H,W = x.shape[0],x.shape[-2],x.shape[-1]
        tokens_dict = {}

        def make_hook(i):
            if i == 'final':
                def hook_fn_final(module, input, output):
                    # output: (B, N+1, C)
                    tokens_dict["final_input"] = output
                return hook_fn_final
            else:
                def hook_fn(module, input, output):
                    # output: (B, N+1, C)
                    tokens_dict.update(self.convert_output_to_dict(output, i,B,H,W))  # Convert output to dict format
                return hook_fn
        

        # Register hooks
        hook_handles = []
        for i in self.layers_to_hook:
            if i == 'final':
                h = self.model.blocks[-1].register_forward_hook(make_hook(i))
            else:
                h = self.model.blocks[i].register_forward_hook(make_hook(i))
            hook_handles.append(h)

        _ = self.model(x)

        # Apply final norm manually to last hooked output
        if "final_input" in tokens_dict:
            final_output = self.model.norm(tokens_dict["final_input"])  # (B, N+1, C)
            tokens_dict.pop("final_input")  # Remove the raw output of the last layer
            tokens_dict.update(self.convert_output_to_dict(final_output, 'final',B,H,W))  # Convert final output to dict format

        # Cleanup
        for h in hook_handles:
            h.remove()

        return tokens_dict
    
    def to(self,device):
        """
        Move the model to the specified device.
        """
        self.model.to(device)
        if self.modality_specific_proj_head is not None:
            self.modality_specific_proj_head.to(device)
        if self.modality_shared_proj_head is not None:
            self.modality_shared_proj_head.to(device)
        return self