import sys
import os
MCNET_DIR = os.path.join(os.environ["ANYTHERMAL_PROJECT_ROOT"], "baselines", "segmentation", "MCNet")

sys.path.append(os.path.join(MCNET_DIR,"TorchSeg-MCNet/model/MCNet/scut.mcnet.R101"))
sys.path.append(os.path.join(MCNET_DIR,"TorchSeg-MCNet/furnace"))
sys.path.append(os.path.join(MCNET_DIR,"TorchSeg-MCNet"))
sys.path.append(os.path.join(MCNET_DIR,"TorchSeg-MCNet/furnace/utils"))
from network import MCNet
from .base_model import *


class MCNetModel(BaseSegmentationModel):
    def build_model(self):
        model = MCNet(self.num_classes, criterion=None, edge_criterion=None)
        model = model.to(self.device)
        return model
    
    def forward(self, images):
        return self.model(images)
    def preprocess(self, images,keep_ratio=False,resize=True):
        """Return a preprocessed tensor from a tensor of images"""
        return images
    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
    def parameters(self):
        """Return the model parameters."""
        return self.model.parameters()