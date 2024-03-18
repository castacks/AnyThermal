import timm
import torch.nn.functional as F
from torch import nn
import torch

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name, pretrained, trainable
    ):
        super().__init__()

        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.device=CFG.device
        self.temperature=CFG.temperature
        self.encoder_camera = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=False)
        self.encoder_lidar = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=True)

    def forward(self, batch):
        # Getting camera Image and lidar range image Features
        camera_image_features = self.encoder_camera(batch["camera_image"])
        lidar_image_features = self.encoder_lidar(batch["lidar_image"])

        #Calculating the loss
        loss = torch.nn.functional.mse_loss(camera_image_features, lidar_image_features)

        return loss

    def get_camera_embeddings(self, batch):
        image_features = self.encoder_camera(batch["camera_image"].to(self.device))
        return image_features

    def get_lidar_embeddings(self, batch):
        image_features = self.encoder_lidar(batch["lidar_image"].to(self.device))
        return image_features

def get_topk(query_image_embeddings, lidar_image_embeddings, n=1):
    dot_similarity = query_image_embeddings @ lidar_image_embeddings.T
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    return values, indices