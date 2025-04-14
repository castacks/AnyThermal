import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
import torch.optim as optim
import time
import os
import argparse
import sys
from datasets.custom_dataset_loader import Custom_MS2Dataset
from utilities import DinoV2ExtractFeatures
import matplotlib.pyplot as plt
import cv2

dataset = Custom_MS2Dataset("/storage2/datasets/ms2_full/")

plot_list = [0, 1, 3, 27, 28, 29, 35, 38, 41, 42, 43, 44, 47, 50, 52, 53, 59, 60, 61, 72, 73, 74, 77, 78, 79, 80, 81, 106, 107, 108, 109, 110, 126]

for i in plot_list:
    print(dataset[i]["rgb1"].shape)
    cv2.imwrite(f"outputs/rgb_{i}.png",dataset[i]["rgb1"])
    cv2.imwrite(f"outputs/thermal_{i}.png",dataset[i]["thermal1"])
    cv2.imwrite(f"outputs/lidar_{i}.png",dataset[i]["lidar1"])

    