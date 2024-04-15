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

from datasets.custom_dataset_loader import Custom_MS2Dataset
from utilities import DinoV2ExtractFeatures

parser = argparse.ArgumentParser(description='Fine-tuning DINOv2 on ImageNet')
parser.add_argument('--dataset_path', default='/ocean/projects/cis220039p/shared/datasets/MS2_full/', type=str, help='Path to the ImageNet dataset')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--save_path', default='./checkpoints_thermal_global', type=str, help='Path to save the checkpoints')
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--resume_epoch_num', default=0, type=int, help='Epoch number to resume training from')
parser.add_argument('--loss_type', default="similarity", type=str, help='Loss type: mse or similarity')
args = parser.parse_args()
print(args)

# Load models
rgb_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
rgb_model.eval()
thermal_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
thermal_model.eval()

# Load Dino feature extractor model
# rgb_model = DinoV2ExtractFeatures("dinov2_vits14", 11,"token", device="cuda")
# rgb_model.dino_model.eval()
# thermal_model = DinoV2ExtractFeatures("dinov2_vits14", 11,"token", device="cuda")
# thermal_model.dino_model.train()

# Freeze the RGB model, so that only the lidar model is trained
# for param in rgb_model.dino_model.parameters():
#     param.requires_grad = False

# Freeze the RGB model, so that only the thermal model is trained
for param in rgb_model.parameters():
    param.requires_grad = False

# Fine-tune the thermal model
# optimizer = optim.SGD(thermal_model.dino_model.blocks[:].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer = optim.SGD(thermal_model.blocks[:].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def loss_fn(outputs, targets):
    return F.mse_loss(outputs, targets)

def train():

    # Load the dataset
    dataset = Custom_MS2Dataset(args.dataset_path)

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Create the save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Load the checkpoint if resuming training
    start_epoch = args.resume_epoch_num
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save_path, "thermal" +str(start_epoch) + '.pth'))
        thermal_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming training from epoch {}".format(start_epoch))
    
    # Train the model
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, images in enumerate(train_dataloader):
            rgb_images1 = images['rgb1'].cuda()
            thermal_images1 = images['thermal1'].cuda()

            # print(rgb_images1.shape, thermal_images1.shape)

            # Forward pass
            rgb_output1 = rgb_model(rgb_images1)
            thermal_output1 = thermal_model(thermal_images1)

            # Compute loss
            # print(thermal_output1.shape, rgb_output1.shape)
            loss = loss_fn(thermal_output1, rgb_output1)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch: %d, Batch: %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        scheduler.step()

        # Save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': thermal_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, os.path.join(args.save_path, "thermal" +str(epoch) + '.pth'))

        print("Epoch {} of {} took {:.3f}s".format(epoch, args.epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(loss))

if __name__ == "__main__":
    train()