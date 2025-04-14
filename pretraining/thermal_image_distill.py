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
import wandb

from datasets.custom_dataset_loader import Custom_MS2Dataset
from datasets.wisard_dataset import Wisard_Dataset
from datasets.cart_dataset import CART_dataset

from utilities import DinoV2ExtractFeatures

parser = argparse.ArgumentParser(description='Fine-tuning DINOv2 on ImageNet')
parser.add_argument('--dataset_path', default='/storage2/datasets/ms2_full/', type=str, help='Path to the ImageNet dataset')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--save_path', default='./checkpoints_ce/cart_train_both', type=str, help='Path to save the checkpoints')
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--resume_epoch_num', default=0, type=int, help='Epoch number to resume training from')
parser.add_argument('--loss_type', default="ce", type=str, help='Loss type: mse or similarity')
parser.add_argument('--wandb_use',default=False, type=bool, help='Use wandb for logging')
parser.add_argument('--dataset', default="ms2", type=str, help='Dataset to use: ms2 or wisard')
parser.add_argument('--train_both', default=True, type=bool, help='Train both RGB and thermal models')
args = parser.parse_args()
print(args)

#Initialize wandb
if args.wandb_use:
    wandb.init(project="multiloc", entity="jkarhade", name="thermal_image_distill")

# Load models
rgb_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
rgb_model.eval()
thermal_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
thermal_model.eval()

temperature = 1
embedding_dim = 384

# Freeze the RGB model, so that only the thermal model is trained
for param in rgb_model.parameters():
    param.requires_grad = False

# Fine-tune the thermal model
if args.train_both:
    print("Training both models")
    optimizer = optim.SGD(list(thermal_model.blocks[:].parameters()) + list(rgb_model.blocks[:].parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    print("Training only thermal model")
    optimizer = optim.SGD(thermal_model.blocks[:].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def loss_fn_mse(outputs, targets):
    return F.mse_loss(outputs, targets)

def contrastive_loss(teacher_embed, student_embed, temperature=0.07):

    # Normalize embeddings
    teacher_embed = F.normalize(teacher_embed, dim=-1)
    student_embed = F.normalize(student_embed, dim=-1)
    
    # Compute cosine similarity between embeddings
    similarity = torch.matmul(teacher_embed, student_embed.T) / temperature
    
    # Generate mask to exclude similarity of same embeddings
    mask = torch.eye(similarity.size(0), dtype=torch.bool).cuda()
    
    # Compute logits for positive and negative pairs
    positive_pairs = torch.diag(similarity)  # similarity of same embeddings
    negative_pairs = similarity[~mask].view(similarity.size(0), -1)  # similarity of different embeddings
    # Compute log probabilities
    logits = torch.cat([positive_pairs.unsqueeze(1), negative_pairs], dim=1)
    # Apply log-softmax to logits
    log_probs = F.log_softmax(logits, dim=1)
    # Negative log probability of the true pairs (positive pairs)
    loss = -log_probs[:, 0].mean()
    
    return loss

def train():

    # Load the dataset
    if args.dataset == "ms2":
        print("Using MS2 dataset")
        dataset = Custom_MS2Dataset(args.dataset_path)
    elif args.dataset == "wisard":
        print("Using Wisard dataset")
        dataset = Wisard_Dataset(args.dataset_path)
    elif args.dataset == "cart":
        print("Using CART dataset")
        dataset = CART_dataset(args.dataset_path)
        # x = dataset[5732]

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

            # Forward pass
            rgb_output1 = rgb_model(rgb_images1)
            thermal_output1 = thermal_model(thermal_images1)

            # Compute loss
            if args.loss_type == "mse":
                loss = loss_fn_mse(thermal_output1, rgb_output1)

            elif args.loss_type == "ce":
                loss = contrastive_loss(rgb_output1, thermal_output1)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb_use:
                wandb.log({"loss": loss.item()})

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

        if args.train_both:
            torch.save({
                'epoch': epoch,
                'model_state_dict': rgb_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, os.path.join(args.save_path, "rgb" +str(epoch) + '.pth'))

        print("Epoch {} of {} took {:.3f}s".format(epoch, args.epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(loss))

if __name__ == "__main__":
    train()