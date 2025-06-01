import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import wandb
import time
import sys
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc')
from custom_datasets.cart_dataset import *
import yaml
from tqdm import tqdm
import gc
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T

# Define the color map (from screenshot)
ID_TO_RGB = {
    0: (255, 36, 0),        # Unknown
    1: (0, 0, 0),           # Background
    2: (242, 216, 196),     # Bare ground
    3: (89, 70, 54),        # Rocky terrain
    4: (166, 166, 166),     # Developed structures
    5: (82, 89, 90),        # Road
    6: (155, 230, 0),       # Shrubs
    7: (0, 138, 53),        # Trees
    8: (0, 216, 245),       # Sky
    9: (13, 127, 252),      # Water
    10: (255, 249, 0),      # Vehicles
    11: (254, 0, 170),      # Person
}

def label_to_rgb(mask_tensor):
    mask_np = mask_tensor.cpu().numpy()
    h, w = mask_np.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in ID_TO_RGB.items():
        rgb_img[mask_np == k] = color
    return Image.fromarray(rgb_img)

# Simple Segmentation Head
class SegmentationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=12):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)

def transform_images(images,patch_size=14):
    # Resize the images to the required input size
    # import pdb; pdb.set_trace()
    width = images.shape[-2]
    height = images.shape[-1]
    new_width = (width // patch_size) * patch_size
    new_height = (height // patch_size) * patch_size
    images = F.interpolate(images, size=(new_width, new_height), mode='bilinear', align_corners=False)
    return images
# Training pipeline
def train_segmentation_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb_use:
        wandb.init(project="mm_segmentation", name=args.wandb_name, config=vars(args))

    # Load DINOv2 backbone
    backbone = torch.hub.load('facebookresearch/dinov2', args.model_name).to(device)
    
    # Dataset selection
    if args.dataset == "cart":
        print("Using CART dataset")
        train_seq_list = return_cart_split_segmentation("train")
        val_seq_list = return_cart_split_segmentation("val")
        data_root = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files"
        frame_list_root = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/parv/filter/static_segments_output/frames"
        train_dataset = CART(root_frame_dir=frame_list_root, db_modality="thr_seg", q_modality="seg_mask", datasets_folder=data_root, seq=train_seq_list,augment=False) #PARV_TODO enab;e augumenta in segmentation
        val_dataset = CART(root_frame_dir=frame_list_root, db_modality="thr_seg", q_modality="seg_mask", datasets_folder=data_root, seq=val_seq_list, augment=False)
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    val_iter = iter(val_loader)
    # Segmentation Head
    head = SegmentationHead(num_classes=train_dataset.semantic_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    # Resume training
    if args.resume:
        save_path = args.resume_path
        model_resume_path = os.path.join(args.resume_path, f"model{args.resume_epoch_num}.pth")
        yaml_path = yaml.safe_load(os.path.join(args.resume_path, "config.yaml"))
        backbone_path = yaml_path["model_path"]
        if backbone_path == "":
            if args.model_path:
                backbone_path = args.model_path
        if backbone_path == "":
            raise ValueError("Please provide a valid model path to resume from.")
        backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        ckpt = torch.load(model_resume_path, map_location=device)
        head.load_state_dict(ckpt['seg_head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt['epoch']!= args.resume_epoch_num:
            raise ValueError(f"Checkpoint epoch {ckpt['epoch']} does not match resume epoch {args.resume_epoch_num}. Please check the resume path and epoch number.")
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        save_path = os.path.join(args.save_path,args.dataset, args.model_name, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(save_path, exist_ok=True)        
        backbone_path = args.model_path
        if backbone_path == "":
            raise ValueError("Please provide a valid model path to initialize the backbone.")
        if not os.path.exists(backbone_path):
            raise FileNotFoundError(f"Model path {backbone_path} does not exist.")
        backbone.load_state_dict(torch.load(backbone_path, map_location=device)["student_model_state_dict"])
        #dump args in the folder 
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)
    save_vis_dir = os.path.join(save_path, "visualizations")
    os.makedirs(save_vis_dir, exist_ok=True)
    backbone.eval()

    for epoch in range(start_epoch, args.epochs):
        
        total_loss = 0.0
        for batch_number,train_dict in enumerate(tqdm(train_loader)):
            # import pdb; pdb.set_trace()
            head.train()
            imgs_dict = train_dict[0]
            imgs, masks = transform_images(imgs_dict[train_dataset.db_modality]).to(device), transform_images(imgs_dict[train_dataset.q_modality]).to(device).long()
            masks = masks.squeeze(-3)  # Remove channel dimension if present
            with torch.no_grad():
                features = backbone.get_intermediate_layers(imgs, n=1, reshape=True)[0]
            preds = head(features)
            preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_number+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - Batch {batch_number+1}/{len(train_loader)} - Loss: {total_loss/10:.4f}")
                if args.wandb_use:
                    wandb.log({"train_loss": total_loss/10, "epoch": epoch+1, "batch": batch_number+1})
                total_loss = 0.0
                head.eval()
                with torch.no_grad():
                    try:
                        val_imgs_dict = next(val_iter)[0]
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_imgs_dict = next(val_iter)[0]

                    imgs, masks = transform_images(val_imgs_dict[val_dataset.db_modality]).to(device), transform_images(val_imgs_dict[val_dataset.q_modality]).to(device).long()
                    masks = masks.squeeze(-3)  # Remove channel dimension if present
                    features = backbone.get_intermediate_layers(imgs, n=1, reshape=True)[0]
                    preds = head(features)
                    preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    loss = criterion(preds, masks)
                    val_loss = loss.item()
                    print(f"Validation Loss: {val_loss:.4f}")
                    if args.wandb_use:
                        wandb.log({"val_loss": val_loss, "epoch": epoch+1, "batch": batch_number+1})
                    pred_classes = preds.argmax(dim=1)
                    for j in range(imgs.size(0)):
                        pred_img = label_to_rgb(pred_classes[j])
                        gt_img = label_to_rgb(masks[j])
                        img_path = os.path.join(save_vis_dir, f"sample_e{epoch}_{j}_pred.png")
                        gt_path = os.path.join(save_vis_dir, f"sample_e{epoch}_{j}_gt.png")
                        pred_img.save(img_path)
                        gt_img.save(gt_path)
                        #save the original image also 
                        original_img = T.ToPILImage()(imgs[j].cpu())
                        original_img_path = os.path.join(save_vis_dir, f"sample_e{epoch}_{j}_orig.png")
                        original_img.save(original_img_path)
                gc.collect()
                torch.cuda.empty_cache()

        # print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f}")
        # if args.wandb_use:
        #     wandb.log({"train_loss": total_loss, "val_loss": val_loss, "epoch": epoch})

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'seg_head': head.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(save_path, f"model{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train segmentation head on DINOv2 features')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
    parser.add_argument('--dataset', default='cart', type=str, help='Dataset name')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    parser.add_argument('--save_path', default='./checkpoints/segmentation', type=str, help='Path to save the checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--resume_path', default='', type=str, help='Checkpoint file to resume from')
    parser.add_argument('--resume_epoch_num', default=0, type=str, help='Path to the DINOv2 model weights')
    parser.add_argument('--wandb_use', action='store_true', help='Use wandb for logging')
    parser.add_argument('--model_name', default='dinov2_vitb14', type=str, help='Name of the encoder model')
    parser.add_argument('--model_path', default="", type=str, help='Path to the encoder weights')
    parser.add_argument('--wandb_name', default="", type=str, help='Path to the encoder weights')

    args = parser.parse_args()

    train_segmentation_pipeline(args)
