import argparse
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from torch.utils.data import DataLoader
import os
import wandb
import time
import sys
import yaml
from tqdm import tqdm
import gc
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T

sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc')
from custom_datasets.cart_dataset import *
from custom_models.dinov2_segmentation_model import SegmentationHead
from custom_models.mmdistill_dinov2_model import MMDistillSegmentationModel
from segment_utils import label_to_rgb
from contextlib import nullcontext
import math
def viz_pred_masks(imgs, preds, masks, epoch, save_vis_dir, semantic_id_to_rgb,test, index):
    pred_classes = preds.argmax(dim=1)
    for j,idx in enumerate(index):
        pred_img = label_to_rgb(pred_classes[j].cpu(), semantic_id_to_rgb)
        gt_img = label_to_rgb(masks[j].cpu(), semantic_id_to_rgb)
        orig_img = T.ToPILImage()(imgs[j].cpu())

        test_folder = "test" if test else "train"
        os.makedirs(os.path.join(save_vis_dir, test_folder), exist_ok=True)
        pred_img.save(os.path.join(save_vis_dir,test_folder, f"sample_e{epoch}_{idx}_pred.png"))
        gt_img.save(os.path.join(save_vis_dir,test_folder, f"sample_e{epoch}_{idx}_gt.png"))
        orig_img.save(os.path.join(save_vis_dir, test_folder,f"sample_e{epoch}_{idx}_orig.png"))

def dataloader_loop(args, optimizer, model, dataloader, test, epoch, semantic_id_to_rgb=None, save_vis_dir=None):
    criterion = nn.CrossEntropyLoss()
    with (torch.inference_mode() if test else nullcontext()):
        total_loss = 0.0
        for batch_number, batch in enumerate(tqdm(dataloader)):
            imgs_dict = batch[0]
            index = batch[1]
            imgs = imgs_dict["thr_seg"].to(model.device)
            masks = imgs_dict["seg_mask"].to(model.device).long()
            masks = masks.squeeze(-3)
            preds = model.forward(imgs)
            num_classes = preds.shape[1]
            valid_mask = masks >=0  # Assuming 255 is the ignore index
            # preds_new = preds.permute(0, 2, 3, 1).reshape(-1, num_classes)  # shape: (32, 3)
            valid_mask = valid_mask.reshape(-1)  # shape: (B*H*W)
            # new_masks = masks.reshape(-1)  # shape: (B*H*W)

            loss = criterion(preds.permute(0, 2, 3, 1).reshape(-1, num_classes)[valid_mask], masks.reshape(-1)[valid_mask])

            if not test:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if test and args.save_visualizations and semantic_id_to_rgb is not None and save_vis_dir is not None:
                viz_pred_masks(imgs, preds, masks, epoch, save_vis_dir, semantic_id_to_rgb,test,index)

        total_loss /= len(dataloader)
        loss_str = "train_loss" if not test else "val_loss"
        print(f"Epoch {epoch+1}/{args.epochs}  - {loss_str}: {total_loss:.4f}")
        if args.wandb_use:
            wandb.log({loss_str: total_loss, "epoch": epoch+1, "lr": optimizer.param_groups[0]['lr']})


def train_segmentation_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb_use:
        wandb.init(project="mm_segmentation", name=args.wandb_name, config=vars(args))

    backbone = torch.hub.load('facebookresearch/dinov2', args.model_name).to(device)

    if args.dataset == "cart":
        print("Using CART dataset")
        train_seq_list = return_cart_split_segmentation_geographic("train", "socal", "thermal")
        val_seq_list = return_cart_split_segmentation_geographic("val", "socal", "thermal")
        train_dataset = CART(root_frame_dir=None, db_modality="thr_seg", q_modality="seg_mask", datasets_folder=None, seq=train_seq_list, augment=True, seq_as_txt="thermal", crop_images=False)
        val_dataset = CART(root_frame_dir=None, db_modality="thr_seg", q_modality="seg_mask", datasets_folder=None, seq=val_seq_list, augment=False, seq_as_txt="thermal", crop_images=False)
    else:
        raise ValueError("Unsupported dataset")

    semantic_id_to_rgb = val_dataset.semantic_id_to_rgb
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)

    start_epoch = 0
    if args.resume:
        save_path = args.resume_path
        model_resume_path = os.path.join(args.resume_path, f"model{args.resume_epoch_num}.pth")
        with open(os.path.join(args.resume_path, "config.yaml"), 'r') as f:
            yaml_dict = yaml.safe_load(f)
        backbone_path = yaml_dict["model_path"]
        if backbone_path == "":
            backbone_path = args.model_path
        if backbone_path == "":
            raise ValueError("Please provide a valid model path to resume from.")
        ckpt = torch.load(model_resume_path, map_location=device)
        if ckpt['epoch'] != args.resume_epoch_num:
            raise ValueError(f"Checkpoint epoch {ckpt['epoch']} does not match resume epoch {args.resume_epoch_num}.")
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        save_path = os.path.join(args.save_path, args.dataset, args.model_name, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(save_path, exist_ok=True)
        backbone_path = args.model_path
        if backbone_path == "" or not os.path.exists(backbone_path):
            raise FileNotFoundError(f"Model path {backbone_path} does not exist.")
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    save_vis_dir = os.path.join(save_path, "visualizations")
    os.makedirs(save_vis_dir, exist_ok=True)

    if args.resume:
        model = MMDistillSegmentationModel(
            model_type=args.model_name,
            frozen_backbone=True,
            frozen_head=False,
            device=device,
            num_classes=train_dataset.semantic_classes,  # This will be updated later based on the dataset
            model_path = model_resume_path,
            pre_upscale=args.pre_upscale
        )
    else:
        model = MMDistillSegmentationModel(
            model_type=args.model_name,
            frozen_backbone=True,
            frozen_head=False,
            device=device,
            num_classes=train_dataset.semantic_classes,  # This will be updated later based on the dataset
            backbone_path=backbone_path,
            pre_upscale=args.pre_upscale
        )

    initial_lr = args.learning_rate

    def warmup_cosine_lr_lambda(epoch):
        if epoch < 10:
            return epoch / 10.  # warmup from 0.1 to 1.0
        else:
            progress = (epoch - 10) / max(1, args.epochs - 10)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return cosine_decay * (1 - 1/1000) + 1/1000  # decays from 1 → 0.001

    
    from torch.optim.lr_scheduler import LambdaLR
    optimizer = torch.optim.Adam(model.model.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Resumed optimizer state from {model_resume_path}")
    # scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr_lambda)

    # scheduler.step()
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("Training...")
        dataloader_loop(args, optimizer, model, train_loader, test=False, epoch=epoch, semantic_id_to_rgb=semantic_id_to_rgb, save_vis_dir=save_vis_dir)
        print("Validation ...")
        dataloader_loop(args, optimizer, model, val_loader, test=True, epoch=epoch, semantic_id_to_rgb=semantic_id_to_rgb, save_vis_dir=save_vis_dir)
        gc.collect()
        torch.cuda.empty_cache()
        # scheduler.step()

        torch.save({
            'epoch': epoch,
            'seg_head': model.model.head.state_dict(),
            'optimizer': optimizer.state_dict(),
            "backbone_path": backbone_path,
            "pre_upscale": args.pre_upscale,
        }, os.path.join(save_path, f"model{epoch}.pth"))

        # scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train segmentation head on DINOv2 features')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--dataset', default='cart', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--save_path', default='./checkpoints/segmentation', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', default='', type=str)
    parser.add_argument('--resume_epoch_num', default=0, type=int)
    parser.add_argument('--wandb_use', action='store_true')
    parser.add_argument('--model_name', default='dinov2_vitb14', type=str)
    parser.add_argument('--model_path', default="", type=str)
    parser.add_argument('--wandb_name', default="", type=str)
    parser.add_argument('--train_easy', action='store_true')
    parser.add_argument('--pre_upscale', action='store_true')
    parser.add_argument('--save_visualizations', action='store_true')
    parser.add_argument('--crop_images', default=False, help='Rescale images during cropping')

    args = parser.parse_args()

    train_segmentation_pipeline(args)
