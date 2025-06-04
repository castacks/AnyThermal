import argparse
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
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
from custom_models.dinov2_segmentation_model import SegmentationHead
from custom_models.mmdistill_dinov2_model import MMDistillSegmentationModel
from utils.segment_utils import label_to_rgb, transform_images
from contextlib import nullcontext

def viz_pred_masks(imgs,preds, masks, epoch, save_vis_dir, semantic_id_to_rgb):
    pred_classes = preds.argmax(dim=1)
    for j in range(imgs.size(0)):
        pred_img = label_to_rgb(pred_classes[j],semantic_id_to_rgb)
        gt_img = label_to_rgb(masks[j],semantic_id_to_rgb)
        img_path = os.path.join(save_vis_dir, f"sample_e{epoch}_{j}_pred.png")
        gt_path = os.path.join(save_vis_dir, f"sample_e{epoch}_{j}_gt.png")
        pred_img.save(img_path)
        gt_img.save(gt_path)
        #save the original image also 
        original_img = T.ToPILImage()(imgs[j].cpu())
        original_img_path = os.path.join(save_vis_dir, f"sample_e{epoch}_{j}_orig.png")
        original_img.save(original_img_path)

def dataloader_loop(args,optimizer,model,dataloader,test, epoch):
    criterion = nn.CrossEntropyLoss()
    with (torch.inference_mode() if test else nullcontext()):
        total_loss = 0.0
        for batch_number,batch in enumerate(tqdm(dataloader)):
            imgs_dict = batch[0]
            imgs, masks = transform_images(imgs_dict["thr_seg"]).to(model.device), transform_images(imgs_dict["seg_mask"],mask=True).to(model.device).long()
            masks = masks.squeeze(-3)  # Remove channel dimension if present
            preds = model.forward(imgs)
            loss = criterion(preds, masks)

            if not test:

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
        total_loss /= len(dataloader)
        loss_str = "train_loss" if not test else "val_loss"

        print(f"Epoch {epoch+1}/{args.epochs}  - {loss_str}: {total_loss:.4f}")
        if args.wandb_use:
            wandb.log({loss_str: total_loss, "epoch": epoch+1})


def train_segmentation_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb_use:
        wandb.init(project="mm_segmentation", name=args.wandb_name, config=vars(args))

    # Load DINOv2 backbone

    
    backbone = torch.hub.load('facebookresearch/dinov2', args.model_name).to(device)
    
    # Dataset selection
    if args.dataset == "cart":
        print("Using CART dataset")
        if args.train_easy:
            train_seq_list = return_cart_split_segmentation("train_easy")
        else:
            train_seq_list = return_cart_split_segmentation("train")
        val_seq_list = return_cart_split_segmentation("val")
        data_root = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files"
        frame_list_root = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/parv/filter/static_segments_output/frames"
        train_dataset = CART(root_frame_dir=frame_list_root, db_modality="thr_seg", q_modality="seg_mask", datasets_folder=data_root, seq=train_seq_list,augment=True) #PARV_TODO enab;e augumenta in segmentation
        val_dataset = CART(root_frame_dir=frame_list_root, db_modality="thr_seg", q_modality="seg_mask", datasets_folder=data_root, seq=val_seq_list, augment=False)
    else:
        raise ValueError("Unsupported dataset")
    semantic_id_to_rgb = val_dataset.semantic_id_to_rgb
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,persistent_workers=True)
    val_iter = iter(val_loader)
    # Segmentation Head
    # head = SegmentationHead(num_classes=train_dataset.semantic_classes).to(device)
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
        # backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        ckpt = torch.load(model_resume_path, map_location=device)
        # head.load_state_dict(ckpt['seg_head'])
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
        # backbone.load_state_dict(torch.load(backbone_path, map_location=device)["student_model_state_dict"])
        #dump args in the folder 
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
    
    # model.model.backbone.eval()  # Freeze the backbone
    optimizer = torch.optim.Adam(model.model.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs):
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        # Training loop
        print("Training...")
        dataloader_loop(args, optimizer, model, train_loader, test=False, epoch=epoch)
        print("Validation ...")
        dataloader_loop(args, optimizer, model, val_loader, test=True, epoch=epoch)
        gc.collect()
        torch.cuda.empty_cache()

        torch.save({
            'epoch': epoch,
            'seg_head': model.model.head.state_dict(),
            'optimizer': optimizer.state_dict(),
            "backbone_path": backbone_path,
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
    parser.add_argument('--train_easy', action='store_true', help='Path to the encoder weights')
    parser.add_argument('--pre_upscale', action='store_true', help='Path to the encoder weights')

    args = parser.parse_args()

    train_segmentation_pipeline(args)
