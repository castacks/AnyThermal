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
from itertools import chain

sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc')
from custom_datasets.cart_dataset import *
from custom_datasets.freiburg_dataset import *
from custom_models.dinov2_segmentation_model import seg_head_str_to_dict
from custom_models.mmdistill_dinov2_model import MMDistillSegmentationModel
from segment_utils import label_to_rgb
from contextlib import nullcontext
import math
from utilities import seed_everything
seed_everything()
db_modality = None
def viz_pred_masks(imgs, preds, masks, epoch, save_vis_dir, semantic_id_to_rgb,test, index):
    pred_classes = preds.argmax(dim=1)
    for j,idx in enumerate(index):
        if j%5 !=0:
            continue
        pred_img = label_to_rgb(pred_classes[j].cpu(), semantic_id_to_rgb)
        gt_img = label_to_rgb(masks[j].cpu(), semantic_id_to_rgb)
        orig_img = T.ToPILImage()(imgs[j].cpu())

        test_folder = "test" if test else "train"
        os.makedirs(os.path.join(save_vis_dir, test_folder), exist_ok=True)
        pred_img.save(os.path.join(save_vis_dir,test_folder, f"sample_e{epoch}_{idx}_pred.png"))
        gt_img.save(os.path.join(save_vis_dir,test_folder, f"sample_e{epoch}_{idx}_gt.png"))
        orig_img.save(os.path.join(save_vis_dir, test_folder,f"sample_e{epoch}_{idx}_orig.png"))

def compute_iou_metrics(preds, masks, num_classes):
    """
    Computes class IoU, mean IoU, and frequency-weighted IoU.
    """
    preds = preds.numpy()   # shape (B, H, W)
    masks = masks.numpy()                 # shape (B, H, W)
    # import pdb; pdb.set_trace()

    iou_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)
    union_per_class = np.zeros(num_classes)

    for cls in range(num_classes):
        cls_pred = (preds == cls)
        cls_gt = (masks == cls)
        valid_mask = masks >= 0
        # import pdb; pdb.set_trace() 
        cls_pred = cls_pred & valid_mask
        cls_gt = cls_gt & valid_mask

        intersection = np.logical_and(cls_pred, cls_gt).sum()
        union = np.logical_or(cls_pred, cls_gt).sum()
        total = cls_gt.sum()

        iou_per_class[cls] = intersection / union if union > 0 else np.nan
        union_per_class[cls] = union
        total_per_class[cls] = total

    miou = np.nanmean(iou_per_class)
    fwiou = np.nansum((total_per_class / np.sum(total_per_class)) * iou_per_class)

    return iou_per_class, miou, fwiou

def dice_loss(pred, target, eps=1e-6):
    """
    Args:
        pred: (N, C) - raw logits
        target: (N,) - integer class indices
    Returns:
        scalar Dice loss
    """
    N, C = pred.shape
    pred_soft = torch_F.softmax(pred, dim=1)          # (N, C)
    target_one_hot = torch_F.one_hot(target, num_classes=C).float()  # (N, C)

    intersection = (pred_soft * target_one_hot).sum(dim=0)     # (C,)
    union = pred_soft.sum(dim=0) + target_one_hot.sum(dim=0)   # (C,)

    dice = (2 * intersection + eps) / (union + eps)            # (C,)
    return 1 - dice.mean()


def dataloader_loop(args, optimizer, model, dataloader, test, epoch, semantic_id_to_rgb=None, save_vis_dir=None,class_weights=None):
    # import pdb; pdb.set_trace()
    global db_modality
    losses = []
    for loss_type in args.loss_type:
        if loss_type not in ['ce', 'weighted_ce', 'dice']:
            raise ValueError(f"Unsupported loss_type type: {loss_type}")
        if loss_type == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'weighted_ce':
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == 'dice':
            loss_fn = dice_loss
        losses.append((loss_type,loss_fn))
    with (torch.inference_mode() if test else nullcontext()):
        total_loss = 0.0
        total_individual_loss = {}
        for loss_type, loss_fn in losses:
            total_individual_loss[loss_type] = 0.0
        all_preds = []
        all_gts = []
        for batch_number, batch in enumerate(tqdm(dataloader)):
            imgs_dict = batch[0]
            index = batch[1]
            imgs = imgs_dict[db_modality].to(model.device)
            masks = imgs_dict["seg_mask"].to(model.device).long()
            # import pdb; pdb.set_trace()
            masks = masks.squeeze(-3)
            preds = model.forward(imgs)
            all_preds.append(preds.argmax(dim=1).cpu())
            all_gts.append(masks.cpu())
            num_classes = preds.shape[1]
            valid_mask = masks >=0  # Assuming 255 is the ignore index
            # preds_new = preds.permute(0, 2, 3, 1).reshape(-1, num_classes)  # shape: (32, 3)
            valid_mask = valid_mask.reshape(-1)  # shape: (B*H*W)
            # new_masks = masks.reshape(-1)  # shape: (B*H*W)

            individual_losses = []
            for loss_type,loss_fn in losses:
                preds_reshaped = preds.permute(0, 2, 3, 1).reshape(-1, num_classes)
                masks_reshaped = masks.reshape(-1)
                loss = loss_fn(preds_reshaped[valid_mask], masks_reshaped[valid_mask])
                individual_losses.append((loss_type,loss))
                

            loss = sum(l[1] for l in individual_losses)

            if not test:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            for loss_type, individual_loss in individual_losses:
                total_individual_loss[loss_type] += individual_loss.item()

            if test and args.save_visualizations and semantic_id_to_rgb is not None and save_vis_dir is not None:
                viz_pred_masks(imgs, preds, masks, epoch, save_vis_dir, semantic_id_to_rgb,test,index)

        loss_str = "train_loss" if not test else "val_loss"

        log_dict = {"epoch": epoch+1, "lr": optimizer.param_groups[0]['lr']}
        total_loss /= len(dataloader)
        for loss_type, individual_loss in total_individual_loss.items():
            total_individual_loss[loss_type] /= len(dataloader)
            log_dict[f"{loss_str}/{loss_type}"] = total_individual_loss[loss_type]
    
        print(f"Epoch {epoch+1}/{args.epochs}  - {loss_str}: {total_loss:.4f}")
        log_dict[f"{loss_str}/total_loss"] = total_loss
        if args.wandb_use:
            wandb.log(log_dict)
        
        # Compute and log IoU metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        iou_per_class, miou, fwiou = compute_iou_metrics(all_preds, all_gts, num_classes)

        if args.wandb_use:
            wandb.log({
                f"{loss_str}/mIoU": miou,
                f"{loss_str}/FWIoU": fwiou,
                **{f"{loss_str}/IoU_class_{i}": v for i, v in enumerate(iou_per_class)},
                "epoch": epoch+1,
            })

def calculate_class_weights(dataloader):
    print("Calculating class weights...")
    class_counts = np.zeros(dataloader.dataset.semantic_classes)
    for batch_number, batch in enumerate(dataloader):
        masks = batch[0]["seg_mask"]
        unique, counts = np.unique(masks.numpy(), return_counts=True)
        unique = unique.astype(int)  # Ensure unique class indices are integers
        # import pdb; pdb.set_trace()
        unique_masked = unique[unique >= 0]  # Exclude ignore index (usually -1 or 255)
        counts_masked = counts[unique >= 0]
        class_counts[unique_masked] += counts_masked
    class_weights = 1.0 / (class_counts + 1e-5)  # Add small value to avoid division by zero
    class_weights = class_weights / np.sum(class_weights)  # Normalize to sum to 1
    print(f"Class counts: {class_counts}")
    return torch.tensor(class_weights, dtype=torch.float32)


def train_segmentation_pipeline(args):
    global db_modality
    assert args.modality != "rgb" or args.model_path ==""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.wandb_name = f"{args.dataset}_{args.modality}_{args.head_name}_{args.wandb_name}"
    if args.wandb_use:
        wandb.init(project="mm_segmentation", name=args.wandb_name, config=vars(args))

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)

    if args.dataset == "cart":
        print("Using CART dataset")
        train_seq_list = return_cart_split_segmentation_geographic("train", "socal", "thermal")
        val_seq_list = return_cart_split_segmentation_geographic("val", "socal", "thermal")
        db_modality = "thr_seg"
        train_dataset = CART(root_frame_dir=None, db_modality=db_modality, q_modality="seg_mask", datasets_folder=None, seq=train_seq_list, augment=True, seq_as_txt="thermal", crop_images=False)
        val_dataset = CART(root_frame_dir=None, db_modality=db_modality, q_modality="seg_mask", datasets_folder=None, seq=val_seq_list, augment=False, seq_as_txt="thermal", crop_images=False)
    elif args.dataset == "freiburg":
        print("Using FREIBURG dataset")
        train_seq_list = return_freiburg_split("train",segmentation=True)
        val_seq_list = return_freiburg_split("val",segmentation=True)
        data_root = "/ocean/projects/cis220039p/mdt2/datasets/freiburg"
        db_modality = "thr_seg"
        train_dataset = Freiburg(db_modality=db_modality, q_modality="seg_mask", datasets_folder=data_root, seq=train_seq_list, augment=False, crop_images=False)
        val_dataset = Freiburg(db_modality=db_modality, q_modality="seg_mask", datasets_folder=data_root, seq=val_seq_list, augment=False, crop_images=False)
    else:
        raise ValueError("Unsupported dataset")
    semantic_id_to_rgb = val_dataset.semantic_id_to_rgb
    if args.num_workers > 0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    class_weights = calculate_class_weights(train_loader).to(device)

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
        save_path = os.path.join(args.save_path, args.dataset, f'{time.strftime("%Y%m%d-%H%M%S")}_{args.wandb_name}')
        os.makedirs(save_path, exist_ok=True)
        backbone_path = args.model_path
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    save_vis_dir = os.path.join(save_path, "visualizations")
    os.makedirs(save_vis_dir, exist_ok=True)

    un_frozen_layer_index = args.un_frozen_layer_index
    if args.unfreeze_last_norm:
        un_frozen_layer_index.append("norm")

    if args.resume:
        model = MMDistillSegmentationModel(
            head_model=args.head_name,
            un_frozen_layer_index=args.un_frozen_layer_index,
            frozen_head=False,
            modality=args.modality,
            device=device,
            num_classes=train_dataset.semantic_classes,  # This will be updated later based on the dataset
            model_path = model_resume_path,
            upscale_method = args.upscale_method
        )
    else:
        model = MMDistillSegmentationModel(
            head_model=args.head_name,
            un_frozen_layer_index=args.un_frozen_layer_index,
            frozen_head=False,
            modality=args.modality,
            device=device,
            num_classes=train_dataset.semantic_classes,  # This will be updated later based on the dataset
            backbone_path=backbone_path,
            upscale_method = args.upscale_method
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
    optimizer = torch.optim.Adam(chain(model.unfrozen_parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Resumed optimizer state from {model_resume_path}")
    # scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr_lambda)

    # scheduler.step()
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("Training...")
        dataloader_loop(args, optimizer, model, train_loader, test=False, epoch=epoch, semantic_id_to_rgb=semantic_id_to_rgb, save_vis_dir=save_vis_dir, class_weights=class_weights)
        print("Validation ...")
        dataloader_loop(args, optimizer, model, val_loader, test=True, epoch=epoch, semantic_id_to_rgb=semantic_id_to_rgb, save_vis_dir=save_vis_dir, class_weights=class_weights)
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
    parser.add_argument('--dataset', default='cart', type=str,choices=['cart', 'freiburg'])
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--save_path', default='./checkpoints/segmentation', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', default='', type=str)
    parser.add_argument('--resume_epoch_num', default=0, type=int)
    parser.add_argument('--wandb_use', action='store_true')
    parser.add_argument('--head_name', default='linear', type=str)
    parser.add_argument('--model_path', default="", type=str)
    parser.add_argument('--wandb_name', default="", type=str)
    parser.add_argument('--train_easy', action='store_true')
    parser.add_argument('--pre_upscale', action='store_true')
    parser.add_argument('--save_visualizations', action='store_true')
    parser.add_argument('--crop_images', default=False, help='Rescale images during cropping')
    parser.add_argument('--loss_type', type=str, nargs='+', default=['weighted_ce'], choices=['ce', 'weighted_ce', 'dice'],help='Loss function to use')
    parser.add_argument('--un_frozen_layer_index', type=int, nargs='+', default=[],
                help='List of layer indices to unfreeze')
    parser.add_argument('--unfreeze_last_norm', action='store_true')
    parser.add_argument('--modality', default='thr', type=str, choices=['thr', 'rgb'], help='Loss function to use')
    parser.add_argument('--upscale_method', default='bilinear', type=str, choices=['bilinear', 'loftup'], help='Loss function to use')

    args = parser.parse_args()

    train_segmentation_pipeline(args)
