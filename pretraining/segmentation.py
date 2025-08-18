import argparse
import torch
import torch.nn as nn
from torch.nn.functional import  softmax, one_hot
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
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/custom_datasets") # Add the custom models directory to the path
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc')
from custom_datasets.cart_dataset import *
from custom_datasets.freiburg_dataset import *
from custom_datasets.mfnet_dataset import *
from custom_datasets.pst900_dataset import *
from custom_models.dinov2_segmentation_model import MMDistillSegmentationModel ,seg_head_str_to_dict
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


class DiceLoss(nn.Module):
    def __init__(self, skip_classes=None, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.skip_classes = skip_classes
        self.eps = eps

    def forward(self, pred, target):
        return dice_loss(pred, target, skip_classes=self.skip_classes, eps=self.eps)

def dice_loss(pred, target,skip_classes=None, eps=1e-6):
    # pred: (N, C) logits for masked pixels
    # target: (N,) integer labels for masked pixels

    N, C = pred.shape

    # 1) Convert logits to probabilities
    p = torch.softmax(pred, dim=1)                         # (N, C)

    # 2) Convert targets to one-hot encoding
    t = torch.nn.functional.one_hot(target, C).float()     # (N, C)

    # 3) Determine which classes to include
    if skip_classes is None:
        skip_classes = []

    # Find classes that actually appear in target (after masking)
    present = (t.sum(dim=0) > 0)

    # Remove skipped classes
    for sc in skip_classes:
        if 0 <= sc < C:
            present[sc] = False

    # If nothing is left, return 0 loss
    if present.sum() == 0:
        return pred.new_tensor(0.)

    # 4) Keep only the selected classes
    p_sel = p[:, present]
    t_sel = t[:, present]

    # 5) Compute Dice per class
    inter = (p_sel * t_sel).sum(dim=0)                     # intersection
    union = p_sel.sum(dim=0) + t_sel.sum(dim=0)            # sum of predictions + targets
    dice = (2 * inter + eps) / (union + eps)                # per-class Dice

    # 6) Return mean Dice loss over selected classes
    return 1 - dice.mean()



def dataloader_loop(args, optimizer, model, dataloader, test, epoch, semantic_id_to_rgb=None, save_vis_dir=None,class_weights=None):
    # import pdb; pdb.set_trace()
    global db_modality
    losses = []
    skip_classes = dataloader.dataset.skip_classes_in_segmentation() if args.skip_classes else []

    for loss_type in args.loss_type:
        if loss_type not in ['ce', 'weighted_ce', 'dice']:
            raise ValueError(f"Unsupported loss_type type: {loss_type}")
        if loss_type == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'weighted_ce':
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == 'dice':
            loss_fn = DiceLoss(skip_classes=skip_classes)
        losses.append((loss_type,loss_fn))

    
    if test:
        model.eval()
    else:
        model.train()
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
            masks = masks.squeeze(-3)
            preds = model.forward(imgs)
            all_preds.append(preds.argmax(dim=1).cpu())
            all_gts.append(masks.cpu())
            num_classes = preds.shape[1]
            valid_mask = (masks >= 0).reshape(-1)  # shape: (B*H*W)

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

            torch.cuda.empty_cache()
            import gc; gc.collect()
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
                "epoch": epoch,
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
    print(f"Class counts: {class_counts}  min count: {np.min(class_counts)}  max count: {np.max(class_counts)}")
    return torch.tensor(class_weights, dtype=torch.float32)


def train_segmentation_pipeline(args):
    global db_modality
    assert args.modality != "rgb" or args.model_path ==""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.wandb_name = f"{args.dataset}_{args.modality}_{args.head_name}_{'_'.join(args.loss_type)}_{args.upscale_method}_{args.wandb_name}"
    if args.augment:
        args.thermal_segmentation_augmentation = sorted(set(args.thermal_segmentation_augmentation))  # Sort to ensure consistent naming
        args.wandb_name += "_augmented" + "_".join(args.thermal_segmentation_augmentation)
    if args.dropout_prob > 0:
        args.wandb_name += f"_dropout{args.dropout_prob}"
    if args.skip_classes:
        args.wandb_name += "_skip_classes"
    if args.wandb_use:
        wandb.init(project="mm_segmentation", name=args.wandb_name, config=vars(args))

    if "cart" in args.dataset:
        print("Using CART dataset")

        type_of_split = args.dataset.split("_")[-1]
        if type_of_split == "random":
            train_seq_list = return_cart_split_segmentation_random("train")
            val_seq_list = return_cart_split_segmentation_random("val")
        elif type_of_split == "geographic":
            train_seq_list = return_cart_split_segmentation_geographic("train", "socal", "thermal") + return_cart_split_segmentation_geographic("val", "socal", "thermal")
            val_seq_list = return_cart_split_segmentation_geographic(split="val",area="northcarolina",mode="thermal") + return_cart_split_segmentation_geographic(split="val",area="kentucky",mode="thermal")
        else:
            raise ValueError("Unsupported CART split type. Use 'random' or 'geographic'.")
        db_modality = "thr_seg"
        train_dataset = CART(args=args,root_frame_dir=None, db_modality=db_modality, q_modality="seg_mask", datasets_folder=None, seq=train_seq_list, augment=args.augment, seq_as_txt="thermal", crop_images=False)
        val_dataset = CART(args=args,root_frame_dir=None, db_modality=db_modality, q_modality="seg_mask", datasets_folder=None, seq=val_seq_list, augment=False, seq_as_txt="thermal", crop_images=False)
    elif args.dataset == "freiburg":
        print("Using FREIBURG dataset")
        train_seq_list = return_freiburg_split("train",segmentation=True)
        val_seq_list = return_freiburg_split("val",segmentation=True)
        data_root = "/ocean/projects/cis220039p/mdt2/datasets/freiburg"
        db_modality = "thr_seg"
        train_dataset = Freiburg(args=args,db_modality=db_modality, q_modality="seg_mask", datasets_folder=data_root, seq=train_seq_list, augment=args.augment, crop_images=False)
        val_dataset = Freiburg(args=args,db_modality=db_modality, q_modality="seg_mask", datasets_folder=data_root, seq=val_seq_list, augment=False, crop_images=False)
    elif args.dataset == "mfnet":
        print("Using MFNet dataset")
        train_seq_list = return_mfnet_split("train")
        val_seq_list = return_mfnet_split("test")
        db_modality = "thr"
        train_dataset = MFNet(args=args,root_frame_dir=None, db_modality=db_modality, q_modality="seg_mask", datasets_folder=None, seq=train_seq_list, augment=args.augment, crop_images=False)
        val_dataset = MFNet(args=args,root_frame_dir=None, db_modality=db_modality, q_modality="seg_mask", datasets_folder=None, seq=val_seq_list, augment=False, crop_images=False)
    elif args.dataset == "pst900":
        print("Using PST900 dataset")
        train_seq_list = return_pst900_split("train")
        val_seq_list = return_pst900_split("test")
        db_modality = "thr"
        train_dataset = PST900(args=args,root_frame_dir=None, db_modality=db_modality, q_modality="seg_mask", datasets_folder=None, seq=train_seq_list, augment=args.augment, crop_images=False)
        val_dataset = PST900(args=args,root_frame_dir=None, db_modality=db_modality, q_modality="seg_mask", datasets_folder=None, seq=val_seq_list, augment=False, crop_images=False)

    else:
        raise ValueError("Unsupported dataset")

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
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
        model = MMDistillSegmentationModel(args= args,
            backbone_model_type= args.model_type,
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
        model = MMDistillSegmentationModel(args = args,
            backbone_model_type = args.model_type,
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
    print("weight decay", args.weight_decay)
    optimizer = torch.optim.AdamW(chain(model.unfrozen_parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        }, os.path.join(save_path, f"model{epoch}.pth"))

        # scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train segmentation head on DINOv2 features')
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--dataset', default='cart', type=str,choices=['cart_geographic', 'cart_random', 'freiburg', 'mfnet','pst900'], help='Dataset to use for training')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--save_path', default='./checkpoints/segmentation', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', default='', type=str)
    parser.add_argument('--resume_epoch_num', default=0, type=int)
    parser.add_argument('--wandb_use', default = True, type=bool, help='Use wandb for logging')
    parser.add_argument('--head_name', default='linear', type=str)
    parser.add_argument('--model_path', default="", type=str)
    parser.add_argument('--wandb_name', required=True, type=str)
    parser.add_argument('--save_visualizations', action='store_true')
    parser.add_argument('--crop_images', default=False, help='Rescale images during cropping')
    parser.add_argument('--loss_type', type=str, nargs='+', default=['weighted_ce'], choices=['ce', 'weighted_ce', 'dice'],help='Loss function to use')
    parser.add_argument('--un_frozen_layer_index', type=int, nargs='+', default=[],
                help='List of layer indices to unfreeze')
    parser.add_argument('--unfreeze_last_norm', action='store_true')
    parser.add_argument('--modality', default='thr', type=str, choices=['thr', 'rgb'], help='Loss function to use')
    parser.add_argument('--upscale_method', default='bilinear', type=str, choices=['bilinear', 'loftup','pre_bilinear'], help='Loss function to use')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--model_type',default="dinov2_vitb14", type=str, choices=['dinov2_vitb14', 'dinov2_vitb14_reg'], help='Loss function to use')
    parser.add_argument('--dropout_prob', default=0., type=float)
    parser.add_argument('--thermal_segmentation_augmentation', type=str, nargs='+', default=['hflip'], choices=['hflip', 'vflip', 'brightness_contrast',"noise","gamma","crop_with_random_ratio","crop_with_fixed_ratio"],help='Loss function to use')
    parser.add_argument('--skip_classes', action='store_true')
    args = parser.parse_args()

    assert args.model_type or args.model_path, "Please provide a model type or a model path to load the backbone."

    train_segmentation_pipeline(args)
