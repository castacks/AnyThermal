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
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/custom_datasets") # Add the custom models directory to the path

from custom_models.mmdistill_dinov2_model import MMDistillVPRModel
from torch.optim import Adam
from contextlib import nullcontext
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.miners import DistanceWeightedMiner
import matplotlib.pyplot as plt
import random
import networkx as nx
import gc
from custom_datasets.multi_dataset_loader import *
from itertools import chain
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import TripletMarginLoss
import faiss
import faiss.contrib.torch_utils
from utilities import *

def get_all_triplets(indices, positive_index_per_query, extra_margin_positive_index_per_query, device):
    """
    Fully GPU-tensorized, batch-local triplet miner:
    - Mines all (anchor, positive, negative) triplets within the batch.
    - No Python loops.
    - Only considers positives/negatives present in the current batch.

    Args:
        indices (torch.Tensor): [B] tensor of dataset indices for current batch.
        positive_index_per_query (list of lists): Positive dataset indices for each dataset index.
        extra_margin_positive_index_per_query (list of lists): Expanded positives (excluded from negatives).
        device (torch.device): Target device (e.g., CUDA).

    Returns:
        a_idx, p_idx, n_idx: Tensors of anchor, positive, negative batch indices.
    """
    B = indices.size(0)  # batch size
    indices = indices.to(device)

    # --- Build positive mask ---
    # For each anchor, positives are in positive_index_per_query[anchor_dataset_idx]
    positive_mask = torch.zeros((B, B), dtype=torch.bool, device=device)  # Shape [B, B]
    for anchor_batch_idx in range(B):
        anchor_dataset_idx = indices[anchor_batch_idx].item()
        pos_dataset_indices = positive_index_per_query[anchor_dataset_idx]
        if pos_dataset_indices:  # If any positives exist
            # Create mask of which batch samples are positives
            positive_mask[anchor_batch_idx] = torch.isin(indices, torch.tensor(pos_dataset_indices, device=device))
    positive_mask.fill_diagonal_(False)  # Remove self-positives

    # --- Build negative mask ---
    # For each anchor, negatives are NOT in extra_margin_positive_index_per_query[anchor_dataset_idx]
    negative_mask = torch.ones((B, B), dtype=torch.bool, device=device)  # Start with all True
    for anchor_batch_idx in range(B):
        anchor_dataset_idx = indices[anchor_batch_idx].item()
        excl_dataset_indices = extra_margin_positive_index_per_query[anchor_dataset_idx]
        if len(excl_dataset_indices) > 0:
            negative_mask[anchor_batch_idx] &= ~torch.isin(indices, torch.tensor(excl_dataset_indices, device=device))
    negative_mask.fill_diagonal_(False)  # Remove self-negatives

    # --- Get all (anchor, positive) pairs ---
    a_p_pairs = torch.nonzero(positive_mask, as_tuple=False)  # Shape [N_pos, 2]
    if a_p_pairs.size(0) == 0:
        # No valid positives found in this batch
        return None
    anchors = a_p_pairs[:, 0]     # Anchor batch indices
    positives = a_p_pairs[:, 1]   # Positive batch indices

    # --- For each (a, p), get all negatives ---
    neg_candidates_mask = negative_mask[anchors]  # Shape [N_pos, B]
    neg_idx_pairs = torch.nonzero(neg_candidates_mask, as_tuple=False)  # Shape [N_neg, 2]

    if neg_idx_pairs.size(0) == 0:
        # No valid negatives found in this batch
        return None

    # Repeat anchors and positives for each negative
    a_repeat = anchors[neg_idx_pairs[:, 0]]  # Anchor batch indices
    p_repeat = positives[neg_idx_pairs[:, 0]]  # Positive batch indices
    n_repeat = neg_idx_pairs[:, 1]  # Negative batch indices

    return a_repeat, p_repeat, n_repeat

def compute_recall_at_k(query_feats, db_feats, ground_truth, ks=[1, 5, 10], exclude_self=True):
    
    # sim = torch.matmul(query_feats, db_feats.T)  # (N, M)
    recall = {k: 0 for k in ks}
    # top_k = torch.topk(sim, k=max(ks), dim=1).indices

    # try:
    #     if use_gpu:
    #         index = faiss.IndexFlatIP(d)
    #         res = faiss.StandardGpuResources()
    #         index = faiss.index_cpu_to_gpu(res, 0, index)
    #         print("🔋 FAISS running on GPU")
    #     else:
    #         raise RuntimeError("GPU disabled by user")

    # except (ImportError, AttributeError, RuntimeError) as e:
    #     print(f"⚠️ FAISS GPU not available, falling back to CPU. Reason: {e}")
    res = faiss.StandardGpuResources()  # uses all available GPUs

    d = db_feats.shape[1]
    index = faiss.GpuIndexFlatL2(res, d)
    # index = faiss.IndexFlatIP(d)
    
    index.add(db_feats.detach().numpy())
    _, indices = index.search(query_feats.detach().numpy(), max(ks)+1)

    total_valid = 0
    for i, positives in enumerate(ground_truth):
        if exclude_self:
            positives = set([p for p in positives if p != i])
        if not positives:
            continue
        total_valid += 1
        for k in ks:
            if exclude_self:
                retrieved = indices[i][:k+1]
                retrieved = [r for r in retrieved if r != i]  # Exclude self from retrieved indices
                retrieved = retrieved[:k]
            else:
                retrieved = indices[i][:k]
            if any(pred.item() in positives for pred in retrieved):
                recall[k] += 1
    # import pdb; pdb.set_trace()  # Debugging line to inspect the recall values
    return {f"recall@{k}": recall[k] / total_valid if total_valid > 0 else 0.0 for k in ks}


def run(args,model_dict, dataloader, optimizer, device, epoch, train=True):
    mode = "train" if train else "val"
    
    # create_label_graph = False
    # if args.miner_type == "multi_similarity":
    #     loss_fn = MultiSimilarityLoss()
    #     create_label_graph = True
    # elif args.miner_type == "gps_cosine":
    loss_fn = TripletMarginLoss(margin=args.margin, p=2, reduction='none')
    total_loss_vpr = 0
    total_loss_vpr_updated = 0
    total_loss_allignment = 0
    total_loss_allignment_updated = 0
    memory_feats, memory_labels = [], []
    
    all_ground_truth = [[] for _ in range(len(dataloader.dataset))]
    all_rgb_feats = torch.zeros((len(dataloader.dataset), args.features_dim))
    all_thr_feats = torch.zeros((len(dataloader.dataset), args.features_dim))


    db_modality = args.teacher_modality
    q_modality = args.student_modality
    positive_index_per_query = np.array(dataloader.dataset.soft_positives, dtype=object)
    extra_margin_positive_index_per_query = np.array(dataloader.dataset.extra_margin_soft_positives, dtype=object)
    

    for batch_item in tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}"):
        batch, _ = batch_item["item"]
        indices = batch_item["batch_id"].tolist()
        rgb = batch[db_modality].to(device)
        thermal = batch[q_modality].to(device)

        log_dict = {}
    

        with torch.no_grad() if (not train or epoch ==0) else nullcontext():
            feats_rgb = model_dict["rgb"].extract_feature(rgb, test=False)
            feats_thr = model_dict["thr"].extract_feature(thermal, test=False)
            
            allignmnet_loss = 1 - F.cosine_similarity(feats_rgb, feats_thr, dim=1).mean()

            log_dict.update({f"{mode}/allignmnet_loss": allignmnet_loss.item()})

            
            feats = torch.cat([feats_rgb, feats_thr], dim=0)
            cat_indices  = torch.cat([torch.tensor(indices, device=device), torch.tensor(indices, device=device)], dim=0)
            triplets = get_all_triplets(cat_indices, positive_index_per_query, extra_margin_positive_index_per_query,feats.device)

            if triplets:
                a, p, n = triplets
                all_loss = loss_fn(feats[a], feats[p], feats[n])
                active_losses = all_loss[all_loss > 0]
                log_dict.update({f"{mode}/num_active_triplets": len(active_losses), f"{mode}/num_triplets": len(a)})
                log_dict.update({f"{mode}/triplet_mean": all_loss.mean().item(),f"{mode}/active_triplet_mean": active_losses.mean().item()})
                loss = active_losses.mean() if len(active_losses) > 0 else torch.tensor(0.0, device=device)
            else:
                loss = 0
            final_loss = loss + allignmnet_loss
            if train and epoch!=0:
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()

            if loss != 0:
                total_loss_vpr += loss.item()
                total_loss_vpr_updated += 1
            total_loss_allignment += allignmnet_loss.item()
            total_loss_allignment_updated += 1

            wandb.log(log_dict)

            


        assert torch.all(all_rgb_feats[indices] == 0), "all_rgb_feats should be zero before filling"
        assert torch.all(all_thr_feats[indices] == 0), "all_thr_feats should be zero before filling"

        all_rgb_feats[indices] = feats_rgb.cpu()
        all_thr_feats[indices] = feats_thr.cpu()
        for batch_idx in indices:

            if all_ground_truth[batch_idx] != []:
                print(f"Overwriting ground truth for index {batch_idx} in {mode} epoch {epoch}")
                import pdb; pdb.set_trace()  # Debugging line to inspect the ground truth

            all_ground_truth[batch_idx] = positive_index_per_query[batch_idx]

    for idx in range(len(all_ground_truth)):
        if not all_ground_truth[idx]:
            print(f"Warning: No ground truth found for index {idx} in {mode} epoch {epoch}. This may indicate an issue with the dataset or dataloader.")
            import pdb; pdb.set_trace()

    db_feats = F.normalize(all_rgb_feats, dim=1)
    query_feats = F.normalize(all_thr_feats, dim=1)
    recall_metrics = compute_recall_at_k(query_feats, db_feats, all_ground_truth, exclude_self=True)

    # Optional retrieval visualization
    if args.debug_viz:
        sim = torch.matmul(query_feats, db_feats.T)
        top_k = torch.topk(sim, k=5, dim=1).indices
        for idx in random.sample(range(len(query_feats)), min(5, len(query_feats))):
            q_img = dataloader.dataset.__getitem__(idx)['thr'].permute(1, 2, 0)
            retrieved_imgs = [dataloader.dataset.__getitem__(j)['rgb'].permute(1, 2, 0) for j in top_k[idx]]
            fig, axs = plt.subplots(1, 6, figsize=(15, 3))
            axs[0].imshow(q_img)
            axs[0].set_title("Query")
            for i in range(5):
                axs[i + 1].imshow(retrieved_imgs[i])
                axs[i + 1].set_title(f"Top-{i + 1}")
            wandb.log({f"{mode}/retrieval_{idx}": wandb.Image(fig)})
            plt.close(fig)
    

    return total_loss_vpr / total_loss_vpr_updated , total_loss_allignment / total_loss_allignment_updated , recall_metrics


def build_head_dict(arch_name):
    if arch_name == "netvlad":
        print(f"Using NetVLAD aggregation head for {arch_name}")
        default_agg_dict = {
            "agg_arch":'NetVLAD',
            "agg_config":{
                'num_clusters': 64,
            }
        }
        return default_agg_dict
    elif arch_name == "salad":
        print(f"Using SALAD aggregation head for {arch_name}")
        default_agg_dict = {
            "agg_arch":'SALAD',
            "agg_config":{
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            }
        }
        return default_agg_dict
    else:
        raise ValueError(f"Unknown head architecture: {arch_name}")

def initialise_netvlad_head(model_dict, dataloader, device):
    """
    Initializes the NetVLAD head for both RGB and thermal models.
    This is done by passing a batch of data through the model to set up the head.
    """
    print("Initialising NetVLAD head...")

    model_dict['rgb'].head[1][0].initialize_netvlad_layer(
        args, dataloader.dataset, model_dict['rgb'],'rgb')
    model_dict['thr'].head[1][0].initialize_netvlad_layer(
        args, dataloader.dataset, model_dict['thr'],'thr')

    

def main(args):
    dataset_name = "_".join(args.dataset)
    if args.eval_dataset:
        dataset_name += "_eval_" + "_".join(args.eval_dataset)
    args.save_dir = os.path.join(args.save_dir, dataset_name)
    #append date and time 
    args.save_dir = os.path.join(args.save_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    wandb_name = f"{args.name}_{dataset_name}_{args.head_arch}_margin_{args.margin}_same_backbone{args.same_backbone}_frozen_backbone_{args.frozen_backbone}_un_frozen_layer_index_{'_'.join(map(str, args.un_frozen_layer_index))}"
    wandb.init(project="mm_vpr", name=wandb_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader = build_dataset(args)
    print("Train dataset size: ", len(train_dataloader.dataset))
    print("Val dataset size: ", len(val_dataloader.dataset))

    agg_dict = build_head_dict(args.head_arch)

    if args.same_backbone:
        raise NotImplementedError("Same backbone is not implemented yet")
        rgb_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,modality='rgb', device=device,backbone_path = args.backbone_path,head_config=agg_dict)
        thr_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,model=rgb_model.model,modality='thr', device=device,head_config=agg_dict)
        trainable_params = thr_model.trainable_params()
    else:
        rgb_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,modality='rgb', device=device,backbone_model_type="dinov2_vitb14",head_config=agg_dict)
        thr_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,backbone_path = args.backbone_path,modality='thr', device=device,head_config=agg_dict)

        trainable_params = chain(thr_model.trainable_params(), rgb_model.trainable_params())
        
    optimizer = Adam(
        trainable_params,
        lr=0.001, weight_decay=0.001
    )
    
    model_dict = {"rgb": rgb_model, "thr": thr_model}

    # initialise_netvlad_head(model_dict,train_dataloader,device)

    

    for epoch in range(0,args.epochs+1):

        if epoch % args.save_interval == 0:

            save_dict = {"thermal_state_dict": thr_model.state_dict()}
            if not args.same_backbone:
                save_dict["rgb_state_dict"] = rgb_model.state_dict()

            torch.save(save_dict, os.path.join(args.save_dir, f"model_{epoch}.pth"))

        train_loss_vpr, train_loss_align, train_recall_metrics = run(args,model_dict, train_dataloader, optimizer, device, epoch, train=True)
        log_dict = {"epoch": epoch, "train/avg_loss_vpr": train_loss_vpr, "train/avg_loss_align": train_loss_align}
        for k, v in train_recall_metrics.items():
            log_dict.update({f"train/{k}": v})
        wandb.log(log_dict)
        
        with torch.no_grad():
            val_loss_vpr, val_loss_align, val_recall_metrics = run(args,model_dict, val_dataloader, optimizer, device, epoch, train=False)
            log_dict = {"epoch": epoch, "val/avg_loss_vpr": val_loss_vpr, "val/avg_loss_align": val_loss_align}
            for k, v in val_recall_metrics.items():
                log_dict.update({f"val/{k}": v})
            wandb.log(log_dict)

        print(f"Epoch {epoch} - Train Loss: {train_loss_vpr+train_loss_align:.4f} | Val Loss: {val_loss_vpr+val_loss_align:.4f}")    
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="mmdistill")
    parser.add_argument('--dataset', type=str, nargs='+',
                    help='List of datasets to use in training and eval')
    parser.add_argument('--eval_dataset', default=[],type=str, nargs='+',
                    help='List of datasets to use in training and eval')    
    parser.add_argument('--backbone_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default="checkpoints/vpr")
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument("--augment", action='store_true', help="Use data augmentation for training")
    parser.add_argument('--train_num_workers', type=int, default=4)
    parser.add_argument('--eval_num_workers', type=int, default=4)

    parser.add_argument('--train', default=True,type=bool, help='Mode to build datasets and dataloaders')
    parser.add_argument('--use_odom', default=True,type=bool, help='Mode to build datasets and dataloaders')
    parser.add_argument('--rescale_during_crop', default=False, help='Rescale images during cropping')
    parser.add_argument('--teacher_modality', default='rgb', type=str, help='modality which will be frozen unless "unfreeze teacher" is true')
    parser.add_argument('--student_modality', default='thr', type=str, help='modality for which encoder has to be trained')
    parser.add_argument('--vpr_test', default=False, help='Rescale images during cropping')
    parser.add_argument('--same_backbone', action='store_true', help='Rescale images during cropping')
    parser.add_argument('--un_frozen_layer_index', type=int, nargs='+', default=[],
                    help='List of layer indices to unfreeze')
    parser.add_argument('--head_arch', type=str, choices=['netvlad', 'salad'], default='netvlad')
    parser.add_argument('--debug_viz', action='store_true', help='Enable Top-K retrieval visualization')
    parser.add_argument('--intra_dataset_batch', type=bool, default=True, help='Enable Top-K retrieval visualization')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss')
    parser.add_argument('--no_crop_images', dest='crop_images', action='store_false', help='Disable image cropping')
    parser.set_defaults(crop_images=True)
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--conv_output_dim', type=int, default=-1, help='Disable shuffling of dataset')
    parser.add_argument('--fc_output_dim', type=int, default=-1, help='Disable shuffling of dataset')
    parser.add_argument('--add_bn', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--cart_split', default='vpr',type=str, help='Task to run, currently only vpr is supported')
    parser.add_argument('--debug', action='store_true', help='Disable shuffling of dataset')
    args = parser.parse_args()

    args.frozen_backbone = True if args.un_frozen_layer_index == [] else False
 
    assert args.conv_output_dim<0 or args.fc_output_dim<0, "conv_output_dim and fc_output_dim cannot be both set."
    main(args)
