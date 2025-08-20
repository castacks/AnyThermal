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

from custom_models.dinov2_vpr_model import MMDistillVPRModel
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
import yaml
import pandas as pd

def batched_dot_similarity(embeddings, a_idx, b_idx, chunk_size=100000):
    """
    Compute dot products (cosine similarity) between normalized embeddings[a_idx] and embeddings[b_idx] in chunks.
    Returns: Tensor of shape (len(a_idx),)
    """
    results = []
    for start in range(0, len(a_idx), chunk_size):
        end = start + chunk_size
        a_emb = embeddings[a_idx[start:end]]  # (chunk, D)
        b_emb = embeddings[b_idx[start:end]]  # (chunk, D)
        sim = torch.sum(a_emb * b_emb, dim=1)  # (chunk,)
        results.append(sim)
    return torch.cat(results, dim=0)

def pick_mixed_negatives(s_ap, s_neg, num_per_ap=3, margin=0.1, hard_frac=0.5, temp=10.0):
    # s_ap: scalar; s_neg: (M,) tensor of cosine sims
    if s_neg.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=s_neg.device)
    vals, idx = torch.sort(s_neg, descending=True)
    lo, hi = s_ap - margin, s_ap
    semi_mask = (s_neg >= lo) & (s_neg < hi)
    semi_idx = torch.nonzero(semi_mask, as_tuple=False).squeeze(1)

    n_hard = max(1, int(round(num_per_ap*hard_frac)))
    n_semi = max(0, num_per_ap - n_hard)
    hard_idx = idx[:min(n_hard, idx.numel())]

    if n_semi > 0 and semi_idx.numel() > 0:
        probs = torch.softmax(s_neg[semi_idx]*temp, dim=0)
        semi_take = min(n_semi, semi_idx.numel())
        semi_idx = semi_idx[torch.multinomial(probs, semi_take, replacement=False)]
        pick = torch.unique(torch.cat([hard_idx, semi_idx], 0))[:num_per_ap]
    else:
        pick = hard_idx[:num_per_ap]
    return pick

def get_top_n_hardest_triplets_cosine(triplets, embeddings, margin, top_n, verbose=True):
    """
    Args:
        triplets: tuple of (anchor_indices, positive_indices, negative_indices), each a tensor of shape (T,)
        embeddings: tensor of shape (N, D), assumed L2-normalized
        margin: float, margin for cosine-based triplet loss
        top_n: int, number of hardest triplets to keep per (anchor, positive) pair
        verbose: bool, whether to print timing for each step

    Returns:
        Tuple of tensors: (anchor_indices, positive_indices, negative_indices) after mining
    """
    times = {}
    start = time.perf_counter()

    a_idx, p_idx, n_idx = triplets
    T = a_idx.shape[0]

    t1 = time.perf_counter()
    # import pdb; pdb.set_trace()  # Debugging line to inspect the triplets
    sim_ap = batched_dot_similarity(embeddings, a_idx, p_idx, chunk_size=10000).cpu()
    sim_an = batched_dot_similarity(embeddings, a_idx, n_idx, chunk_size=10000).cpu()
    losses = (sim_an - sim_ap + margin).clamp(min=0.0)
    times['compute_loss'] = time.perf_counter() - t1

    t2 = time.perf_counter()
    df = pd.DataFrame({
        'a': a_idx.cpu().numpy(),
        'p': p_idx.cpu().numpy(),
        'n': n_idx.cpu().numpy(),
        'loss': losses.cpu().numpy(),
    })
    times['create_dataframe'] = time.perf_counter() - t2

    t3 = time.perf_counter()
    df_sorted = df.sort_values(['a', 'p', 'loss'], ascending=[True, True, False])
    times['sort_by_loss'] = time.perf_counter() - t3

    t4 = time.perf_counter()
    df_topn = df_sorted.groupby(['a', 'p'], sort=False).head(top_n)
    times['groupby_head'] = time.perf_counter() - t4

    t5 = time.perf_counter()
    device = embeddings.device
    a_out = torch.tensor(df_topn['a'].values, dtype=torch.long, device=device)
    p_out = torch.tensor(df_topn['p'].values, dtype=torch.long, device=device)
    n_out = torch.tensor(df_topn['n'].values, dtype=torch.long, device=device)
    times['convert_to_tensor'] = time.perf_counter() - t5

    total = time.perf_counter() - start
    times['total'] = total

    if verbose:
        print("Time Profiling (seconds):")
        for k, v in times.items():
            print(f"  {k:<20}: {v:.6f}")

    return a_out, p_out, n_out

class RadiusContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0, p: float = 2, reduction: str = 'mean'):
        """
        GPS-aware contrastive loss:
        - Pulls positives (label=1) close
        - Pushes negatives (label=0) farther than margin

        Args:
            margin (float): Minimum distance required between query and negative.
            p (float): Power for distance calculation (e.g., 2 for Euclidean).
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.margin = margin
        self.p = p
        assert self.p in [1, 2], "Only p = 1 or p = 2 are supported."
        self.reduction = reduction

    def forward(self, query: torch.Tensor, other: torch.Tensor, label: torch.Tensor):
        """
        Args:
            query: [N, D] query embeddings
            other: [N, D] paired embeddings (positive or negative)
            label: [N] binary labels (1 = positive, 0 = negative)
        Returns:
            loss: [N] if reduction='none', scalar otherwise
        """
        dists = F.pairwise_distance(query, other, p=2)  # shape: [N]

        pos_loss = label * dists.pow(self.p)
        neg_loss = (1 - label) * F.relu(self.margin - dists).pow(self.p)
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss  # shape: [N]

def build_pos_neg_masks(indices, positive_index_per_query, extra_margin_positive_index_per_query, device):
    """
    Builds positive and negative masks for the given indices based on the provided positive and negative index lists.

    Args:
        indices (torch.Tensor): [B] tensor of dataset indices for current batch.
        positive_index_per_query (list of lists): Positive dataset indices for each dataset index.
        extra_margin_positive_index_per_query (list of lists): Expanded positives (excluded from negatives).
        device (torch.device): Target device (e.g., CUDA).

    Returns:
        positive_mask, negative_mask: Tensors of shape [B, B] indicating positive and negative pairs.
    """
    B = indices.size(0)
    indices = indices.to(device)

    # --- Build positive mask ---
    positive_mask = torch.zeros((B, B), dtype=torch.bool, device=device)
    for anchor_batch_idx in range(B):
        anchor_dataset_idx = indices[anchor_batch_idx].item()
        pos_dataset_indices = positive_index_per_query[anchor_dataset_idx]
        if len(pos_dataset_indices) > 0:
            positive_mask[anchor_batch_idx] = torch.isin(indices, torch.tensor(pos_dataset_indices, device=device))
    positive_mask.fill_diagonal_(False)  # Remove self-positives

    # --- Build negative mask ---
    negative_mask = torch.ones((B, B), dtype=torch.bool, device=device)
    for anchor_batch_idx in range(B):
        anchor_dataset_idx = indices[anchor_batch_idx].item()
        excl_dataset_indices = extra_margin_positive_index_per_query[anchor_dataset_idx]
        if len(excl_dataset_indices) > 0:
            negative_mask[anchor_batch_idx] &= ~torch.isin(indices, torch.tensor(excl_dataset_indices, device=device))
    negative_mask.fill_diagonal_(False)  # Remove self-negatives

    return positive_mask, negative_mask

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
    positive_mask, negative_mask = build_pos_neg_masks(
        indices, positive_index_per_query, extra_margin_positive_index_per_query, device
    )

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

def generate_pairs_with_labels(indices, positive_index_per_query, extra_margin_positive_index_per_query, device):

    positive_mask, negative_mask = build_pos_neg_masks(
        indices, positive_index_per_query, extra_margin_positive_index_per_query, device
    )

    # --- Get all (anchor, positive) pairs ---
    a_p_pairs = torch.nonzero(positive_mask, as_tuple=False)  # Shape [N_pos, 2]
    if a_p_pairs.size(0) == 0:
        return None, None

    a_n_pairs = torch.nonzero(negative_mask, as_tuple=False)  # Shape [N_neg, 2]
    if a_n_pairs.size(0) == 0:
        return None, None
    
    labels = [torch.ones((len(a_p_pairs),), dtype=torch.long, device=device)] + \
            [torch.zeros((len(a_n_pairs),), dtype=torch.long, device=device)]
    
    labels = torch.cat(labels, dim=0)

    pairs = torch.cat([a_p_pairs, a_n_pairs], dim=0)  # Shape [N_pos + N_neg, 2]

    return pairs,labels



def compute_recall_at_k(query_feats, db_feats, ground_truth, ks=[1, 5, 10], exclude_self=True):
    
    recall = {k: 0 for k in ks}
    d = db_feats.shape[1]
    print("Computing recall at k..., initializing FAISS index")
    # index = faiss.IndexFlatL2(d)  # Using L2 distance for the index
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, d)
    print("FAISS index initialized")
    index.add(db_feats)
    print("DB features added to FAISS index")
    _, indices = index.search(query_feats, max(ks)+1)
    print("Search completed, computing recall")

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

# -------------------------
# Curriculum margin helper
# -------------------------
def compute_curriculum_margin(epoch: int, mode: str,
                              margin_start: float, margin_end: float,
                              ramp_epochs: int,
                              last_val_metrics: dict = None) -> float:
    """
    Returns the margin to use this epoch.
    mode='epoch': linear ramp for first `ramp_epochs` then clamp.
    mode='metric': simple example policy based on recall@1 (customize as needed).
    """
    if mode == 'epoch':
        if ramp_epochs <= 0:
            return margin_end
        t = min(max(epoch, 0), ramp_epochs)
        alpha = t / float(ramp_epochs)
        return margin_start + alpha * (margin_end - margin_start)
    elif mode == 'metric':
        r1 = (last_val_metrics or {}).get('recall@1', None)
        if r1 is None:
            return margin_start
        return margin_end if r1 >= 0.6 else 0.5 * (margin_start + margin_end)
    else:
        return margin_end


def run(args,model_dict, dataloader, optimizer, device, epoch, train=True, current_margin=None):
    mode = "train" if train else "val"

    if "triplet" in args.loss_type or "hard_triplet" in args.loss_type:
        assert current_margin is not None, "current_margin must be provided for triplet losses"
        loss_fn = TripletMarginLoss(margin=current_margin, p=2, reduction='none')
    elif "pair" in args.loss_type:
        loss_fn = RadiusContrastiveLoss(margin=args.margin, p=2, reduction='none')
    else:
        raise ValueError(f"Neither triplet nor pair loss specified in args.loss_type: {args.loss_type}")
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
    positive_index_per_query = np.array(dataloader.dataset.hard_positives_per_query, dtype=object)
    extra_margin_positive_index_per_query = np.array(dataloader.dataset.extra_margin_soft_positives, dtype=object)
    
    # scaler = torch.cuda.amp.GradScaler()
    for batch_item in tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}"):
        optimizer.zero_grad()

        batch, _ = batch_item["item"]
        indices = batch_item["batch_id"].tolist()
        rgb = batch[db_modality].to(device)
        thermal = batch[q_modality].to(device)

        log_dict = {}
    

        with torch.no_grad() if (not train or epoch ==0) else nullcontext():
            with nullcontext():
                feats_rgb = model_dict["rgb"].extract_feature(rgb, test=False)
                feats_thr = model_dict["thr"].extract_feature(thermal, test=False)
            
                if "allign" in args.loss_type:
                    allignmnet_loss = 1 - F.cosine_similarity(feats_rgb, feats_thr, dim=1).mean()

                    log_dict.update({f"{mode}/allignmnet_loss": allignmnet_loss.item()})

                
                feats = torch.cat([feats_rgb, feats_thr], dim=0)
                cat_indices  = torch.cat([torch.tensor(indices, device=device), torch.tensor(indices, device=device)], dim=0)

                if "triplet" in args.loss_type or "hard_triplet" in args.loss_type:
                    
                    with torch.no_grad():
                        triplets = get_all_triplets(cat_indices, positive_index_per_query, extra_margin_positive_index_per_query,feats.device)

                        if triplets and "hard_triplet" in args.loss_type:
                            triplets = get_top_n_hardest_triplets_cosine(
                                triplets, feats, current_margin, args.num_negatives_per_positive,verbose=False)
                    if triplets:
                        num_triplets = len(triplets[0])
                        a, p, n = triplets

                        all_loss = torch.zeros((len(a),), device=device)
                        for i in range(0, num_triplets, args.num_triplets_per_iter):
                            end = min(i + args.num_triplets_per_iter, num_triplets)
                            all_loss[i:end] = loss_fn(feats[a[i:end]], feats[p[i:end]], feats[n[i:end]])
                        
                        active_losses = all_loss[all_loss > 0]
                        log_dict.update({f"{mode}/num_active_triplets": len(active_losses), f"{mode}/num_triplets": len(a)})
                        log_dict.update({f"{mode}/triplet_mean": all_loss.mean().item(),f"{mode}/active_triplet_mean": active_losses.mean().item()})
                        loss = active_losses.mean() if len(active_losses) > 0 else torch.tensor(0.0, device=device)
                    else:
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                elif "pair" in args.loss_type:
                    pairs,labels = generate_pairs_with_labels(cat_indices, positive_index_per_query, extra_margin_positive_index_per_query, feats.device)
                    if pairs is not None:
                        a_idx, pn_idx = pairs[:, 0], pairs[:, 1]
                        all_loss = loss_fn(feats[a_idx], feats[pn_idx],labels)
                        active_losses = all_loss[all_loss > 0]
                        log_dict.update({f"{mode}/num_active_pairs": len(active_losses), f"{mode}/num_pairs": len(pairs)})
                        log_dict.update({f"{mode}/pair_mean": all_loss.mean().item(),f"{mode}/active_pair_mean": active_losses.mean().item()})
                        loss = active_losses.mean() if len(active_losses) > 0 else torch.tensor(0.0, device=device)
                    else:
                        loss = torch.tensor(0.0, device=device,requires_grad=True)
                if "allign" in args.loss_type:
                    final_loss = loss + allignmnet_loss
                else:
                    final_loss = loss
            
            if train and epoch!=0:
                if final_loss.requires_grad:
                    # scaler.scale(final_loss).backward()
                    # scaler.step(optimizer)
                    # scaler.update()
                    final_loss.backward()
                    optimizer.step()
                else:
                    print("Warning: final_loss does not require grad. Skipping update.")

            if loss != 0:
                total_loss_vpr += loss.item()
                total_loss_vpr_updated += 1
            if "allign" in args.loss_type:
                total_loss_allignment += allignmnet_loss.item()
            total_loss_allignment_updated += 1

            wandb.log(log_dict)

            


        assert torch.all(all_rgb_feats[indices] == 0), "all_rgb_feats should be zero before filling"
        assert torch.all(all_thr_feats[indices] == 0), "all_thr_feats should be zero before filling"

        all_rgb_feats[indices] = feats_rgb.cpu().detach()
        all_thr_feats[indices] = feats_thr.cpu().detach()
        for batch_idx in indices:

            if all_ground_truth[batch_idx] != []:
                print(f"Overwriting ground truth for index {batch_idx} in {mode} epoch {epoch}")
                import pdb; pdb.set_trace()  # Debugging line to inspect the ground truth

            all_ground_truth[batch_idx] = positive_index_per_query[batch_idx]
        
        import gc; gc.collect()  # Clear memory after processing each batch
        torch.cuda.empty_cache()  # Clear CUDA memory after processing each batch
        torch.cuda.ipc_collect()

        print("Cuda memory , after batch: ", torch.cuda.memory_allocated(device) / 1e6, "MB")

    
    remaining_indices = [i for i, gt in enumerate(all_ground_truth) if len(gt) == 0]
    if remaining_indices:
        with torch.no_grad():
            remaining_dataset = Subset(dataloader.dataset, remaining_indices)
            remaining_dataset.idx_to_dataset = dataloader.dataset.idx_to_dataset[remaining_indices]
            sampler = IntraDatasetBatchSampler(remaining_dataset.idx_to_dataset,batch_size=args.eval_batch_size)
            remaining_dataloader = DataLoader(remaining_dataset, num_workers=args.eval_num_workers,batch_sampler = sampler)
            for batch_item in tqdm(remaining_dataloader, desc=f"{mode.capitalize()} Remaining Epoch {epoch}"):
                batch, _ = batch_item["item"]
                indices = batch_item["batch_id"].tolist()
                rgb = batch[db_modality].to(device)
                thermal = batch[q_modality].to(device)
                feats_rgb = model_dict["rgb"].extract_feature(rgb, test=False)
                feats_thr = model_dict["thr"].extract_feature(thermal, test=False)
                all_rgb_feats[indices] = feats_rgb.cpu()
                all_thr_feats[indices] = feats_thr.cpu()
                for batch_idx in indices:
                    if all_ground_truth[batch_idx] != []:
                        print(f"Overwriting ground truth for index {batch_idx} in {mode} epoch {epoch}")
                        import pdb; pdb.set_trace()
                    all_ground_truth[batch_idx] = positive_index_per_query[batch_idx]
    
    for i, gt in enumerate(all_ground_truth):
        if len(gt) == 0:
            print(f"Warning: No ground truth for index {i} in {mode} epoch {epoch}. This might affect recall metrics.")
            import pdb; pdb.set_trace()  # Debugging line to inspect the ground truth

    all_rgb_feats = F.normalize(all_rgb_feats, dim=1)
    all_thr_feats = F.normalize(all_thr_feats, dim=1)
    recall_metrics = compute_recall_at_k(all_thr_feats, all_rgb_feats, all_ground_truth, exclude_self=True)

    # Optional retrieval visualization
    if args.debug_viz:
        sim = torch.matmul(all_thr_feats, all_rgb_feats.T)
        top_k = torch.topk(sim, k=5, dim=1).indices
        for idx in random.sample(range(len(all_thr_feats)), min(5, len(all_thr_feats))):
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
    elif arch_name == "netvlad_32":
        print(f"Using NetVLAD aggregation head for {arch_name}")
        default_agg_dict = {
            "agg_arch":'NetVLAD',
            "agg_config":{
                'num_clusters': 32,
            }
        }
        return default_agg_dict
    elif arch_name == "netvlad_128":
        print(f"Using NetVLAD aggregation head for {arch_name}")
        default_agg_dict = {
            "agg_arch":'NetVLAD',
            "agg_config":{
                'num_clusters': 128,
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
    elif arch_name == "salad_32":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 32,
                    'cluster_dim': 128,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_16":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 16,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_8":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 8,
                    'cluster_dim': 128,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_256":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 256,
                    'cluster_dim': 128,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_256_dim_64":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 256,
                    'cluster_dim': 64,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_dim_64":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 64,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_dim_32":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 32,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_dim_64_global_128":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 32,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_dim_32_global_128":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 32,
                    'token_dim': 128,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_dim_16":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 16,
                    'token_dim': 256,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_dim_16_global_128":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 16,
                    'token_dim': 128,
                }
            }
            return default_agg_dict
    elif arch_name == "salad_dim_16_global_64":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 16,
                    'token_dim': 64,
                }
            }
            return default_agg_dict
    

    elif arch_name == "salad_32_dim_64":
            print(f"Using SALAD aggregation head for {arch_name}")
            default_agg_dict = {
                "agg_arch":'SALAD',
                "agg_config":{
                    'num_channels': 768,
                    'num_clusters': 32,
                    'cluster_dim': 64,
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

    model_dict['rgb'].head[0].initialize_netvlad_layer(
        args, dataloader.dataset, model_dict['rgb'],'rgb')
    model_dict['thr'].head[0].initialize_netvlad_layer(
        args, dataloader.dataset, model_dict['thr'],'thr')

    

def main(args):
    dataset_name = "_".join(args.dataset)
    if args.eval_dataset:
        dataset_name += "_eval_" + "_".join(args.eval_dataset)
    args.save_dir = os.path.join(args.save_dir, dataset_name)
    #append date and time 
    
    wandb_name = f"{args.name}_{dataset_name}_{args.head_arch}_margin_{args.margin}_same_backbone{args.same_backbone}_frozen_backbone_{args.frozen_backbone}_un_frozen_layer_index_{'_'.join(map(str, args.un_frozen_layer_index))}"+"_".join(args.loss_type)
    
    if args.equal_samples:
        wandb_name += "_equal_samples"
    if args.aug_list:
        wandb_name += "_aug_" + "_".join(args.aug_list)
    if args.crop_images:
        wandb_name += "_crop_images"
    if args.val_positive_dist_threshold > 0:
        wandb_name += f"_val_positive_dist_{args.val_positive_dist_threshold}"
    wandb.init(project="mm_vpr", name=wandb_name)
    args.save_dir = os.path.join(args.save_dir, time.strftime("%Y-%m-%d_%H-%M-%S")+wandb_name)
    os.makedirs(args.save_dir, exist_ok=True)

    agg_dict = build_head_dict(args.head_arch)
    args.agg_dict = agg_dict

    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader = build_dataset(args)
    print("Train dataset size: ", len(train_dataloader.dataset))
    print("Val dataset size: ", len(val_dataloader.dataset))


    if args.same_backbone:
        thr_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,backbone_path = args.backbone_path,modality='thr', device=device,head_config=agg_dict,backbone_model_type="dinov2_vitb14")
        rgb_model = thr_model
        model_dict = {"rgb": rgb_model, "thr": thr_model}

        if args.initialise_netvlad and args.head_arch == "netvlad":
            initialise_netvlad_head(model_dict,train_dataloader,device)
        trainable_params = thr_model.trainable_params()
    else:
        rgb_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,modality='rgb', device=device,backbone_model_type="dinov2_vitb14",head_config=agg_dict)
        thr_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,backbone_path = args.backbone_path,modality='thr', device=device,head_config=agg_dict)
        model_dict = {"rgb": rgb_model, "thr": thr_model}
        if args.initialise_netvlad and args.head_arch == "netvlad":
            initialise_netvlad_head(model_dict,train_dataloader,device)
        trainable_params = chain(thr_model.trainable_params(), rgb_model.trainable_params())
        
    optimizer = Adam(
        trainable_params,
        lr=0.001, weight_decay=0.001
    )

    # Track last val metrics if you want metric-based curriculum later
    last_val_metrics = None

    for epoch in range(args.start_epoch,args.epochs+1):

        # --- curriculum margin for this epoch ---
        if args.curriculum_mode != 'none':
            current_margin = compute_curriculum_margin(
                epoch=epoch,
                mode=args.curriculum_mode,
                margin_start=args.margin_start,
                margin_end=args.margin_end,
                ramp_epochs=args.margin_ramp_epochs,
                last_val_metrics=last_val_metrics
            )
        else:
            current_margin = args.margin
        wandb.log({"sched/current_margin": current_margin, "epoch": epoch})

        if epoch % args.save_interval == 0:

            save_dict = {"thermal_state_dict": thr_model.state_dict()}
            if not args.same_backbone:
                save_dict["rgb_state_dict"] = rgb_model.state_dict()

            torch.save(save_dict, os.path.join(args.save_dir, f"model_{epoch}.pth"))

        train_loss_vpr, train_loss_align, train_recall_metrics = run(
            args,model_dict, train_dataloader, optimizer, device, epoch, train=True, current_margin=current_margin
        )
        log_dict = {"epoch": epoch, "train/avg_loss_vpr": train_loss_vpr, "train/avg_loss_align": train_loss_align}
        for k, v in train_recall_metrics.items():
            log_dict.update({f"train/{k}": v})
        wandb.log(log_dict)
        
        with torch.no_grad():
            val_loss_vpr, val_loss_align, val_recall_metrics = run(
                args,model_dict, val_dataloader, optimizer, device, epoch, train=False, current_margin=current_margin
            )
            log_dict = {"epoch": epoch, "val/avg_loss_vpr": val_loss_vpr, "val/avg_loss_align": val_loss_align}
            for k, v in val_recall_metrics.items():
                log_dict.update({f"val/{k}": v})
            wandb.log(log_dict)
            last_val_metrics = val_recall_metrics  # for metric-based curriculum if enabled

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
    parser.add_argument('--backbone_path', type=str, default = "",
                    help='Path to the backbone model, if not using default backbone')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default="checkpoints/vpr")
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument("--augment", action='store_true', help="Use data augmentation for training")
    parser.add_argument('--train_num_workers', type=int, default=8)
    parser.add_argument('--eval_num_workers', type=int, default=8)

    parser.add_argument('--train', default=True,type=bool, help='Mode to build datasets and dataloaders')
    parser.add_argument('--use_odom', default=True,type=bool, help='Mode to build datasets and dataloaders')
    parser.add_argument('--teacher_modality', default='rgb', type=str, help='modality which will be frozen unless "unfreeze teacher" is true')
    parser.add_argument('--student_modality', default='thr', type=str, help='modality for which encoder has to be trained')
    parser.add_argument('--vpr_test', default=False, help='Rescale images during cropping')
    parser.add_argument('--not_same_backbone', dest='same_backbone',
                        action='store_false',
                        help='Use different backbones for modalities')
    parser.set_defaults(same_backbone=True)    
    parser.add_argument('--un_frozen_layer_index', type=int, nargs='+', default=[],
                    help='List of layer indices to unfreeze')
    parser.add_argument('--head_arch', type=str, choices=['netvlad', 'netvlad_32', 'netvlad_128',
                                                        'salad','salad_8','salad_16','salad_32','salad_dim_64','salad_32_dim_64',
                                                        'salad_dim_32','salad_dim_64_global_128','salad_dim_32_global_128',
                                                        'salad_dim_16','salad_dim_16_global_128','salad_dim_16_global_64', 'salad_256', 'salad_256_dim_64'
                                                        ],
                    default='salad', help='Aggregation head architecture')
    parser.add_argument('--debug_viz', action='store_true', help='Enable Top-K retrieval visualization')
    parser.add_argument('--intra_dataset_batch', type=bool, default=True, help='Enable Top-K retrieval visualization')
    parser.add_argument('--margin', type=float, default=0.3, help='[legacy] Fixed margin for triplet/pair loss (used if not curriculum)')
    parser.add_argument('--crop_images', action='store_true', help='Disable image cropping')
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--conv_output_dim', type=int, default=-1, help='Disable shuffling of dataset')
    parser.add_argument('--fc_output_dim', type=int, default=-1, help='Disable shuffling of dataset')
    parser.add_argument('--add_bn', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--cart_split', default='vpr',type=str, help='Task to run, currently only vpr is supported')
    parser.add_argument('--debug', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--initialise_netvlad', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--rescale_during_crop', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--sampling_weight', default='equal', type=str, help='Sampling weight for the dataset')
    parser.add_argument('--sampling_temperature', default=1., type=float, help='Sampling temperature for the dataset')
    parser.add_argument('--num_triplets_per_iter', default=10000, type=int, help='Sampling temperature for the dataset')
    parser.add_argument('--start_epoch', default=0, type=int, help='Sampling temperature for the dataset')
    parser.add_argument('--equal_samples', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--loss_type', type=str, nargs='+',choices=['triplet' ,'pair','allign',"hard_triplet"], default=['hard_triplet'], help='Loss type to use for training. Can be triplet, pair, allign or hard_triplet')
    parser.add_argument("--aug_list",type=str,nargs="+",default=[], choices=[
            "brightness", "contrast", "gamma","color_jitter",
            "clahe", "blur", "affine", "cutout", "flip"
        ],help="List of augmentations to apply to RGB and thermal images. Choose one or more.")
    parser.add_argument('--val_positive_dist_threshold', type=float, default=-1., help='Distance threshold for positive pairs during validation. If -1, use the default threshold.')
    parser.add_argument('--num_negatives_per_positive', type=int, default=10, help='Number of negatives per positive for triplet loss')

    # ------ NEW: curriculum controls ------
    parser.add_argument('--margin_start', type=float, default=0.05,
                        help='Starting margin for curriculum (triplet).')
    parser.add_argument('--margin_end', type=float, default=0.5,
                        help='Final (max) margin for curriculum (triplet).')
    parser.add_argument('--margin_ramp_epochs', type=int, default=25,
                        help='Epochs to linearly ramp margin from start to end.')
    parser.add_argument('--curriculum_mode', type=str, choices=['none','epoch','metric'], default='none',
                        help='How to adapt margin. "epoch" = linear ramp by epoch; "metric" = simple recall@1 policy.')

    args = parser.parse_args()

    args.eval_batch_size = args.batch_size if args.eval_batch_size == -1 else args.eval_batch_size

    args.frozen_backbone = True if args.un_frozen_layer_index == [] else False
    if args.un_frozen_layer_index != []:
        args.un_frozen_layer_index.append("norm")
 
    assert args.conv_output_dim<0 or args.fc_output_dim<0, "conv_output_dim and fc_output_dim cannot be both set."
    
    args.dataset = sorted(args.dataset)
    args.eval_dataset = sorted(args.eval_dataset) if args.eval_dataset else []
    main(args)