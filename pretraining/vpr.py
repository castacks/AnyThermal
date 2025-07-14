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

def gps_triplet_miner(feats, gps_coords, pos_sim_threshold, neg_sim_threshold, pos_dist_thresh, neg_dist_thresh,mode=None):
    feats = F.normalize(feats, dim=1)
    cosine_sim = torch.matmul(feats, feats.T)
    gps_dist = torch.cdist(gps_coords, gps_coords, p=2)

    B = feats.shape[0]
    
    hard_pos_mask = (gps_dist < pos_dist_thresh) & (cosine_sim < pos_sim_threshold)
    hard_neg_mask = (gps_dist > neg_dist_thresh) & (cosine_sim > neg_sim_threshold)
    
    if ~torch.any(hard_pos_mask):
        print("No hard positive pairs found. Min cosine similarity:", cosine_sim[gps_dist < pos_dist_thresh].min().item())
        return None
    if ~torch.any(hard_neg_mask):
        print("No hard negative pairs found. Max cosine similarity:", cosine_sim[gps_dist > neg_dist_thresh].max().item())
        return None

    a_pos, p = torch.where(hard_pos_mask)
    a_neg, n = torch.where(hard_neg_mask)

    anchor_set = set(a_pos.tolist()) & set(a_neg.tolist())
    triplets = []
    for a in anchor_set:
        pos_indices = p[a_pos == a]
        neg_indices = n[a_neg == a]
        for pos_idx in pos_indices:
            for neg_idx in neg_indices:
                if pos_idx != a and neg_idx != a:
                    triplets.append((a, pos_idx.item(), neg_idx.item()))

    if not triplets:
        return None

    a, p, n = zip(*triplets)
    return torch.tensor(a), torch.tensor(p), torch.tensor(n)


def get_miner(args, feats, labels, gps_coords=None):
    if args.miner_type == "multi_similarity":
        miner = MultiSimilarityMiner(epsilon=0.1)
        return miner(feats, labels)
    elif args.miner_type == "gps_cosine":
        assert gps_coords is not None
        return gps_triplet_miner(feats, gps_coords,
                                         pos_sim_threshold=args.pos_sim_threshold,
                                         neg_sim_threshold= args.neg_sim_threshold,
                                            pos_dist_thresh=args.pos_threshold,
                                            neg_dist_thresh=args.neg_threshold)
    else:
        raise ValueError(f"Unknown miner: {args.miner_type}")

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


def run(args,model_dict, dataloader, optimizer, device, epoch, train=True, use_memory_bank=False):
    mode = "train" if train else "val"
    
    create_label_graph = False
    if args.miner_type == "multi_similarity":
        loss_fn = MultiSimilarityLoss()
        create_label_graph = True
    elif args.miner_type == "gps_cosine":
        loss_fn = TripletMarginLoss(margin=args.margin, p=2, reduction='none')
    total_loss_vpr = 0
    total_loss_allignment = 0
    memory_feats, memory_labels = [], []
    
    all_ground_truth = [[] for _ in range(len(dataloader.dataset))]
    all_rgb_feats = torch.zeros((len(dataloader.dataset), model_dict["rgb"].model.head.final_output_dim))
    all_thr_feats = torch.zeros((len(dataloader.dataset), model_dict["thr"].model.head.final_output_dim))


    db_modality = args.teacher_modality
    q_modality = args.student_modality
    positive_index_per_query = np.array(dataloader.dataset.soft_positives, dtype=object)
    extra_margin_positive_index_per_query = np.array(dataloader.dataset.extra_margin_soft_positives, dtype=object)
    gps_database = torch.from_numpy(dataloader.dataset.db_coords)

    global_batch_with_hard_pairs = 0
    total_indices_list = []
    for batch_item in tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}"):
        batch, _ = batch_item["item"]
        indices = batch_item["batch_id"].tolist()
        total_indices_list.extend(indices)
        dataset_id = batch_item["dataset_id"].tolist()
        rgb = batch[db_modality].to(device)
        thermal = batch[q_modality].to(device)
        positive_indices_list = positive_index_per_query[indices]
        gps_coords = torch.cat([gps_database[indices],gps_database[indices]]).to(device)

        if create_label_graph:
            # Graph-based connected component labeling
            index_to_label = {}
            current_label = 0
            G = nx.Graph()
            for idx, pos_list in zip(indices, positive_indices_list):
                for p in pos_list:
                    G.add_edge(idx, p)
            for component in nx.connected_components(G):
                for i in component:
                    index_to_label[i] = current_label
                current_label += 1

            labels_rgb = torch.tensor([index_to_label[i] for i in indices], device=device)
            labels_thr = torch.tensor([index_to_label[i] for i in indices], device=device)
            labels = torch.cat([labels_rgb, labels_thr], dim=0)

        with torch.no_grad() if not train else nullcontext():
            feats_rgb = model_dict["rgb"].extract_feature(rgb, test=False)
            feats_thr = model_dict["thr"].extract_feature(thermal, test=False)
            feats = torch.cat([feats_rgb, feats_thr], dim=0)

            if use_memory_bank and memory_feats:
                feats = torch.cat([feats, memory_feats[0]], dim=0)
                if create_label_graph:
                    labels = torch.cat([labels, memory_labels[0]], dim=0)

            if args.miner_type == "multi_similarity":
                hard_pairs = get_miner(args, feats, labels)
                if len(hard_pairs[0]) == 0:
                    print(f"Warning: No hard pairs found for epoch {epoch}, batch {indices}. Skipping this batch.")
                    continue

                loss = loss_fn(feats, labels, hard_pairs)
            elif args.miner_type == "gps_cosine":
                triplets = gps_triplet_miner(feats, gps_coords,
                                            pos_sim_threshold=args.pos_sim_threshold,
                                            neg_sim_threshold= args.neg_sim_threshold,
                                                pos_dist_thresh=args.pos_threshold,
                                                neg_dist_thresh=args.neg_threshold)
                if triplets is None or len(triplets[0]) == 0:
                    print(f"No hard triplets found for batch, skipping.")
                    continue

                a, p, n = triplets
                all_loss = loss_fn(feats[a], feats[p], feats[n])
                active_losses = all_loss[all_loss > 0]
                if train:
                    wandb.log({"num_active_losses": len(active_losses), f"{mode}/num_triplets": len(a)})
                    wand.log({"triplet/mean": all_loss.mean().item(),"triplet/active_mean": active_losses.mean().item()})
                loss = active_losses.mean() if len(active_losses) > 0 else torch.tensor(0.0, device=device)
                if len(active_losses) == 0:
                    print(f"Warning: No active losses found for batch {indices}. Skipping this batch.")
                    continue



                with torch.no_grad():
                    d_ap = F.pairwise_distance(feats[a], feats[p])
                    d_an = F.pairwise_distance(feats[a], feats[n])
                    wandb.log({
                        f"{mode}/d_ap": d_ap.mean().item(),
                        f"{mode}/d_an": d_an.mean().item(),
                        f"{mode}/loss_fn_margin": loss_fn.margin
                    })
            else:
                raise ValueError(f"Unknown miner type: {args.miner_type}")

            allignmnet_loss = 1 - F.cosine_similarity(feats_rgb, feats_thr, dim=1).mean()
            final_loss = loss + allignmnet_loss
            if train:
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()

            if use_memory_bank:
                memory_feats.append(feats.detach())
                memory_feats = [torch.cat(memory_feats, dim=0)]

                if create_label_graph:
                    memory_labels.append(labels.detach())
                    memory_labels = [torch.cat(memory_labels, dim=0)]

                if memory_feats[0].shape[0] > args.memory_bank_size:
                    memory_feats[0] = memory_feats[0][-args.memory_bank_size:]
                    # # memory_labels[0] = memory_labels[0][-args.memory_bank_size:]
            total_loss_vpr += loss.item()
            total_loss_allignment += allignmnet_loss.item()


        assert torch.all(all_rgb_feats[indices] == 0), "all_rgb_feats should be zero before filling"
        assert torch.all(all_thr_feats[indices] == 0), "all_thr_feats should be zero before filling"

        all_rgb_feats[indices] = feats_rgb.cpu()
        all_thr_feats[indices] = feats_thr.cpu()
        for idx,pos_list in zip(indices,positive_indices_list):
            if all_ground_truth[idx] != []:
                print(f"Overwriting ground truth for index {idx} in {mode} epoch {epoch}")
                import pdb; pdb.set_trace()  # Debugging line to inspect the ground truth

            all_ground_truth[idx] = set(pos_list)

        # # Optional: graph visualization
        # if random.random() < 0.01:
        #     fig, ax = plt.subplots(figsize=(6, 4))
        #     nx.draw_networkx(G.subgraph(indices), with_labels=True, node_size=500, font_size=8)
        #     ax.set_title("Positive Pairs Graph")
        #     wandb.log({f"{mode}/positive_groups": wandb.Image(fig)})
        #     plt.close(fig)

        # # Optional: hard pair visualization
        # if args.debug_viz and train and random.random() < 0.3 and len(hard_pairs[0]) > 0:
        #     i, j = hard_pairs[0][0].item(), hard_pairs[1][0].item()
        #     img1 = batch['rgb'][i % len(batch['rgb'])].permute(1, 2, 0).cpu()
        #     img2 = batch['thr'][j % len(batch['thr'])].squeeze().cpu()
        #     # import pdb; pdb.set_trace()  # Debugging line to inspect the images
        #     img2 = img2[0]
        #     fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        #     axs[0].imshow(img1)
        #     axs[0].set_title('Anchor (RGB)')
        #     axs[1].imshow(img2, cmap='hot')
        #     axs[1].set_title('Positive/Negative (Thermal)')
        #     wandb.log({f"{mode}/sample_pair": wandb.Image(fig)})
        #     plt.close(fig)

        global_batch_with_hard_pairs += 1

    if len(total_indices_list) != len(set(total_indices_list)):
        print(f"Warning: Duplicate indices found in {mode} epoch {epoch}. This may indicate an issue with the dataset or dataloader.")
    if len(total_indices_list) != len(dataloader.dataset):
        print(f"Warning: Not all indices were processed in {mode} epoch {epoch}. Expected {len(dataloader.dataset)}, but got {len(total_indices_list)}.")
        import pdb; pdb.set_trace()  # Debugging line to inspect the indices
    for idx in range(len(all_ground_truth)):
        if not all_ground_truth[idx]:
            print(f"Warning: No ground truth found for index {idx} in {mode} epoch {epoch}. This may indicate an issue with the dataset or dataloader.")
            import pdb; pdb.set_trace()
    if global_batch_with_hard_pairs == 0:
        print(f"No hard pairs found in {mode} epoch {epoch}")
        return float('inf')

    db_feats = F.normalize(all_rgb_feats, dim=1)
    query_feats = F.normalize(all_thr_feats, dim=1)
    recall_metrics = compute_recall_at_k(query_feats, db_feats, all_ground_truth, exclude_self=True)
    for k, v in recall_metrics.items():
        wandb.log({f"{mode}/{k}": v})
        print(f"{mode.capitalize()} {k}: {v:.4f}")

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
    

    return total_loss_vpr / global_batch_with_hard_pairs , total_loss_allignment / global_batch_with_hard_pairs


def build_head_dict(arch_name):
    if arch_name == "netvlad":
        print(f"Using NetVLAD aggregation head for {arch_name}")
        default_agg_dict = {
            "agg_arch":'NetVLAD'
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

def main(args):
    dataset_name = "_".join(args.dataset)
    args.save_dir = os.path.join(args.save_dir, dataset_name)
    #append date and time 
    args.save_dir = os.path.join(args.save_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    wandb_name = f"{args.name}_{dataset_name}_{args.head_arch}_{args.miner_type}_same_backbone{args.same_backbone}_memory_bank_{args.memory_bank}_frozen_backbone_{args.frozen_backbone}_un_frozen_layer_index_{'_'.join(map(str, args.un_frozen_layer_index))}"
    wandb.init(project="mm_vpr", name=wandb_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader = build_dataset(args)
    print("Train dataset size: ", len(train_dataloader.dataset))
    print("Val dataset size: ", len(val_dataloader.dataset))

    agg_dict = build_head_dict(args.head_arch)

    if args.same_backbone:
        rgb_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,modality='rgb', device=device,backbone_path = args.backbone_path,head_config=agg_dict)
        thr_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,model=rgb_model.model,modality='thermal', device=device,head_config=agg_dict)
        head_params = chain(thr_model.model.head.parameters())
    else:
        rgb_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,modality='rgb', device=device,backbone_model_type="dinov2_vitb14",head_config=agg_dict)
        thr_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,backbone_path = args.backbone_path,modality='thermal', device=device,head_config=agg_dict)

        head_params = chain(thr_model.model.head.parameters(), rgb_model.model.head.parameters())
        
    
    backbone_params = chain(*[thr_model.model.backbone.blocks[i].parameters() for i in args.un_frozen_layer_index])
    optimizer = Adam(
            chain(head_params, backbone_params),
            lr=0.001, weight_decay=0.001
        )
    
    model_dict = {"rgb": rgb_model, "thr": thr_model}

    for epoch in range(0,args.epochs+1):

        if epoch % args.save_interval == 0:

            save_dict = {"thermal_vpr_head": thr_model.model.head.state_dict(),
                        "backbone_path": args.backbone_path
                        }
            if not args.same_backbone:
                save_dict["rgb_vpr_head"] = rgb_model.model.head.state_dict()

            torch.save(save_dict, os.path.join(args.save_dir, f"model_{epoch}.pth"))

        if epoch>0:
            train_loss_vpr, train_loss_align = run(args,model_dict, train_dataloader, optimizer, device, epoch, train=True, use_memory_bank=args.memory_bank)
            wandb.log({"epoch": epoch, "train/avg_loss_vpr": train_loss_vpr, "train/avg_loss_align": train_loss_align})
        
        with torch.no_grad():
            val_loss_vpr, val_loss_align = run(args,model_dict, val_dataloader, optimizer, device, epoch, train=False, use_memory_bank=args.memory_bank)
            wandb.log({"epoch": epoch, "val/avg_loss": val_loss_vpr, "val/avg_loss_align": val_loss_align})

        if epoch >0:
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f} (No training in epoch 0)")
        
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="mmdistill")
    parser.add_argument('--dataset', type=str, nargs='+',
                    help='List of datasets to use in training and eval')    
    parser.add_argument('--backbone_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default="checkpoints/vpr")
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--miner_type', type=str, choices=['multi_similarity', 'gps_cosine'], default='multi_similarity')
    parser.add_argument('--memory_bank', action='store_true')
    parser.add_argument("--augment", action='store_true', help="Use data augmentation for training")
    parser.add_argument('--train_num_workers', type=int, default=4)
    parser.add_argument('--eval_num_workers', type=int, default=4)

    parser.add_argument('--train_easy', action='store_true', help="Use easy training split for CART dataset")
    parser.add_argument('--train', default=True,type=bool, help='Mode to build datasets and dataloaders')
    parser.add_argument('--use_odom', default=True,type=bool, help='Mode to build datasets and dataloaders')
    parser.add_argument('--rescale_during_crop', default=False, help='Rescale images during cropping')
    parser.add_argument('--teacher_modality', default='rgb', type=str, help='modality which will be frozen unless "unfreeze teacher" is true')
    parser.add_argument('--student_modality', default='thr', type=str, help='modality for which encoder has to be trained')
    parser.add_argument('--vpr_test', default=False, help='Rescale images during cropping')
    parser.add_argument('--same_backbone', action='store_true', help='Rescale images during cropping')
    parser.add_argument('--frozen_backbone', default=True,type=bool, help='Rescale images during cropping')
    parser.add_argument('--un_frozen_layer_index', type=int, nargs='+', default=[],
                    help='List of layer indices to unfreeze')
    parser.add_argument('--head_arch', type=str, choices=['netvlad', 'salad'], default='netvlad')
    parser.add_argument('--memory_bank_size', type=int, default=1024, help='Max memory bank size')
    parser.add_argument('--debug_viz', action='store_true', help='Enable Top-K retrieval visualization')
    parser.add_argument('--intra_dataset_batch', type=bool, default=True, help='Enable Top-K retrieval visualization')
    parser.add_argument('--pos_sim_threshold', type=float, default=0.98, help='Positive cosine similarity threshold')
    parser.add_argument('--neg_sim_threshold', type=float, default=0.6, help='Negative cosine similarity threshold')
    parser.add_argument('--pos_threshold', type=float, default=25.0, help='Positive GPS distance threshold')
    parser.add_argument('--neg_threshold', type=float, default=30.0, help='Negative GPS distance threshold')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
    parser.add_argument('--no_crop_images', dest='crop_images', action='store_false', help='Disable image cropping')
    parser.set_defaults(crop_images=True)
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--conv_output_dim', type=int, default=-1, help='Disable shuffling of dataset')
    parser.add_argument('--fc_output_dim', type=int, default=-1, help='Disable shuffling of dataset')
    parser.add_argument('--add_bn', action='store_true', help='Disable shuffling of dataset')

    args = parser.parse_args()

    assert conv_output_dim<0 or fc_output_dim<0, "conv_output_dim and fc_output_dim cannot be both set."
    main(args)
