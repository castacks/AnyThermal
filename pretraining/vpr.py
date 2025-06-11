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


def get_miner(miner_name, feats, labels, gps_coords=None):
    if miner_name == "multi_similarity":
        miner = MultiSimilarityMiner(epsilon=0.1)
        return miner(feats, labels)
    elif miner_name == "distance_weighted":
        miner = DistanceWeightedMiner()
        return miner(feats, labels)
    elif miner_name == "gps_cosine":
        assert gps_coords is not None
        # Hard negatives = far apart in GPS, but similar embeddings
        cosine_sim = torch.nn.functional.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=-1)
        gps_dist = torch.cdist(gps_coords, gps_coords, p=2)
        hard_pairs = (gps_dist > 20.0) & (cosine_sim > 0.7)
        idx_anchor, idx_pos_neg = torch.where(hard_pairs)
        return [idx_anchor, idx_pos_neg]
    else:
        raise ValueError(f"Unknown miner: {miner_name}")

def run(args,model_dict, dataloader, optimizer, device, epoch, train=True, miner_type="multi_similarity", use_memory_bank=False):
    mode = "train" if train else "val"
    # if train:
    #     model.train()
    # else:
    #     model.eval()

    loss_fn = MultiSimilarityLoss()
    total_loss = 0
    memory_feats, memory_labels = [], []
    
    db_modality = args.teacher_modality
    q_modality = args.student_modality
    positive_index_per_query = dataloader.dataset.soft_positives
    positive_index_per_query = np.array(positive_index_per_query, dtype=object)

    gps_database = torch.from_numpy(dataloader.dataset.db_coords)
    if db_modality != "rgb":
        raise ValueError(f"Database modality {db_modality} is not supported. Only 'rgb' is supported.")
    if q_modality != "thr":
        raise ValueError(f"Query modality {q_modality} is not supported. Only 'thr' (thermal) is supported.")

    global_batch_with_hard_pairs = 0
    for batch_item in tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}"):
        batch,_ = batch_item["item"]
        indices = batch_item["batch_id"].tolist()
        rgb = batch[db_modality].to(device)
        thermal = batch[q_modality].to(device)
        positive_indices_list = positive_index_per_query[indices]
        gps_coords = gps_database[indices]
        if gps_coords is not None:
            gps_coords = gps_coords.to(device)
        else:
            raise ValueError("GPS coordinates are not available for the dataset. Please provide GPS coordinates.")
        # import pdb; pdb.set_trace()

        index_to_label = {}
        current_label = 0
        G = nx.Graph()
        for idx, pos_list in zip(indices, positive_indices_list):
            for p in pos_list:
                # import pdb; pdb.set_trace()  # Debugging line to inspect the positive pairs
                G.add_edge(idx, p)
        for component in nx.connected_components(G):
            for i in component:
                index_to_label[i] = current_label
            current_label += 1

        labels_rgb = torch.tensor([index_to_label[i] for i in indices], device=device)
        labels_thr = torch.tensor([index_to_label[i] for i in indices], device=device)
        labels = torch.cat([labels_rgb, labels_thr], dim=0)

        with torch.no_grad() if not train else nullcontext():
            feats_rgb = model_dict["rgb"].extract_feature(rgb,test=False)
            feats_thr = model_dict["thr"].extract_feature(thermal,test=False)
            feats = torch.cat([feats_rgb, feats_thr], dim=0)

            if use_memory_bank and memory_feats:
                feats = torch.cat([feats] + memory_feats, dim=0)
                labels = torch.cat([labels] + memory_labels, dim=0)
            
            hard_pairs = get_miner(miner_type, feats, labels, gps_coords=gps_coords)
            if len(hard_pairs[0]) == 0:
                print(f"Warning: No hard pairs found for epoch {epoch}, batch {indices}. Skipping this batch.")
                continue
            
            loss = loss_fn(feats, labels, hard_pairs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                

            if use_memory_bank:
                memory_feats = [feats.detach()]
                memory_labels = [labels.detach()]

            total_loss += loss.item()
            # wandb.log({f"{mode}/loss": loss.item()})

        if random.random() < 0.01:
            fig, ax = plt.subplots(figsize=(6, 4))
            G_sub = G.subgraph(indices)
            nx.draw_networkx(G_sub, with_labels=True, node_size=500, font_size=8)
            ax.set_title("Positive Pairs Graph")
            wandb.log({f"{mode}/positive_groups": wandb.Image(fig)})
            plt.close(fig)

        if train and random.random() < 0.3 and len(hard_pairs[0]) > 0:
        
            i, j = hard_pairs[0][0].item(), hard_pairs[1][0].item()
            img1 = batch['rgb'][i % len(batch['rgb'])].permute(1, 2, 0).cpu()
            img2 = batch['thr'][j % len(batch['thr'])].squeeze().cpu()
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            axs[0].imshow(img1)
            axs[0].set_title('Anchor (RGB)')
            axs[1].imshow(img2[0], cmap='hot')
            axs[1].set_title('Positive/Negative (Thermal)')
            wandb.log({f"{mode}/sample_pair": wandb.Image(fig)})
            plt.close(fig)
                
        global_batch_with_hard_pairs += 1

    return total_loss / global_batch_with_hard_pairs


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
    wandb_name = f"{args.name}_{dataset_name}_{args.head_arch}_same_backbone{args.same_backbone}"
    wandb.init(project="mm_vpr", name=wandb_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader = build_dataset(args)
    print("Train dataset size: ", len(train_dataloader.dataset))
    print("Val dataset size: ", len(val_dataloader.dataset))

    agg_dict = build_head_dict(args.head_arch)

    if args.same_backbone:
        rgb_model = MMDistillVPRModel(frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,modality='rgb', device=device,backbone_path = args.backbone_path,head_config=agg_dict)
        thr_model = MMDistillVPRModel(frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,model=rgb_model.model,modality='thermal', device=device,head_config=agg_dict)
        head_params = chain(thr_model.model.head.parameters())
    else:
        rgb_model = MMDistillVPRModel(frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,modality='rgb', device=device,backbone_model_type="dinov2_vitb14",head_config=agg_dict)
        thr_model = MMDistillVPRModel(frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,backbone_path = args.backbone_path,modality='thermal', device=device,head_config=agg_dict)

        head_params = chain(thr_model.model.head.parameters(), rgb_model.model.head.parameters())
        
    
    backbone_params = chain(*[thr_model.model.backbone.blocks[i].parameters() for i in args.un_frozen_layer_index])
    # import pdb; pdb.set_trace()  # Debugging line to inspect the model parameters
    optimizer = Adam(
            chain(head_params, backbone_params),
            lr=0.001, weight_decay=0.01
        )
    
    model_dict = {"rgb": rgb_model, "thr": thr_model}

    for epoch in range(0,args.epochs+1):
        if epoch>0:
            train_loss = run(args,model_dict, train_dataloader, optimizer, device, epoch, train=True, miner_type=args.miner_type, use_memory_bank=args.memory_bank)
            wandb.log({"epoch": epoch, "train/avg_loss": train_loss})

        with torch.no_grad():
            val_loss = run(args,model_dict, val_dataloader, optimizer, device, epoch, train=False, miner_type=args.miner_type, use_memory_bank=args.memory_bank)
            wandb.log({"epoch": epoch, "val/avg_loss": val_loss})

        if epoch >0:
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f} (No training in epoch 0)")
            
        if epoch % args.save_interval == 0:

            save_dict = {"thermal_vpr_head": thr_model.model.head.state_dict(),
                        "backbone_path": args.backbone_path
                        }
            if not args.same_backbone:
                save_dict["rgb_vpr_head"] = rgb_model.model.head.state_dict()

            torch.save(save_dict, os.path.join(args.save_dir, f"model_{epoch}.pth"))
        
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
    parser.add_argument('--miner_type', type=str, choices=['multi_similarity', 'distance_weighted', 'gps_cosine'], default='multi_similarity')
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


    args = parser.parse_args()
    main(args)
