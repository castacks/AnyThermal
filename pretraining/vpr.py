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
from custom_datasets.ms2_dataset import MS2
from custom_datasets.cart_dataset import CART, return_cart_split
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

def run(model_dict, dataloader, optimizer, device, epoch, train=True, miner_type="multi_similarity", use_memory_bank=False):
    mode = "train" if train else "val"
    # if train:
    #     model.train()
    # else:
    #     model.eval()

    loss_fn = MultiSimilarityLoss()
    total_loss = 0
    memory_feats, memory_labels = [], []
    
    db_modality = dataloader.dataset.db_modality
    q_modality = dataloader.dataset.q_modality
    positive_index_per_query = dataloader.dataset.soft_positives_per_query
    gps_database = torch.from_numpy(dataloader.dataset.db_coords)
    if db_modality != "rgb":
        raise ValueError(f"Database modality {db_modality} is not supported. Only 'rgb' is supported.")
    if q_modality != "thr":
        raise ValueError(f"Query modality {q_modality} is not supported. Only 'thr' (thermal) is supported.")

    global_batch_with_hard_pairs = 0
    for batch,indices in tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}"):
        indices = indices.tolist()
        rgb = batch[db_modality].to(device)
        thermal = batch[q_modality].to(device)
        positive_indices_list = positive_index_per_query[indices]
        gps_coords = gps_database[indices] if hasattr(dataloader.dataset, 'db_coords') else None
        if gps_coords is not None:
            gps_coords = gps_coords.to(device)
        # import pdb; pdb.set_trace()
        feats_rgb = model_dict["rgb"].extract_feature(rgb,test=False)
        feats_thr = model_dict["thr"].extract_feature(thermal,test=False)
        feats = torch.cat([feats_rgb, feats_thr], dim=0)

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
        # import pdb; pdb.set_trace()  # Debugging line to inspect the index_to_label mapping
        labels_rgb = torch.tensor([index_to_label[i] for i in indices], device=device)
        labels_thr = torch.tensor([index_to_label[i] for i in indices], device=device)
        labels = torch.cat([labels_rgb, labels_thr], dim=0)

        if use_memory_bank and memory_feats:
            feats = torch.cat([feats] + memory_feats, dim=0)
            labels = torch.cat([labels] + memory_labels, dim=0)
        if train:
            hard_pairs = get_miner(miner_type, feats, labels, gps_coords=gps_coords)
            if len(hard_pairs[0]) == 0:
                print(f"Warning: No hard pairs found for epoch {epoch}, batch {indices}. Skipping this batch.")
                continue
            loss = loss_fn(feats, labels, hard_pairs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad(): #PARV_TODO - add torch no grad above not this low in case of val 
                hard_pairs = get_miner(miner_type, feats, labels, gps_coords=gps_coords)
                if len(hard_pairs[0]) == 0:
                    print(f"Warning: No hard pairs found for epoch {epoch}, batch {indices}. Skipping this batch.")
                    continue
                loss = loss_fn(feats, labels, hard_pairs)

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

def main(args):
    wandb.init(project="mm_vpr", name=args.name)
    teacher_modality = 'rgb'
    student_modality = 'thr'
    if args.dataset == 'ms2':
        train_dataset = MS2(mode='train')
        val_dataset = MS2(mode='val')
    else:
        print("Using CART dataset")
        if args.train_easy:
            print("Using easy training split")
            train_seq_list = return_cart_split("train_easy")
        else:
            print("Using normal training split")
            train_seq_list = return_cart_split("train")
        val_seq_list = return_cart_split("val")
        data_root = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files"
        frame_list_root = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/parv/filter/static_segments_output/frames"
        train_dataset = CART(root_frame_dir=frame_list_root,db_modality=teacher_modality,q_modality=student_modality,datasets_folder=data_root,seq=train_seq_list, augment=args.augment,vpr_train=True)
        val_dataset = CART(root_frame_dir=frame_list_root,db_modality=teacher_modality,q_modality=student_modality,datasets_folder=data_root,seq=val_seq_list, augment=False,vpr_train=True) #no augmentation for val dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    rgb_model = MMDistillVPRModel(frozen_backbone=True,frozen_head=False,backbone_path = args.backbone_path,modality='rgb', device=device)
    thr_model = MMDistillVPRModel(frozen_backbone=True,frozen_head=False,model = rgb_model.model,modality='thermal', device=device)

    optimizer = Adam(thr_model.model.head.parameters(), lr=0.001, weight_decay=0.01)
    
    model_dict = {"rgb": rgb_model, "thr": thr_model}

    for epoch in range(0,args.epochs+1):
        if epoch>0:
            train_loss = run(model_dict, train_loader, optimizer, device, epoch, train=True, miner_type=args.miner_type, use_memory_bank=args.memory_bank)
            wandb.log({"epoch": epoch, "train/avg_loss": train_loss})

        with torch.no_grad():
            val_loss = run(model_dict, val_loader, optimizer, device, epoch, train=False, miner_type=args.miner_type, use_memory_bank=args.memory_bank)
            wandb.log({"epoch": epoch, "val/avg_loss": val_loss})

        if epoch >0:
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f} (No training in epoch 0)")
            
        if epoch % args.save_interval == 0:
            torch.save({"vpr_head": thr_model.model.head.state_dict(),
                        "backbone_path": args.backbone_path
                        }
                       
                       , os.path.join(args.save_dir, f"model_{epoch}.pth"))
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="dinov2_netvlad")
    parser.add_argument('--dataset', type=str, choices=['cart', 'ms2'], default='ms2')
    parser.add_argument('--backbone_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default="checkpoints/vpr")
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--miner_type', type=str, choices=['multi_similarity', 'distance_weighted', 'gps_cosine'], default='multi_similarity')
    parser.add_argument('--memory_bank', action='store_true')
    parser.add_argument("augment", action='store_true', help="Use data augmentation for training")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_easy', action='store_true', help="Use easy training split for CART dataset")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
