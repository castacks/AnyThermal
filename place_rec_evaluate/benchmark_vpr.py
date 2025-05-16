import os
import tyro
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Literal

from torch.utils.data import DataLoader
import faiss
import faiss.contrib.torch_utils
from custom_models.str_to_cls import get_model_from_string
from custom_models.utils import *
from custom_datasets.thermal_dataloader import Thermal_day_night_MS2
import csv
import datetime
import random
from torch.nn import functional as F
from torchvision import transforms as T
@dataclass
class BenchmarkArgs:
    model_names: List[str] = field(default_factory=lambda: ["alexnet", "resnet18", "resnet50"])
    dataset_name: Literal["thermal_day_night"] = "thermal_day_night"
    dataset_root: str = "datasets/"
    top_k_vals: List[int] = field(default_factory=lambda: [1, 5, 10])
    batch_size: int = 1
    save_qual: bool = True
    qual_k: int = 5
    output_dir: str = "qualitative_outputs"
    use_faiss_gpu: bool = True
    use_wandb: bool = True
    wandb_project: str = "PlaceRecBench"
    wandb_entity: str = ""
    wandb_group: str = "ClassificationModelBenchmark"
    seq: str = ""
    exclude_exact_query_in_db: bool = False
    combine_all_seq_only : bool = False
    combine_all_seq_also : bool = False
    db_q_mode: Literal["RGB_THERMAL","THERMAL_RGB"] = "RGB_THERMAL"
    keep_aspect_ratio_during_preprocess: bool = False

    #PARV_TODO add arguments for keep_Aspect_ratio, mode (RGB_THERMAL) or (THERMAL_RGB) - dataset,query

def extract_all_features(model, dataset, batch_size=1):
    features = []
    if model.own_recall_method == False:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for imgs, _ in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(model.device)
            with torch.no_grad():
                feats = model.extract_feature(imgs,keep_ratio=args.keep_aspect_ratio_during_preprocess)
                assert len(feats.shape) == 2, f"Feature shape is not 2D: {feats.shape}"
                assert feats.shape[0] == imgs.shape[0], f"Feature batch size does not match input batch size: {feats.shape} vs {imgs.shape}"
                features.append(feats.cpu())
        return torch.cat(features, dim=0)
    else:
        return features #return empty list if the model has its own recall method


def plot_top_k_retrievals(db_dataset, qu_dataset, pos_per_qu,top_k_indices, save_dir, qual_k=5, log_to_wandb=False):
    os.makedirs(save_dir, exist_ok=True)
    padding = 20
    true_color = (0, 255, 0)
    query_color = (125, 0, 125)
    false_color = (255,   0,   0)   # False retrievals


    selected_indices = random.sample(range(len(qu_dataset)), min(10, len(qu_dataset)))

    for i in selected_indices:
        fig = plt.figure(figsize=(5 * (1 + qual_k), 5), dpi=200)
        gs = fig.add_gridspec(1, 1 + qual_k)
        q_img = normalise_img(qu_dataset[i][0])
        q_img = to_np(q_img, np.uint8)
        q_img = q_img.transpose(1, 2, 0)  # Convert to HWC format since pad_img requires that 
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(pad_img(q_img, padding, query_color))
        ax.set_title("Query")
        ax.axis("off")

        for j in range(qual_k):
            db_idx = top_k_indices[i, j]
            db_img = normalise_img(db_dataset[db_idx][0])
            db_img = to_np(db_img, np.uint8)
            db_img = db_img.transpose(1, 2, 0)  # Convert to HWC format since pad_img requires that
            ax = fig.add_subplot(gs[0, j + 1])
            color_mask = false_color if db_idx not in pos_per_qu[i] else true_color
            ax.imshow(pad_img(db_img, padding, color_mask))
            ax.set_title(f"DB {db_idx}")
            ax.axis("off")

        fig.tight_layout()
        path = os.path.join(save_dir, f"query_{i}_topk.png")
        fig.savefig(path)
        plt.close(fig)

        if log_to_wandb:
            wandb.log({f"Qualitative/query_{i}": wandb.Image(path)})
def evaluate_retrieval_faiss(query_feats, db_feats, pos_per_query, top_k_vals, use_gpu=True,exclude_exact_query_in_db=True):
    recalls = {k: 0 for k in top_k_vals}
    d = db_feats.shape[1]

    try:
        if use_gpu:
            index = faiss.IndexFlatIP(d)
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("🔋 FAISS running on GPU")
        else:
            raise RuntimeError("GPU disabled by user")

    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"⚠️ FAISS GPU not available, falling back to CPU. Reason: {e}")
        index = faiss.IndexFlatIP(d)

    index.add(db_feats.numpy())
    _, indices = index.search(query_feats.numpy(), max(top_k_vals))

    for i, retrieved in enumerate(indices):
        gt = pos_per_query[i]
        for k in top_k_vals:
            if exclude_exact_query_in_db:
                topk_plus1 = retrieved[:k+1] # if exact query in the top k frames, then look at k+1 frames and Exclude the alligned query itself
                filtered = [idx for idx in topk_plus1 if idx != i]
                if any(idx in gt for idx in filtered[:k]):
                    recalls[k] += 1
            else:
                if any(idx in gt for idx in retrieved[:k]):
                    recalls[k] += 1

    total = len(pos_per_query)
    return {f"R@{k}": recalls[k] / total for k in top_k_vals}, indices

def vpr(db_model,qu_model,args,seq,save_dir,model_name, db_dataset,db_feats, qu_dataset,qu_feats, pos_per_query, top_k_vals, use_gpu=True):
    # import pdb; pdb.set_trace()
    if db_model.own_recall_method == False:
        recalls, top_k_indices = evaluate_retrieval_faiss(
            qu_feats, db_feats, pos_per_query, top_k_vals, use_gpu=use_gpu,exclude_exact_query_in_db=args.exclude_exact_query_in_db)
    else:
        # assumes db_model == qu_model
        recalls, top_k_indices = db_model.evaluate_retrieval(db_dataset, qu_dataset, pos_per_query, top_k_vals, use_gpu=use_gpu,exclude_exact_query_in_db=args.exclude_exact_query_in_db)
    print(f"📊 Recalls:")
    for k, v in recalls.items():
        print(f"  - R@{k}: {v:.4f}")
    
    if args.use_wandb:
        for k, v in recalls.items():
            wandb.log({f"{args.dataset_name}/{seq}/{model_name}/R@{k}": v})

    if args.save_qual:
        qual_dir = os.path.join(save_dir,seq, model_name)
        os.makedirs(qual_dir, exist_ok=True)
        print(f"Saving qualitative results to {qual_dir}")
        plot_top_k_retrievals(db_dataset, qu_dataset, pos_per_query,top_k_indices, qual_dir,
                            qual_k=args.qual_k, log_to_wandb=args.use_wandb)

    return recalls, top_k_indices


@torch.no_grad()
def run(args: BenchmarkArgs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, args.dataset_name,timestamp)
    os.makedirs(save_dir, exist_ok=True)

    #dump args in the output directory
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    csv_dir = os.path.join(save_dir, "csv_results")

    seqs = args.seq.split(" ")
    os.makedirs(csv_dir, exist_ok=True)
    master_csv_path = os.path.join(csv_dir, f"recall_results.csv")

    # Open the master CSV once for all models
    csv_file = open(master_csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    header_row = ["seq","q/db","model"]
    for k in args.top_k_vals:
        header_row.append(f"R@{k}")
    csv_writer.writerow(header_row)
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group,
            config=vars(args),
            reinit=True
        )

    recall_dict = {}

    for model_name in args.model_names:
        if model_name in["imagebind","mmdistill_dinov2_fixed","mmdistill_dinov2_variable"]:
            if args.db_q_mode == "RGB_THERMAL":
                db_model = get_model_from_string(f"{model_name}_rgb")
                qu_model = get_model_from_string(f"{model_name}_thermal")
            elif args.db_q_mode == "THERMAL_RGB":
                db_model = get_model_from_string(f"{model_name}_thermal")
                qu_model = get_model_from_string(f"{model_name}_rgb")
            else:
                raise ValueError(f"Mode {args.db_q_mode} not supported. Choose either RGB_THERMAL or THERMAL_RGB")
        else:
            db_model = get_model_from_string(model_name)
            qu_model = db_model

        print(f"🏁 Benchmarking: {model_name}")

        all_db_dataset = []
        all_qu_dataset = []
        all_db_feats = []
        all_qu_feats = []
        all_pos_per_qu = []

        for seq in seqs:
            if args.dataset_name == "thermal_day_night":
                dataset = Thermal_day_night_MS2(
                    seq=seq,
                    db_modality="rgb",
                    q_modality="thr",
                    datasets_folder=args.dataset_root
                )
            else:
                raise ValueError(f"Dataset {args.dataset_name} not supported")

            db_dataset = torch.utils.data.Subset(dataset, range(dataset.database_num))
            qu_dataset = torch.utils.data.Subset(dataset, range(dataset.database_num, len(dataset)))
            pos_per_qu = dataset.soft_positives_per_query

            for i in range(len(pos_per_qu)):
                if len(pos_per_qu[i]) == 1 and args.exclude_exact_query_in_db:
                    print(f"⚠️ Warning: Query {i} has only one positive match. This may affect recall calculations since we will ignore exact match")
            all_db_dataset.append(db_dataset)
            all_qu_dataset.append(qu_dataset)
            all_db_feats.append(extract_all_features(db_model, db_dataset, batch_size=args.batch_size))
            all_qu_feats.append(extract_all_features(qu_model, qu_dataset, batch_size=args.batch_size))
            all_pos_per_qu.append(pos_per_qu)
        
        if len(seqs) >1 and (args.combine_all_seq_also or args.combine_all_seq_only):
            
            combined_db_dataset = torch.utils.data.ConcatDataset(all_db_dataset)
            combined_qu_dataset = torch.utils.data.ConcatDataset(all_qu_dataset)
            combined_db_feats = torch.cat(all_db_feats, dim=0)
            combined_qu_feats = torch.cat(all_qu_feats, dim=0)
            combined_pos_per_qu = np.concatenate(all_pos_per_qu, axis=0)
            seq_name = "combined_seq"
            if seq_name not in recall_dict:
                recall_dict[seq_name] = {}
            recalls, top_k_indices = vpr(db_model,qu_model,
                args,seq_name,save_dir,model_name, combined_db_dataset, combined_db_feats,combined_qu_dataset,combined_qu_feats, combined_pos_per_qu, args.top_k_vals, use_gpu=args.use_faiss_gpu)

            recall_dict[seq_name][model_name] = []

            for k, v in recalls.items():
                recall_dict[seq_name][model_name].append(round(v,4))

        if not args.combine_all_seq_only:
            for seq_idx in range(len(seqs)):
                seq_name = seqs[seq_idx]
                if seq_name not in recall_dict:
                    recall_dict[seq_name] = {}
                    recall_dict[seq_name]['q/db'] = f'{len(all_qu_dataset[seq_idx])}/{len(all_db_dataset[seq_idx])}'
                
                recalls, top_k_indices = vpr(db_model,qu_model,
                    args,seq_name,save_dir,model_name,all_db_dataset[seq_idx], all_db_feats[seq_idx],all_qu_dataset[seq_idx] ,all_qu_feats[seq_idx], all_pos_per_qu[seq_idx], args.top_k_vals, use_gpu=args.use_faiss_gpu)
                recall_dict[seq_name][model_name] =[]
                for k, v in recalls.items():
                    recall_dict[seq_name][model_name].append(round(v,4))
    for k1 in recall_dict.keys():
        for k2 in recall_dict[k1].keys():
            if k2 == 'q/db':
                continue
            model_row = [k1,recall_dict[k1]['q/db'],k2]
            for v in recall_dict[k1][k2]:
                model_row.append(v)
            csv_writer.writerow(model_row)                
        
    if args.use_wandb:
        wandb.finish()
    csv_file.close()
    print(f"✅ Combined CSV results saved to: {master_csv_path}")


if __name__ == "__main__":
    args = tyro.cli(BenchmarkArgs)
    run(args)
