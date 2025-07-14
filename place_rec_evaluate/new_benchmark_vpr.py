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
import sys 
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc") # Add the parent directory to the path
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/custom_datasets") # Add the custom models directory to the path
from torch.utils.data import DataLoader
import faiss
import faiss.contrib.torch_utils
from custom_models.str_to_cls import get_model_from_string
from custom_models.utils import *
from custom_datasets.multi_dataset_loader import *

import csv
import datetime
import random
from torch.nn import functional as F
from torchvision import transforms as T
import pacmap  # Make sure you pip install pacmap
import seaborn as sns

@dataclass
class BenchmarkArgs:
    model_names: List[str] = field(default_factory=lambda: ["alexnet", "resnet18", "resnet50"])
    dataset: List[str] = field(default_factory=lambda: ["ms2"])
    # dataset_root: str = "datasets/"
    top_k_vals: List[int] = field(default_factory=lambda: [1, 5, 10])
    batch_size: int = 1
    save_qual: bool = True
    qual_k: int = 5
    output_dir: str = "qualitative_outputs/vpr"
    use_faiss_gpu: bool = True
    use_wandb: bool = True
    wandb_project: str = "PlaceRecBench"
    wandb_entity: str = ""
    wandb_group: str = "ClassificationModelBenchmark"
    exclude_exact_query_in_db: bool = False
    combine_all_seq_only : bool = False
    combine_all_seq_also : bool = False
    db_q_mode: List[str] = field(default_factory=lambda: ["RGB_THERMAL"])
    keep_aspect_ratio_during_preprocess: bool = False
    train: bool = False
    dataset_splits: List[str] = field(default_factory=lambda: ["val"])
    teacher_modality: Literal["rgb", "thr"] = "rgb"
    student_modality: Literal["rgb", "thr"] = "thr"
    train_easy: bool = False
    use_odom: bool = True
    vpr_test: bool = True
    common_database: bool = False
    crop_images: bool = False
    dist_thresh: int = 25
    cart_split: str = "vpr"
    debug: bool = False
    viz_clusters: bool = False
    rescale_during_crop: bool = False
    same_backbone: bool = False

def extract_all_features(model, dataset, batch_size=1):
    features = []
    if model.own_recall_method == False:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4)
        for imgs, _ in tqdm(dataloader, desc="Extracting features"):
            # import pdb; pdb.set_trace()
            imgs = imgs.to(model.device)
            with torch.no_grad():
                feats = model.extract_feature(imgs,keep_ratio=args.keep_aspect_ratio_during_preprocess)
                assert len(feats.shape) == 2, f"Feature shape is not 2D: {feats.shape}"
                assert feats.shape[0] == imgs.shape[0], f"Feature batch size does not match input batch size: {feats.shape} vs {imgs.shape}"
                features.append(feats.cpu())
        return torch.cat(features, dim=0)
    else:
        return features #return empty list if the model has its own recall method


def plot_top_k_retrievals(db_dataset, qu_dataset, pos_per_qu, top_k_indices, save_dir, qual_k=5, log_to_wandb=False, num_random=5, num_failures=5):
    os.makedirs(save_dir, exist_ok=True)
    padding = 20
    true_color = (0, 255, 0)
    query_color = (125, 0, 125)
    false_color = (255, 0, 0)

    # Identify failure cases
    failure_cases = []
    for i in range(len(qu_dataset)):
        retrieved = top_k_indices[i, :qual_k].tolist()
        if all(idx not in pos_per_qu[i] for idx in retrieved):
            failure_cases.append(i)

    # Randomly sample
    all_indices = set(range(len(qu_dataset)))
    failure_cases = random.sample(failure_cases, min(num_failures, len(failure_cases)))
    remaining_indices = list(all_indices - set(failure_cases))
    random_cases = random.sample(remaining_indices, min(num_random, len(remaining_indices)))

    selected_indices = random_cases + failure_cases

    for i in selected_indices:
        fig = plt.figure(figsize=(5 * (1 + qual_k), 5), dpi=200)
        gs = fig.add_gridspec(1, 1 + qual_k)
        q_img = normalise_img(qu_dataset[i][0])
        q_img = to_np(q_img, np.uint8).transpose(1, 2, 0)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(pad_img(q_img, padding, query_color))
        ax.set_title(f"Query {i}")
        ax.axis("off")

        is_failure = all(top_k_indices[i, j] not in pos_per_qu[i] for j in range(qual_k))

        for j in range(qual_k):
            db_idx = top_k_indices[i, j]
            db_img = normalise_img(db_dataset[db_idx][0])
            db_img = to_np(db_img, np.uint8).transpose(1, 2, 0)
            color_mask = false_color if db_idx not in pos_per_qu[i] else true_color
            ax = fig.add_subplot(gs[0, j + 1])
            ax.imshow(pad_img(db_img, padding, color_mask))
            ax.set_title(f"DB {db_idx}")
            ax.axis("off")

        fig.tight_layout()
        category = "failure" if is_failure else "random"
        path = os.path.join(save_dir, f"{category}_query_{i}_topk.png")
        fig.savefig(path)
        plt.close(fig)

        if log_to_wandb:
            wandb.log({f"{category}_query_{i}": wandb.Image(path)})

def evaluate_retrieval_faiss(no_positive_matches_for_queries,query_feats, db_feats, pos_per_query, top_k_vals, use_gpu=True,exclude_exact_query_in_db=True):
    recalls = {k: 0 for k in top_k_vals}
    d = db_feats.shape[1]

    try:
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, d)
            print("🔋 FAISS running on GPU")
        else:
            raise RuntimeError("GPU disabled by user")

    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"⚠️ FAISS GPU not available, falling back to CPU. Reason: {e}")
        index = faiss.IndexFlatIP(d)

    index.add(db_feats.numpy())
    _, indices = index.search(query_feats.numpy(), max(top_k_vals)+1)

    for i, retrieved in enumerate(indices):
        if no_positive_matches_for_queries[i]:
            continue
        gt = pos_per_query[i]
        for k in top_k_vals:
            if exclude_exact_query_in_db:
                gt = [idx for idx in gt if idx != i]  # Exclude the query itself from ground truth
                # if exact query in the top k frames, then look at k+1 frames and Exclude the alligned query itself
                filtered = [idx for idx in retrieved[:k+1] if idx != i]
                filtered = filtered[:k]  # Ensure we only consider the top k after filtering
            else:
                filtered = retrieved[:k]
            if any(idx in gt for idx in filtered):
                    recalls[k] += 1

    total = np.sum(~no_positive_matches_for_queries)
    return {f"R@{k}": recalls[k] / total for k in top_k_vals}, indices

def vpr(dataset_name,db_model,qu_model,args,split,save_dir,model_name, no_positive_matches_for_queries,db_dataset,db_feats, qu_dataset,qu_feats, pos_per_query, top_k_vals, use_gpu=True):
    # import pdb; pdb.set_trace()
    if db_model.own_recall_method == False:
        recalls, top_k_indices = evaluate_retrieval_faiss(no_positive_matches_for_queries,
            qu_feats, db_feats, pos_per_query, top_k_vals, use_gpu=use_gpu,exclude_exact_query_in_db=args.exclude_exact_query_in_db)
    else:
        recalls, top_k_indices = db_model.evaluate_retrieval(qu_model,no_positive_matches_for_queries,db_dataset, qu_dataset, pos_per_query, top_k_vals, use_gpu=use_gpu,exclude_exact_query_in_db=args.exclude_exact_query_in_db)
    print(f"📊 Recalls:")
    for k, v in recalls.items():
        print(f"  - R@{k}: {v:.4f}")
    
    if args.use_wandb:
        for k, v in recalls.items():
            wandb.log({f"{dataset_name}/{split}/{model_name}/R@{k}": v})

    if args.save_qual:
        qual_dir = os.path.join(save_dir,split, model_name)
        os.makedirs(qual_dir, exist_ok=True)
        print(f"Saving qualitative results to {qual_dir}")
        plot_top_k_retrievals(db_dataset, qu_dataset, pos_per_query,top_k_indices, qual_dir,
                            qual_k=args.qual_k, log_to_wandb=args.use_wandb)

    return recalls, top_k_indices

def plot_pacmap_db_and_query(db_feats, qu_feats,
                              db_modality, qu_modality,
                              save_dir, base_filename,
                              log_to_wandb=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    """
    Generates PaCMAP plots:
    - Combined DB + Query (color-coded by modality)
    - DB-only PaCMAP
    - Q-only PaCMAP
    Uses pure Matplotlib for full control over legends and colors.
    """
    import pacmap
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    # Colors
    db_color = "blue"
    qu_color = "red"

    # Initialize PaCMAP
    reducer = pacmap.PaCMAP(n_components=2, MN_ratio=0.5, FP_ratio=2.0,random_state=42)

    # 🟦🟥 1️⃣ Combined DB + Query PaCMAP
    all_feats = torch.cat([db_feats, qu_feats], dim=0).numpy()
    embedding = reducer.fit_transform(all_feats)

    # Split embeddings
    db_emb = embedding[:len(db_feats)]
    qu_emb = embedding[len(db_feats):]

    plt.figure(figsize=(8, 6))
    plt.scatter(db_emb[:, 0], db_emb[:, 1], color=db_color, label=f"DB ({db_modality.upper()})", alpha=0.7, s=20)
    plt.scatter(qu_emb[:, 0], qu_emb[:, 1], color=qu_color, label=f"Q ({qu_modality.upper()})", alpha=0.7, s=20)
    plt.title(f"PaCMAP: {db_modality.upper()} (DB) vs {qu_modality.upper()} (Query)")
    plt.legend()
    plt.tight_layout()

    combined_path = os.path.join(save_dir, f"{base_filename}_combined_pacmap.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"📈 Saved combined DB vs Q PaCMAP: {combined_path}")

    reducer = pacmap.PaCMAP(n_components=2, MN_ratio=0.5, FP_ratio=2.0,random_state=42)

    # 🟦 2️⃣ DB-only PaCMAP
    db_embedding = reducer.fit_transform(db_feats.numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(db_embedding[:, 0], db_embedding[:, 1], color=db_color, alpha=0.7, s=20)
    plt.title(f"PaCMAP: {db_modality.upper()} (DB)")
    plt.tight_layout()

    db_path = os.path.join(save_dir, f"{base_filename}_db_pacmap.png")
    plt.savefig(db_path)
    plt.close()
    print(f"📘 Saved DB-only PaCMAP: {db_path}")
    
    reducer = pacmap.PaCMAP(n_components=2, MN_ratio=0.5, FP_ratio=2.0,random_state=42)


    # # 🟥 3️⃣ Query-only PaCMAP
    qu_embedding = reducer.fit_transform(qu_feats.numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(qu_embedding[:, 0], qu_embedding[:, 1], color=qu_color, alpha=0.7, s=20)
    plt.title(f"PaCMAP: {qu_modality.upper()} (Query)")
    plt.tight_layout()

    qu_path = os.path.join(save_dir, f"{base_filename}_query_pacmap.png")
    plt.savefig(qu_path)
    plt.close()
    print(f"📕 Saved Query-only PaCMAP: {qu_path}")

@torch.no_grad()
def run(args: BenchmarkArgs):
    dataset_name = "_".join(args.dataset)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.same_backbone:
        timestamp += "_same_backbone"
    save_dir = os.path.join(args.output_dir, dataset_name,timestamp)
    os.makedirs(save_dir, exist_ok=True)

    #dump args in the output directory
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    csv_dir = os.path.join(save_dir, "csv_results")

    dataset_splits = args.dataset_splits
    os.makedirs(csv_dir, exist_ok=True)
    
    
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group,
            config=vars(args),
            reinit=True
        )

    recall_dict = {}

    for db_q_mode in args.db_q_mode:
        file_name = f"recall_results_{db_q_mode}"
        if args.same_backbone:
            file_name += "_same_backbone"
        master_csv_path = os.path.join(csv_dir, f"{file_name}.csv")
        # Open the master CSV once for all models
        csv_file = open(master_csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)

        header_row = ["dataset","dataset_splits","q/db","model"]
        for k in args.top_k_vals:
            header_row.append(f"R@{k}")
        csv_writer.writerow(header_row)

        for split in dataset_splits:
            if db_q_mode == "RGB_THERMAL":
                db_modality = "rgb"
                q_modality = "thr"
            elif db_q_mode == "THERMAL_RGB":
                db_modality = "thr"
                q_modality = "rgb"
            elif db_q_mode == "THERMAL_THERMAL":
                db_modality = "thr"
                q_modality = "thr"

            args.dataset_split_for_eval = split
            

            # import pdb; pdb.set_trace()
            dataset = build_dataset(args,return_dataloader=False)

            db_dataset = dataset.db_dataset
            qu_dataset = dataset.qu_dataset
            pos_per_qu = dataset.soft_positives if not args.common_database else dataset.common_soft_positives

            no_positive_matches_for_queries = np.zeros(len(pos_per_qu), dtype=bool)

            for i in range(len(pos_per_qu)):
                if len(pos_per_qu[i]) == 1 and args.exclude_exact_query_in_db:
                    print(f"⚠️ Warning: Query {i} has only one positive match. This may affect recall calculations since we will ignore exact match")
                    no_positive_matches_for_queries[i] = True

            for model_name in args.model_names:
                rgb_t_methods = ["mmdistill","imagebind"]
                method_is_rgbt_method_flag = False 
                for method in rgb_t_methods:
                    if method not in model_name:
                        continue
                    method_is_rgbt_method_flag = True
                    if db_q_mode == "RGB_THERMAL":
                        db_model = get_model_from_string(args,f"{model_name}_rgb","vpr")
                        qu_model = get_model_from_string(args,f"{model_name}_thr","vpr")
                    elif db_q_mode == "THERMAL_RGB":
                        db_model = get_model_from_string(args,f"{model_name}_thr","vpr")
                        qu_model = get_model_from_string(args,f"{model_name}_rgb","vpr")
                    elif db_q_mode == "THERMAL_THERMAL":
                        db_model = get_model_from_string(args,f"{model_name}_thr","vpr")
                        qu_model = get_model_from_string(args,f"{model_name}_thr","vpr")
                    else:
                        raise ValueError(f"Mode {db_q_mode} not supported. Choose either RGB_THERMAL or THERMAL_RGB")
                    break
                if not method_is_rgbt_method_flag:
                    print(f"initializing model {model_name}")
                    db_model = get_model_from_string(args,model_name,"vpr")
                    qu_model = db_model

                print(f"🏁 Benchmarking: {model_name}")


                assert db_model is not None and qu_model is not None, f"Models for {model_name} not found. Check your model names and dataset split."

                if hasattr(qu_model,"combined_feature_extraction") and qu_model.combined_feature_extraction:
                    db_feats , qu_feats = qu_model.extract_all_features(args,db_model, db_dataset, qu_dataset ,batch_size=args.batch_size)
                else:
                    db_feats = extract_all_features(db_model, db_dataset, batch_size=args.batch_size)
                    qu_feats = extract_all_features(qu_model, qu_dataset, batch_size=args.batch_size)
                

                if args.viz_clusters:
                    plot_pacmap_db_and_query(db_feats, qu_feats,
                            db_modality=db_modality,
                            qu_modality=q_modality,
                            save_dir=os.path.join(save_dir, "pacmap"),
                            base_filename=f"{model_name}_{split}",
                            log_to_wandb=args.use_wandb)

        
                if split not in recall_dict:
                    recall_dict[split] = {}
                    recall_dict[split]['q/db'] = f'{len(qu_dataset)}/{len(db_dataset)}'
                
                recalls, top_k_indices = vpr(dataset_name=dataset_name,db_model =db_model,qu_model =qu_model,
                                                args=args,split=split,save_dir=save_dir,model_name=model_name,
                                                no_positive_matches_for_queries=no_positive_matches_for_queries,
                                                db_dataset=db_dataset, db_feats=db_feats,
                                                qu_dataset=qu_dataset ,qu_feats=qu_feats, 
                                                pos_per_query=pos_per_qu, top_k_vals=args.top_k_vals, 
                                                use_gpu=args.use_faiss_gpu)
                
                recall_dict[split][model_name] =[]
                for k, v in recalls.items():
                    recall_dict[split][model_name].append(round(v,4))
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
    if args.same_backbone:
        run(args)
        args.same_backbone = False
    assert args.same_backbone == False, "Same backbone is not supported for this benchmark. Please set same_backbone to False in the config."
    run(args)
