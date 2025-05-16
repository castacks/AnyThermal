# Doing VLAD with Dino V2 descriptors
"""
    Basic idea is to explore the layers and facets of Dino-v2 and do
    VLAD over them.
"""

import os
import sys

from dataclasses import dataclass, field
import einops as ein
import faiss
import faiss.contrib.torch_utils
import matplotlib.pyplot as plt
import joblib
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader

import timm
import time
import traceback
from tqdm.auto import tqdm
import tyro
from typing import Union, Literal, Tuple, List

import wandb

from utilities import * 
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from custom_datasets.thermal_dataloader import Thermal_day_night_MS2
from custom_datasets.cart_dataloader import CartDataloader
from custom_datasets.tartanair_dataset import TartanAirDataset
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-v2-Descs", 
        wandb_group="VLAD-Descs")
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # Database modality
    db_modality: Literal["rgb", "thr", "lidar"] = "rgb"
    # Query modality
    q_modality: Literal["rgb", "thr", "lidar","depth"] = "thr"

    db_model: str = ""
    q_model: str = ""
    # Dino parameters
    # Model type
    model_type: Literal["dinov2_vits14", "dinov2_vitb14", 
            "dinov2_vitl14", "dinov2_vitg14","dinov2_vits14_reg","dino_vits16"] = "dinov2_vits14"
    """
        Model for Dino-v2 to use as the base model.
    """
    #PARV_TODO: Instead of this it is being done in the dataset which is wrong. use this instead
    # Sub-sample query images (RAM or VRAM constraints) (1 = off)
    sub_sample_qu: int = 1
    # Sub-sample database images (RAM or VRAM constraints) (1 = off)
    sub_sample_db: int = 1
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Show a matplotlib plot for recalls
    show_plot: bool = False

    #retreival stuff
    batch_size: int = 1
    qual_num_rets: int = 5
    faiss_method: Literal["l2", "cosine"] = "cosine"
    use_residual: Literal[1, 0] = bool(0)
    qual_result_percent: float = 0.05

def plot_recalls(largs: LocalArgs, ndb_descs: np.ndarray, 
            nqu_descs: np.ndarray, pos_per_qu: np.ndarray,
            vpr_dl:Union[None, DataLoader]=None, use_percentage=True, 
            use_gpu: bool=False, save_figs:bool= True):
    """
        Calculate the recalls through similarity search (using cosine
        distances).
        
        Parameters:
        - largs:    Local arguments to program. The following are used
                    - top_k_vals: For getting keys
        - ndb_descs:    Normalized database descriptors [N_d, D]
        - nqu_descs:    Normalized query descriptors [N_q, D]
        - pos_per_qu:   Positives (within a distance threshold) per
                        query index. [N_qu, ] list (object) with each
                        index containing positive sample indices.
        - vpr_dl:       DataLoader for images (used for getting 
                        qualitative results). Pass None if certain of
                        no qualitative results (see `save_figs`).
        - use_percentage:   If true, the recall is between [0, 1] and
                            not absolute. It's divided by N_q.
        - use_gpu:      Use GPU for faiss (else use CPU)
        - save_figs:    Save the qualitative results (if False, no
                        qualitative results are saved, and if True,
                        then saving depends on LocalArgs `exp_id`)
        
        Returns:
        - recalls: A dictionary of retrievals
    """
    # Saving preferences
    query_color = (125,   0, 125)   # RGB for query image (1st)
    false_color = (255,   0,   0)   # False retrievals
    true_color =  (  0, 255,   0)   # True retrievals
    padding = 20
    qimgs_result, qimgs_dir = True, \
        f"{largs.prog.cache_dir}/qualitative_retr" # Directory

    #PARV_TODO: add date and time in the save_dir
    if largs.exp_id == False or largs.exp_id is None:   # Don't store
        qimgs_result, qimgs_dir = False, None
    elif type(largs.exp_id) == str:
        if not largs.use_residual:
            qimgs_dir = f"{largs.prog.cache_dir}/experiments/"\
                        f"{largs.exp_id}/qualitative_retr"
        else:
            qimgs_dir = f"{largs.prog.cache_dir}/experiments/"\
                        f"{largs.exp_id}/qualitative_retr_residual_nc"\
                        f"{largs.num_clusters}"
    qimgs_inds = []
    # print(save_figs, largs.qual_result_percent,qimgs_result)
    # import pdb;pdb.set_trace()
    if (not save_figs) or largs.qual_result_percent <= 0:
        qimgs_result = False
    if not qimgs_result:    # Saving query images
        print("Not saving qualitative results")
    else:
        _n_qu = nqu_descs.shape[0]
        qimgs_inds = np.random.default_rng().choice(
                range(_n_qu), int(_n_qu * largs.qual_result_percent),
                replace=False)  # Qualitative images to save
        # qimgs_inds = [32,35,38,39,40,79]
        print(f"There are {_n_qu} query images")
        print(f"Will save {len(qimgs_inds)} qualitative images")
        if not os.path.isdir(qimgs_dir):
            os.makedirs(qimgs_dir)  # Ensure folder exists
            print(f"Created qualitative directory: {qimgs_dir}")
        else:
            print(f"Saving qualitative results in: {qimgs_dir}")
    # FAISS search
    max_k = max(largs.top_k_vals)
    D = ndb_descs.shape[1]
    recalls = dict(zip(largs.top_k_vals, [0]*len(largs.top_k_vals)))
    if largs.faiss_method == "cosine":
        index = faiss.IndexFlatIP(D)
    elif largs.faiss_method == "l2":
        index = faiss.IndexFlatL2(D)
    else:
        raise Exception(f"FAISS method: {largs.faiss_method}!")
    if use_gpu:
        print("Running GPU faiss index")
        res = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(ndb_descs)    # Add database
    # Distances and indices are [N_q, max_k] shape
    distances, indices = index.search(nqu_descs, max_k) # Query
    for i_qu, qu_retr_maxk in enumerate(indices):
        for i_rec in largs.top_k_vals:
            correct_retr_qu = pos_per_qu[i_qu]  # Ground truth
            if np.any(np.isin(qu_retr_maxk[:i_rec], correct_retr_qu)):
                recalls[i_rec] += 1 # Query retrieved correctly
        if i_qu in qimgs_inds and qimgs_result:
            # Save qualitative results
            qual_top_k = qu_retr_maxk[:largs.qual_num_rets]
            correct_retr_qu = pos_per_qu[i_qu]
            color_mask = np.isin(qual_top_k, correct_retr_qu)
            # print(qual_top_k,correct_retr_qu)
            colors_all = [true_color if x else false_color \
                        for x in color_mask]
            retr_dists = distances[i_qu, :largs.qual_num_rets]
            img_q = to_pil_list(    # Dataset is [database] + [query]
                vpr_dl.dataset[ndb_descs.shape[0]+i_qu][0])[0]
            img_q = to_np(img_q, np.uint8)

            subsample = vpr_dl.dataset.subsample
            # Main figure
            fig = plt.figure(figsize=(5*(1+largs.qual_num_rets), 5),
                            dpi=300)
            gs = fig.add_gridspec(1, 1+largs.qual_num_rets)
            ax = fig.add_subplot(gs[0, 0])
            print(f"Query image: {i_qu*subsample} + {ndb_descs.shape[0]}")
            ax.set_title(f"{i_qu*subsample} + {ndb_descs.shape[0]}")  # DS index
            ax.imshow(pad_img(img_q, padding, query_color))
            ax.axis('off')
            for i, db_retr in enumerate(qual_top_k):
                ax = fig.add_subplot(gs[0, i+1])
                img_r = to_pil_list(vpr_dl.dataset[db_retr][0])[0]
                img_r = to_np(img_r, np.uint8)
                ax.set_title(f"{db_retr} ({retr_dists[i]:.4f})")
                ax.imshow(pad_img(img_r, padding, colors_all[i]))
                ax.axis('off')
            fig.set_tight_layout(True)
            save_path = f"{qimgs_dir}/Q_{i_qu}_Top_"\
                        f"{largs.qual_num_rets}.png"
            fig.savefig(save_path)
            plt.close(fig)
            if largs.prog.use_wandb and largs.prog.wandb_save_qual:
                wandb.log({"Qual_Results": wandb.Image(save_path)})
    if use_percentage:
        for k in recalls:
            recalls[k] /= len(indices)  # As a percentage of queries
    return recalls

def build_descriptors(largs: LocalArgs, vpr_ds, verbose: bool=True,db_modality: str="rgb",q_modality: str="thr")-> Tuple[torch.Tensor, torch.Tensor]:

    
    dino_db = torch.hub.load('facebookresearch/dinov2', largs.model_type).cuda()

    if db_modality == "thr" or db_modality =='lidar' or db_modality == 'depth':
        if db_modality == "thr":
            print("Loading thermal weights for database model")
        elif db_modality == "lidar":
            print("Loading lidar weights for database model")
        elif db_modality == "depth":    
            print("Loading depth weights for database model")
        dino_db.load_state_dict(torch.load(largs.db_model)["student_model_state_dict"])
    else:
        print("Loading rgb weights for database model")

    dino_q = torch.hub.load('facebookresearch/dinov2', largs.model_type).cuda()

    if q_modality == "thr" or q_modality =='lidar' or q_modality == 'depth':
        if q_modality == "thr":
            print("Loading thermal weights for query model")
        elif q_modality == "lidar":
            print("Loading lidar weights for query model")
        elif q_modality == "depth":    
            print("Loading depth weights for query model")
        dino_q.load_state_dict(torch.load(largs.q_model)["student_model_state_dict"])
    else:
        print("Loading rgb weights for query model")

    def extract_patch_descriptors_db(indices,allow_flip=False):
        print("allow flipping",allow_flip)
        patch_descs = []
        for i in tqdm(indices, disable=not verbose):
            img = vpr_ds[i][0].to(device)
            c, h, w = img.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14 # local patch of 14*14
            img_in = T.CenterCrop((h_new, w_new))(img)[None, ...]
            if allow_flip:
                flipper = T.RandomHorizontalFlip(p=1)
                img_in = flipper(img_in)
            ret = dino_db(img_in)
            ret = F.normalize(ret,dim=-1)
            patch_descs.append(ret.cpu())
        patch_descs = torch.cat(patch_descs, dim=0) # [N, n_p, d_dim] #PARV_Q what is n_p

        return patch_descs

    def extract_patch_descriptors_q(indices,allow_flip=False):
        print("allow flipping",allow_flip)
        patch_descs = []
        for i in tqdm(indices, disable=not verbose):
            img = vpr_ds[i][0].to(device)
            c, h, w = img.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_in = T.CenterCrop((h_new, w_new))(img)[None, ...]
            if allow_flip:
                flipper = T.RandomHorizontalFlip(p=1)
                img_in = flipper(img_in)
            ret = dino_q(img_in)
            ret = F.normalize(ret,dim=-1)
            patch_descs.append(ret.cpu())
        patch_descs = torch.cat(patch_descs, dim=0) # [N, n_p, d_dim]

        return patch_descs

    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    
    # Get descriptors of the database
    if verbose:
        print("Building VLADs for databases...") #PARV_Q where is the VLAD exactly ?
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    db_img_names = vpr_ds.get_image_relpaths(db_indices)

    full_db = extract_patch_descriptors_db(db_indices,allow_flip=False)
    if verbose:
        print(f"Full database descriptor shape: {full_db.shape}")
    
    # Get descriptors of the queries
    if verbose:
        print("Building VLADs for queries...")
    qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    qu_img_names = vpr_ds.get_image_relpaths(qu_indices)

    full_qu = []
    full_qu = extract_patch_descriptors_q(qu_indices)
    if verbose:
        print(f"Full query descriptor shape: {full_qu.shape}")
    
    return full_db, full_qu

@torch.no_grad()
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything(42)

    if largs.prog.use_wandb:
        # Launch WandB
        print(largs.prog.wandb_proj, largs.prog.wandb_entity, largs.prog.wandb_group, largs.prog.wandb_run_name)
        if largs.prog.wandb_entity == "":
            wandb_run = wandb.init(project=largs.prog.wandb_proj, 
                group=largs.prog.wandb_group, 
                name=largs.prog.wandb_run_name)
        else:
            wandb_run = wandb.init(project=largs.prog.wandb_proj, 
                    entity=largs.prog.wandb_entity, # config=largs,
                    group=largs.prog.wandb_group, 
                    name=largs.prog.wandb_run_name)
        print(f"Initialized WandB run: {wandb_run.name}")
    
    print("--------- Generating VLADs ---------")
    ds_dir = largs.prog.data_dir
    ds_name = largs.prog.dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}")

    # Load dataset
    if ds_name=="thermal_day_night":
        vpr_ds = Thermal_day_night_MS2(seq=largs.prog.ms2_seq,db_modality=largs.db_modality,q_modality=largs.q_modality,datasets_folder=ds_dir)
    elif ds_name=="cart":
        vpr_ds = CartDataloader(largs.bd_args,seq=largs.prog.ms2_seq,db_modality=largs.db_modality,q_modality=largs.q_modality)
    elif ds_name=="tartanair":
        vpr_ds = TartanAirDataset(seq=largs.prog.ms2_seq,db_modality=largs.db_modality,q_modality=largs.q_modality,datasets_folder=ds_dir)
    else:
        raise Exception(f"Dataset {ds_name} not supported")
    db_vlads, qu_vlads = build_descriptors(largs, vpr_ds, verbose=True,db_modality=largs.db_modality,q_modality=largs.q_modality)
    print("--------- Generated VLADs ---------")
    
    print("----- Calculating recalls through top-k matching -----")
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
        db_vlads, qu_vlads, vpr_ds.soft_positives_per_query, 
        sub_sample_db=largs.sub_sample_db, 
        sub_sample_qu=largs.sub_sample_qu)
    print("------------ Recalls calculated ------------")


    print("--------------------- Results ---------------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Model-Type": str(largs.model_type),
        "DB-Modality": str(largs.db_modality),
        "Q-Modality": str(largs.q_modality),
        "Experiment-ID": str(largs.exp_id),
        "DB-Name": str(ds_name),
        "Num-DB": str(len(db_vlads)),
        "Num-QU": str(len(qu_vlads)),
        "Agg-Method": "CLS",
        "Timestamp": str(ts)
    }
    print("Results: ")
    for k in results:
        print(f"- {k}: {results[k]}")
    print("- Recalls: ")
    for k in recalls:
        results[f"R@{k}"] = recalls[k]
        print(f"  - R@{k}: {recalls[k]:.5f}")
    if largs.show_plot:
        plt.plot(recalls.keys(), recalls.values())
        plt.ylim(0, 1)
        plt.xticks(largs.top_k_vals)
        plt.xlabel("top-k values")
        plt.ylabel(r"% recall")
        plt_title = "Recall curve"
        if largs.exp_id is not None:
            plt_title = f"{plt_title} - Exp {largs.exp_id}"
        if largs.prog.use_wandb:
            plt_title = f"{plt_title} - {wandb_run.name}"
        plt.title(plt_title)
        plt.show()

    # Log to WandB
    if largs.prog.use_wandb:
        wandb.log(results)
        for tk in recalls:
            wandb.log({"Recall-All": recalls[tk]}, step=int(tk))
    
    # Add retrievals
    results["Qual-Dists"] = dists
    results["Qual-Indices"] = indices
    save_res_file = None
    if largs.exp_id == True:
        save_res_file = caching_directory
    elif type(largs.exp_id) == str:
        save_res_file = f"{caching_directory}/experiments/"\
                        f"{largs.exp_id}"
    if save_res_file is not None:
        if not os.path.isdir(save_res_file):
            os.makedirs(save_res_file)
        save_res_file = f"{save_res_file}/results_{ts}.gz"
        print(f"Saving result in: {save_res_file}")
        joblib.dump(results, save_res_file)
    else:
        print("Not saving results")

    # Plot recalls
    print("------------ Plot Recalls ------------")
    vpr_dl = DataLoader(vpr_ds, largs.batch_size, pin_memory=True, 
                        shuffle=False)

    plot_recalls(largs, db_vlads, qu_vlads, vpr_ds.soft_positives_per_query,vpr_dl)


    if largs.prog.use_wandb:
        wandb.finish()
    print("--------------------- END ---------------------")


if __name__ == "__main__" :
    largs = tyro.cli(LocalArgs, description=__doc__)
    _start = time.time()
    try:
        main(largs)
    except:
        print("Unhandled exception")
        traceback.print_exc()
    finally:
        print(f"Program ended in {time.time()-_start:.3f} seconds")
        exit(0)

