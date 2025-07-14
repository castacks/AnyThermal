import sys
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/satellite-thermal-geo-localization")  # Add parent directory to path
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc')
from custom_models.mmdistill_dinov2_model import MMDistillVPRModel

import datasets_ws
import commons
import test
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import copy
import wandb
from uuid import uuid4
from custom_datasets.multi_dataset_loader import TripletsDataset, triplet_collate_fn,build_dataset,IntraDatasetBatchSampler

import os
import torch
import argparse

from distill import StoreWithFlag
from itertools import chain
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import shutil
from PIL import Image
import faiss


import sys
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/STHN/global_pipeline")  # Add parent directory to path

from plotting import process_results_simulation
import multiprocessing as mp
from torch.utils.data import ConcatDataset
from torch.cuda.amp import autocast

def save_checkpoint(args, epoch,state, is_best, filename, suffix=""):
    model_path = join(args.save_dir, f"epoch_{epoch:04d}")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    
    model_path = join(model_path, filename)

    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, f"best_model{suffix}.pth"))
        

def warmup_model(model, device="cuda", input_shape=(1, 3, 224, 224)):
    """
    Runs a dummy forward pass to warm up CUDA, avoiding slow first inference.
    """
    print("Running warmup model...")

    # Create a dummy input tensor
    dummy_input = torch.randn(*input_shape, device=device)

    # Run a forward pass without gradients

    with torch.no_grad(), autocast():
        _ = model.extract_feature(dummy_input,test = False)



def test(args, eval_ds, model, model_db=None, test_method="hard_resize", pca=None, visualize=False):
    """Compute features of the given dataset and compute the recalls."""

    assert test_method in [
        "hard_resize",
        "single_query",
        "central_crop",
        "five_crops",
        "nearest_crop",
        "maj_voting",
    ], f"test_method can't be {test_method}"

    model.eval()
    if model_db is not None:
        model_db.eval()
    with torch.no_grad(),autocast():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"

        database_subset_ds = eval_ds.db_dataset


        sampler = IntraDatasetBatchSampler(database_subset_ds.idx_to_dataset,args.infer_batch_size, shuffle = False)

        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_sampler=sampler
        )

        if test_method == "nearest_crop" or test_method == "maj_voting":
            all_features = np.empty(
                (5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim),
                dtype="float32",
            )
        else:
            all_features = np.empty(
                (len(eval_ds), args.features_dim), dtype="float32")
            last_all_features_idx = 0
        


        for batch_item in tqdm(database_dataloader, ncols=100):
            torch.cuda.empty_cache()  # Clear GPU memory
            if model_db is not None:
                features = model_db.extract_feature(batch_item[0].to(args.device),test=False)
            else:
                features = model.extract_feature(batch_item[0].to(args.device),test=False)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            indices = np.arange(features.shape[0]) + last_all_features_idx
            all_features[indices, :] = features
            last_all_features_idx += indices.shape[0]

        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = (
            1 if test_method == "single_query" else args.infer_batch_size
        )
        eval_ds.test_method = test_method
        queries_subset_ds = eval_ds.qu_dataset
        sampler = IntraDatasetBatchSampler(queries_subset_ds.idx_to_dataset, queries_infer_batch_size, shuffle=False)
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds,
            num_workers=args.num_workers,
            batch_sampler=sampler,
        )
        for batch_item in tqdm(queries_dataloader, ncols=100):
            features = model.extract_feature(batch_item[0].to(args.device), test=False)
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            indices = np.arange(features.shape[0]) + last_all_features_idx
            if pca != None:
                features = pca.transform(features)

            all_features[indices, :] = features
            
            last_all_features_idx += indices.shape[0]


    queries_features = all_features[len(database_subset_ds):]
    database_features = all_features[: len(database_subset_ds)]
    logging.info(f"Final feature dim: {queries_features.shape[1]}")
        
    del all_features

    logging.debug("Calculating recalls")
    if args.prior_location_threshold == -1:
        if args.use_faiss_gpu:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.GpuIndexFlatL2(res, args.features_dim)
        else:
            faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(database_features)
        distances, predictions = faiss_index.search(
            queries_features, max(args.recall_values)
        )
        del database_features
    else:
        distances, predictions = [[] for i in range(len(queries_features))], [[] for i in range(len(queries_features))]
        hard_negatives_per_query = eval_ds.get_hard_negatives()
        for query_index in tqdm(range(len(predictions))):
            faiss_index = faiss.IndexFlatL2(args.features_dim)
            faiss_index.add(database_features[hard_negatives_per_query[query_index]])
            distances_single, local_predictions_single = faiss_index.search(
                np.expand_dims(queries_features[query_index], axis=0), max(args.recall_values)
                )
            # logging.debug(f"distances_single:{distances_single}")
            # logging.debug(f"predictions_single:{predictions_single}")
            distances[query_index] = distances_single
            predictions_single = hard_negatives_per_query[query_index][local_predictions_single]
            predictions[query_index] = predictions_single
        distances = np.concatenate(distances, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        del database_features
    if test_method == "nearest_crop":
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[
            :, :20
        ]  # keep only the closer 20 predictions for each query
    elif test_method == "maj_voting":
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting("top1", predictions[q],
                         distances[q], args.majority_weight)
            top_n_voting("top5", predictions[q],
                         distances[q], args.majority_weight)
            top_n_voting("top10", predictions[q],
                         distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[
            :, 0, :20
        ]  # keep only the closer 20 predictions for each query

    # For each query, check if the predictions are correct
    positives_per_query = eval_ds.soft_positives
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val,
            rec in zip(args.recall_values, recalls)]
    )

    if args.use_best_n > 0:
        if visualize:
            if os.path.isdir("visual_loc"):
                shutil.rmtree("visual_loc")
            os.mkdir("visual_loc")
            save_dir = "visual_loc"
            # init dataset
            eval_ds.__getitem__(0)
        samples_to_be_used = args.use_best_n
        error_m = []
        position_m = []
        for query_index in tqdm(range(len(predictions))):
            distance = distances[query_index]
            prediction = predictions[query_index]
            sort_idx = np.argsort(distance)
            if args.use_best_n == 1:
                best_position = eval_ds.db_coords_per_dataset[prediction[sort_idx[0]]]
            else:
                if distance[sort_idx[0]] == 0:
                    best_position = eval_ds.db_coords_per_dataset[prediction[sort_idx[0]]]
                else:
                    mean = distance[sort_idx[0]]
                    sigma = distance[sort_idx[0]] / distance[sort_idx[-1]]
                    X = np.array(distance[sort_idx[:samples_to_be_used]]).reshape((-1,))
                    weights = np.exp(-np.square(X - mean) / (2 * sigma ** 2))  # gauss
                    weights = weights / np.sum(weights)

                    x = y = 0
                    for p, w in zip(eval_ds.db_coords_per_dataset[prediction[sort_idx[:samples_to_be_used]]], weights.tolist()):
                        y += p[0] * w
                        x += p[1] * w
                    best_position = (y, x)
            actual_position = eval_ds.q_coords_per_dataset[query_index]
            error = np.linalg.norm((actual_position[0]-best_position[0], actual_position[1]-best_position[1]))
            if error >= 50 and visualize: # Wrong results
                database_index = prediction[sort_idx[0]]
                database_img = eval_ds._find_img_in_h5(database_index, "database")
                if args.G_contrast:
                    query_img = transforms.functional.adjust_contrast(eval_ds._find_img_in_h5(query_index, "queries"), contrast_factor=3)
                else:
                    query_img = eval_ds._find_img_in_h5(query_index, "queries")
                result = Image.new(database_img.mode, (524, 524), (255, 0, 0))
                result.paste(database_img, (6, 6))
                database_img = result
                database_img.save(f"{save_dir}/{query_index}_wrong_d.png")
                query_img.save(f"{save_dir}/{query_index}_wrong_q.png")
            elif error <= 35 and visualize: # Wrong results
                database_index = prediction[sort_idx[0]]
                database_img = eval_ds._find_img_in_h5(database_index, "database")
                if args.G_contrast:
                    query_img = transforms.functional.adjust_contrast(eval_ds._find_img_in_h5(query_index, "queries"), contrast_factor=3)
                else:
                    query_img = eval_ds._find_img_in_h5(query_index, "queries")
                result = Image.new(database_img.mode, (524, 524), (0, 255, 0))
                result.paste(database_img, (6, 6))
                database_img = result
                database_img.save(f"{save_dir}/{query_index}_correct_d.png")
                query_img.save(f"{save_dir}/{query_index}_correct_q.png")
            elif visualize: # Ambiguous results
                database_index = prediction[sort_idx[0]]
                database_img = eval_ds._find_img_in_h5(database_index, "database")
                if args.G_contrast:
                    query_img = transforms.functional.adjust_contrast(eval_ds._find_img_in_h5(query_index, "queries"), contrast_factor=3)
                else:
                    query_img = eval_ds._find_img_in_h5(query_index, "queries")
                result = Image.new(database_img.mode, (524, 524), (128, 128, 128))
                result.paste(database_img, (6, 6))
                database_img = result
                database_img.save(f"{save_dir}/{query_index}_d.png")
                query_img.save(f"{save_dir}/{query_index}_q.png")
            
            error_m.append(error)
            position_m.append(actual_position)
        
        save_dir = join(args.save_dir, f"{args.epoch_num:04d}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        process_results_simulation(error_m, save_dir)
            
    return recalls, recalls_str

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

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmarking Visual Geolocalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--use_extended_data",
        action="store_true",
        help="Use extended data from pix2pix",
    )
    parser.add_argument(
        "--G_test_norm",
        type=str,
        default="batch",
        choices=["batch", "instance"],
        help="Test norm for G",
    )
    parser.add_argument(
        "--G_tanh",
        action="store_true",
        help="tanh for G",
    )
    parser.add_argument(
        "--GAN_epochs_decay",
        type=int,
        default=0,
        help="lr decay epoch num",
    )
    parser.add_argument(
        "--GAN_lr_policy",
        type=str,
        default="linear",
        choices="linear",
        help="lr scheduler.",
    )
    parser.add_argument(
        "--GAN_resize",
        type=int,
        default=[512, 512],
        nargs=2,
        help="Resizing shape for images (HxW).",
    )
    parser.add_argument(
        "--GAN_mode",
        type=str,
        default="lsgan",
        choices=["vanilla", "lsgan"],
        help="Choices of GAN loss"
    )
    parser.add_argument(
        "--GAN_upsample",
        type=str,
        default="bilinear",
        choices=["convtrans", "bilinear"],
        help="Save freq for GAN"
    )
    parser.add_argument(
        "--GAN_save_freq",
        type=int,
        default=0,
        help="Save freq for GAN"
    )
    parser.add_argument(
        "--GAN_norm",
        type=str,
        default="batch",
        choices=["batch", "instance"],
        help="Norm layer in GAN"
    )
    parser.add_argument(
        "--G_contrast",
        action="store_true",
        help="G_contrast"
    )
    parser.add_argument(
        "--G_gray",
        action="store_true",
        help="G_gray"
    )
    parser.add_argument(
        "--G_loss_lambda",
        type=float,
        default=100.0,
        help="G_loss_lambda only for pix2pix"
    )
    parser.add_argument(
        "--visual_all",
        action="store_true",
        help="visual_all"
    )
    parser.add_argument(
        "--DA_only_positive",
        action="store_true",
        help="Domain adaptation only applys to positive database"
    )
    parser.add_argument(
        "--D_net",
        type=str,
        default="none",
        choices=["none", "patchGAN", "patchGAN_deep"],
        help="D_net"
    )
    parser.add_argument(
        "--G_net",
        type=str,
        default="none",
        choices=["none", "unet", "unet_deep"],
        help="G_net"
    )
    parser.add_argument(
        "--lambda_DA",
        type=float,
        default=1.0,
        help="Domain adaptation loss weight"
    )
    parser.add_argument(
        "--DA",
        type=str,
        default='none',
        choices=['none', 'DANN_before', 'DANN_after', 'DANN_before_conv'],
        help="Domain adaptation"
    )
    parser.add_argument(
        "--add_bn",
        action="store_true",
        help="Add bn to compression layers"
    )
    parser.add_argument(
        "--remove_relu",
        action="store_true",
        help="Remove last relu layer of backbone"
    )
    parser.add_argument(
        "--use_faiss_gpu",
        action="store_true",
        help="Choose if we use faiss gpu version for mining. Only work for full and partial."
    )
    parser.add_argument(
        "--prior_location_threshold",
        type=int,
        default=-1,
        help="The threshold of search region from prior knowledge for train and test. If -1, then no prior knowledge"
    )
    parser.add_argument(
        "--use_best_n",
        type=int,
        default=1,
        help="Calculate the position from weighted averaged best n. If n = 1, then it is equivalent to top 1"
    )
    parser.add_argument(
        "--separate_branch",
        action="store_true",
        help="Have two separate branches"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images",
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=16,
        help="Batch size for inference (caching and testing)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="triplet",
        help="loss to be used",
        choices=["triplet", "sare_ind", "sare_joint"],
    )
    parser.add_argument(
        "--margin", type=float, default=0.1, help="margin for the triplet loss"
    )
    parser.add_argument(
        "--epochs_num", type=int, default=1000, help="number of epochs to train for"
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--lr_crn_layer",
        type=float,
        default=5e-3,
        help="Learning rate for the CRN layer",
    )
    parser.add_argument(
        "--lr_crn_net",
        type=float,
        default=5e-4,
        help="Learning rate to finetune pretrained network when using CRN",
    )
    parser.add_argument(
        "--optim", type=str, default="adam", help="_", choices=["adam"]
    )
    parser.add_argument(
        "--cache_refresh_rate",
        type=int,
        default=1000,
        help="How often to refresh cache, in number of queries",
    )
    parser.add_argument(
        "--queries_per_epoch",
        type=int,
        default=5000,
        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate",
    )
    parser.add_argument(
        "--negs_num_per_query",
        type=int,
        default=10,
        help="How many negatives to consider per each query in the loss",
    )
    parser.add_argument(
        "--neg_samples_num",
        type=int,
        default=1000,
        help="How many negatives to use to compute the hardest ones",
    )
    parser.add_argument(
        "--mining",
        type=str,
        default="partial",
        choices=["partial", "full", "random", "msls_weighted"],
    )
    # Model parameters
    # parser.add_argument(
    #     "--backbone",
    #     type=str,
    #     default="resnet18conv4",
    #     choices=[
    #         "alexnet",
    #         "vgg16",
    #         "resnet18conv4",
    #         "resnet18conv5",
    #         "resnet50conv4",
    #         "resnet50conv5",
    #         "resnet101conv4",
    #         "resnet101conv5",
    #         "cct384",
    #         "vit",
    #     ],
    #     help="_",
    # )
    parser.add_argument(
        "--l2",
        type=str,
        default="before_pool",
        choices=["before_pool", "after_pool", "none"],
        help="When (and if) to apply the l2 norm with shallow aggregation layers",
    )
    # parser.add_argument(
    #     "--aggregation",
    #     type=str,
    #     default="netvlad",
    #     choices=[
    #         "netvlad",
    #         "gem",
    #         "spoc",
    #         "mac",
    #         "rmac",
    #         "crn",
    #         "rrm",
    #         "cls",
    #         "seqpool",
    #         "none",
    #     ],
    # )
    parser.add_argument(
        "--netvlad_clusters",
        type=int,
        default=64,
        help="Number of clusters for NetVLAD layer.",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="PCA dimension (number of principal components). If None, PCA is not used.",
    )
    parser.add_argument(
        "--num_non_local", type=int, default=1, help="Num of non local blocks"
    )
    parser.add_argument("--non_local", action="store_true", help="_")
    parser.add_argument(
        "--channel_bottleneck",
        type=int,
        default=128,
        help="Channel bottleneck for Non-Local blocks",
    )
    parser.add_argument(
        "--fc_output_dim",
        type=int,
        default=None,
        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.",
    )
    parser.add_argument(
        "--conv_output_dim",
        type=int,
        default=None,
        help="Output dimension of conv layer. If None, don't use a conv layer.",
    )
    parser.add_argument(
        "--unfreeze",
        action='store_true',
        help="Unfreeze the first few layers for backbone",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="imagenet",
        choices=["imagenet", "gldv2", "places", "none"],
        help="Select the pretrained weights for the starting network",
    )
    parser.add_argument(
        "--off_the_shelf",
        type=str,
        default="imagenet",
        choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048",
    )
    parser.add_argument(
        "--trunc_te", type=int, default=None, choices=list(range(0, 14))
    )
    parser.add_argument(
        "--freeze_te", type=int, default=None, choices=list(range(-1, 14))
    )
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to load checkpoint from, for resuming training or testing.",
    )
    # Other parameters
    parser.add_argument("--device", type=str,
                        default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--num_workers", type=int, default=4, help="num_workers for all dataloaders"
    )
    # parser.add_argument(
    #     "--resize",
    #     type=int,
    #     default=[480,480],
    #     nargs=2,
    #     help="Resizing shape for images (HxW).",
    # )
    parser.add_argument(
        "--test_method",
        type=str,
        default="hard_resize",
        choices=[
            "hard_resize",
            "single_query",
            "central_crop",
            "five_crops",
            "nearest_crop",
            "maj_voting",
        ],
        help="This includes pre/post-processing methods and prediction refinement",
    )
    parser.add_argument(
        "--majority_weight",
        type=float,
        default=0.01,
        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement",
    )
    parser.add_argument("--efficient_ram_testing",
                        action="store_true", help="_")
    parser.add_argument("--val_positive_dist_threshold",
                        type=int, default=-1, help="_")
    parser.add_argument(
        "--train_positives_dist_threshold", type=int, default=-1, help="_"
    )
    parser.add_argument(
        "--recall_values",
        type=int,
        default=[1, 5, 10, 20],
        nargs="+",
        help="Recalls to be computed, such as R@5.",
    )
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float,
                        default=None, help="_")
    parser.add_argument("--horizontal_flip", action="store_true", help="_")
    parser.add_argument("--random_resized_crop",
                        type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float,
                        default=None, help="_")
    # Paths parameters

    parser.add_argument(
        "--pca_dataset_folder",
        type=str,
        default=None,
        help="Path with images to be used to compute PCA (ie: pitts30k/images/train",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="vpr",
        help="Folder name of the current run (saved in ./logs/)",
    )
    parser.add_argument('--dataset', type=str, nargs='+',
                    help='List of datasets to use in training and eval',action=StoreWithFlag)
    parser.add_argument('--eval_dataset', default=[],type=str, nargs='+',
                        help='List of datasets to use in training and eval',action=StoreWithFlag)
    parser.add_argument('--wandb_name', default="",type=str, help='Name to append to wandb run',action=StoreWithFlag)   
    parser.add_argument('--backbone_path', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--train_num_workers', type=int, default=4)
    parser.add_argument('--eval_num_workers', type=int, default=4)
    parser.add_argument('--train', default=False,type=bool, help='Mode to build datasets and dataloaders')
    parser.add_argument('--use_odom', default=True,type=bool, help='Mode to build datasets and dataloaders')
    parser.add_argument('--rescale_during_crop', default=False, help='Rescale images during cropping')
    parser.add_argument('--teacher_modality', default='rgb', type=str, help='modality which will be frozen unless "unfreeze teacher" is true')
    parser.add_argument('--student_modality', default='thr', type=str, help='modality for which encoder has to be trained')
    parser.add_argument('--vpr_test', default=True,type=bool, help='Rescale images during cropping')
    parser.add_argument('--same_backbone', action='store_true', help='Rescale images during cropping')
    parser.add_argument('--frozen_backbone', default=True,type=bool, help='Rescale images during cropping')
    parser.add_argument('--un_frozen_layer_index', type=int, nargs='+', default=[],
                    help='List of layer indices to unfreeze')
    parser.add_argument('--head_arch', type=str, choices=['netvlad', 'salad'], default='')
    parser.add_argument('--debug_viz', action='store_true', help='Enable Top-K retrieval visualization')
    #PARV_TODO implement debug_viz
    parser.add_argument('--intra_dataset_batch', type=bool, default=True, help='Enable Top-K retrieval visualization')
    parser.add_argument('--no_crop_images', dest='crop_images', action='store_false', help='Disable image cropping')
    parser.set_defaults(crop_images=True)
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--common_database', default=False,type=bool, help='Use common database for all datasets')
    parser.add_argument('--cart_split', default='vpr',type=str, help='Task to run, currently only vpr is supported')
    parser.add_argument('--debug', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate',action=StoreWithFlag)
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay',action=StoreWithFlag)
    parser.add_argument('--log_train_recall', action='store_true', help='Disable shuffling of dataset')

    args = parser.parse_args()

    # if not args.debug:
    torch.backends.cudnn.benchmark = True  # Provides a speedup


    # args.use_faiss_gpu = True if args.device == "cuda" else False


    # if args.queries_per_epoch % args.cache_refresh_rate != 0:
    #     raise ValueError(
    #         "Ensure that queries_per_epoch is divisible by cache_refresh_rate, "
    #         + f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}"
    #     )

    if torch.cuda.device_count() >= 2 and args.criterion in ["sare_joint", "sare_ind"]:
        raise NotImplementedError(
            "SARE losses are not implemented for multiple GPUs, "
            + f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss."
        )

    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError(
            "msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}"
        )

    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if (
            args.backbone not in ["resnet50conv5", "resnet101conv5"]
            or args.head_arch != "gem"
            or args.fc_output_dim != 2048
        ):
            raise ValueError(
                "Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048"
            )

    if args.prior_location_threshold != -1 and args.prior_location_threshold <= args.val_positive_dist_threshold:
        raise ValueError(f"Prior position theshold is too small to get enough negative samples. Set it to be at least more than {args.val_positive_dist_threshold}")

    if args.use_best_n < 0:
        raise ValueError("use_best_n must be large than or equal to 0")
    
    if args.separate_branch and args.criterion in ["sare_joint", "sare_ind"]:
        raise ValueError("separate_branch currently only supports triplet loss")

    if args.separate_branch and (args.train_batch_size % torch.cuda.device_count() != 0 or args.infer_batch_size % torch.cuda.device_count() != 0):
        raise ValueError("separate_branch requires the batch size is the times of gpu number")

    if args.fc_output_dim is not None and args.conv_output_dim is not None:
        raise ValueError("fc_output_dim and conv_output_dim cannot be used at the same time")

    if args.GAN_save_freq < 0:
        raise ValueError()
    return args


if __name__ == "__main__":
    # Initial setup: parser, logging...
    args = parse_arguments()
    if args.use_extended_data:
        raise NotImplementedError("Please use train_extended.py")

    mp.set_start_method('spawn', force=True)

    start_time = datetime.now()
    dataset_name = "train"+'_'.join(args.dataset)
    if args.eval_dataset:
        dataset_name += "_eval" + '_'.join(args.eval_dataset)


    wandb_name = f"{dataset_name} - {args.wandb_name}"

    wandb_id = str(uuid4())
    args.save_dir = join(
        "checkpoints",
        args.save_dir,
        f"{dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{wandb_id}",
    )
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    wandb.init(project="mm_vpr", config=vars(args),id=wandb_id, name = wandb_name)
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(
        f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs"
    )

    # Creation of Datasets

    train_ds = None
    args.dataset_split_for_eval = "train"
    train_ds = build_dataset(
        args, return_dataloader = False, build_triplets=True
    )

    train_recall_ds = build_dataset(
        args, return_dataloader = False, build_triplets=False
    )

    

    logging.info(f"Train query set: {train_ds}")

    args.dataset_split_for_eval = "val"
    val_ds = build_dataset(
        args, return_dataloader = False, build_triplets=False
    )
    logging.info(f"Val set: {val_ds}")

    if args.head_arch != "":
        agg_dict = build_head_dict(args.head_arch)
    else:
        agg_dict = None
        
    torch.cuda.init()  # explicit init (PyTorch >=2.2)
    torch.randn(1, device="cuda") 

    if not args.separate_branch:
        thr_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,backbone_path = args.backbone_path,modality='thr', device=device,head_config=agg_dict,backbone_model_type="dinov2_vitb14")
        rgb_model = None
        # head_params = chain(thr_model.head.parameters())
        
        warmup_model(thr_model)
    else:
        rgb_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,modality='rgb', device=device,backbone_model_type="dinov2_vitb14",head_config=agg_dict)
        thr_model = MMDistillVPRModel(args=args,frozen_backbone=args.frozen_backbone,un_frozen_layer_index=args.un_frozen_layer_index,frozen_head=False,backbone_path = args.backbone_path,modality='thr', device=device,head_config=agg_dict,backbone_model_type="dinov2_vitb14")
        warmup_model(rgb_model)
        warmup_model(thr_model)
        # head_params = chain(thr_model.head.parameters(), rgb_model.head.parameters())
        
    # backbone_params = chain(*[thr_model.backbone.blocks[i].parameters() for i in args.un_frozen_layer_index])
    # import pdb; pdb.set_trace()  # Debugging line to inspect the model parameters
    

    # model_dict = {"rgb": rgb_model, "thr": thr_model}

    # Initialize model
    # model = network.GeoLocalizationNet(args)
    # model.to(args.device)
    domain_classifier = None
    #PARV_TODO: implement domain classifier and add it to the optimiser params
    if args.DA.startswith("DANN"):
        domain_classifier = model.create_domain_classifier(args)
    if not args.debug:
        if args.head_arch in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
            if not args.resume:
                train_ds.is_inference = True
                if args.separate_branch:
                    logging.info("Initializing NetVLAD layer for RGB and THR models")
                    rgb_model.head[0].initialize_netvlad_layer(
                        args, train_ds.db_dataset, rgb_model)
                    thr_model.head[0].initialize_netvlad_layer(
                        args, train_ds.qu_dataset, thr_model)
                else:
                    logging.info("Initializing NetVLAD layer for THR model")
                    temp_dataset = ConcatDataset(
                        [train_ds.db_dataset, train_ds.qu_dataset])
                    temp_dataset.idx_to_dataset = np.concatenate(
                        [train_ds.db_dataset.idx_to_dataset, train_ds.qu_dataset.idx_to_dataset], axis=0)
                    thr_model.head[0].initialize_netvlad_layer(
                            args, temp_dataset, thr_model)
                # PARV_TODO : add a case where the head is also same - so then we have ot initialise with a combined RGB+THR dataset
    
    # if args.separate_branch:
    #     logging.info('Backbone has separated branched for database and query')
    #     model_db = copy.deepcopy(model)
    #     model_db = torch.nn.DataParallel(model_db)
    #     if torch.cuda.device_count() >= 2:
    #         # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    #         model_db = convert_model(model_db)
    #         model_db = model_db.to(args.device)

    # model = torch.nn.DataParallel(model)
    # if torch.cuda.device_count() >= 2:
    #     # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    #     model = convert_model(model)
    #     model = model.to(args.device)

    if domain_classifier is not None:
        domain_classifier = torch.nn.DataParallel(domain_classifier)
        # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
        domain_classifier = convert_model(domain_classifier)
        domain_classifier = domain_classifier.to(args.device)

    
    
    if args.separate_branch:    
        if rgb_model.trainable_params() or thr_model.trainable_params():
            optimizer = optim.Adam(
                    chain(rgb_model.trainable_params(), thr_model.trainable_params()),
                    lr=args.learning_rate, weight_decay=args.weight_decay
                )
        else:
            optimizer = None
    else:
        if thr_model.trainable_params():
            optimizer = optim.Adam(
                    thr_model.trainable_params(),
                    lr=args.learning_rate, weight_decay=args.weight_decay
            )
        else:
            optimizer = None

    # # Setup Optimizer and Loss
    # if args.head_arch == "crn":
    #     raise NotImplementedError("CRN is not implemented yet for this training script")
    #     if domain_classifier is not None:
    #         raise NotImplementedError("DA for crn is not Implemented")
    #     crn_params = list(model.module.aggregation.crn.parameters())
    #     net_params = list(model.module.backbone.parameters()) + list(
    #         [
    #             m[1]
    #             for m in model.module.aggregation.named_parameters()
    #             if not m[0].startswith("crn")
    #         ]
    #     )
    #     if args.separate_branch:
    #         net_db_params = list(model_db.module.backbone.parameters()) + list(
    #         [
    #             m[1]
    #             for m in model_db.module.aggregation.named_parameters()
    #             if not m[0].startswith("crn")
    #         ]
    #     )
    #     if args.optim == "adam":
    #         if args.separate_branch:
    #             optimizer = torch.optim.Adam(
    #                 [
    #                     {"params": crn_params, "lr": args.lr_crn_layer},
    #                     {"params": net_params, "lr": args.lr_crn_net},
    #                     {"params": net_db_params, "lr": args.lr_crn_net},
    #                 ]
    #             )
    #         else:
    #             optimizer = torch.optim.Adam(
    #                 [
    #                     {"params": crn_params, "lr": args.lr_crn_layer},
    #                     {"params": net_params, "lr": args.lr_crn_net},
    #                 ]
    #             )
    #         logging.info("You're using CRN with Adam, it is advised to use SGD")
    #     elif args.optim == "sgd":
    #         if args.separate_branch:
    #             optimizer = torch.optim.SGD(
    #                 [
    #                     {
    #                         "params": crn_params,
    #                         "lr": args.lr_crn_layer,
    #                         "momentum": 0.9,
    #                         "weight_decay": 0.001,
    #                     },
    #                     {
    #                         "params": net_params,
    #                         "lr": args.lr_crn_net,
    #                         "momentum": 0.9,
    #                         "weight_decay": 0.001,
    #                     },
    #                     {
    #                         "params": net_db_params,
    #                         "lr": args.lr_crn_net,
    #                         "momentum": 0.9,
    #                         "weight_decay": 0.001,
    #                     },
    #                 ]
    #             )
    #         else:
    #             optimizer = torch.optim.SGD(
    #                 [
    #                     {
    #                         "params": crn_params,
    #                         "lr": args.lr_crn_layer,
    #                         "momentum": 0.9,
    #                         "weight_decay": 0.001,
    #                     },
    #                     {
    #                         "params": net_params,
    #                         "lr": args.lr_crn_net,
    #                         "momentum": 0.9,
    #                         "weight_decay": 0.001,
    #                     },
    #                 ]
    #             )
    # else:
    #     if args.optim == "adam":
    #         if args.separate_branch:
    #             if domain_classifier is not None:
    #                 optimizer = torch.optim.Adam(list(model.parameters()) + list(model_db.parameters()) + list(domain_classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    #             else:
    #                 optimizer = torch.optim.Adam(list(model.parameters()) + list(model_db.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    #         else:
    #             if domain_classifier is not None:
    #                 optimizer = torch.optim.Adam(list(model.parameters()) + list(domain_classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    #             else:
    #                 optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     elif args.optim == "sgd":
    #         if args.separate_branch:
    #             if domain_classifier is not None:
    #                 optimizer = torch.optim.SGD(list(model.parameters()) + list(model_db.parameters()) + list(domain_classifier.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.001)
    #             else:
    #                 optimizer = torch.optim.SGD(list(model.parameters()) + list(model_db.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.001)
    #         else:
    #             if domain_classifier is not None:
    #                 optimizer = torch.optim.SGD(list(model.parameters()) + list(domain_classifier.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.001)
    #             else:
    #                 optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

    if args.criterion == "triplet":
        criterion_triplet = nn.TripletMarginLoss(
            margin=args.margin, p=2, reduction="sum")
    elif args.criterion == "sare_ind":
        criterion_triplet = sare_ind
    elif args.criterion == "sare_joint":
        criterion_triplet = sare_joint

    logging.info(f'Domain adapataion: {args.DA}')
    if args.DA.startswith('DANN'):
        criterion_DA = torch.nn.NLLLoss(reduction='sum')

    # Resume model, optimizer, and other training parameters
    if args.resume:
        #PARV_TODO: implement resume functionality
        raise NotImplementedError("Resume functionality is not implemented yet")
        if args.head_arch != "crn":
            if args.separate_branch:
                (
                    model,
                    model_db,
                    optimizer,
                    best_r5,
                    start_epoch_num,
                    not_improved_num,
                ) = util.resume_train_separate(args, model, model_db, optimizer, DA=domain_classifier)
            else:
                (
                    model,
                    optimizer,
                    best_r5,
                    start_epoch_num,
                    not_improved_num,
                ) = util.resume_train(args, model, optimizer, DA=domain_classifier)
        else:
            # CRN uses pretrained NetVLAD, then requires loading with strict=False and
            # does not load the optimizer from the checkpoint file.
            if args.separate_branch:
                model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train_separate(
                args, model, model_db, strict=False, DA=domain_classifier
            )
            else:
                model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(
                args, model, strict=False, DA=domain_classifier
            )
        logging.info(
            f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}"
        )
    else:
        best_r5 = start_epoch_num = not_improved_num = 0

    # if args.backbone.startswith("vit"):
    #     logging.info(f"Output dimension of the model is {args.features_dim}")
    # else:
    #     # model = model.eval()
    #     logging.info(
    #         f"Output dimension of the model is {args.features_dim}"
    #     )
    #     # logging.info(
    #     #     f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}"
    #     # )

    # Training loop
    for epoch_num in range(start_epoch_num, args.epochs_num):
        args.epoch_num = epoch_num
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)
        epoch_triplet_losses = np.zeros((0, 1), dtype=np.float32)
        if args.DA != 'none':
            p = epoch_num / args.epochs_num # p in [0, 1)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            epoch_DA_losses = np.zeros((0, 1), dtype=np.float32)
        # How many loops should an epoch last (default is 5000/1000=5)
        logging.debug(f"Num queries: {len(train_ds.qu_dataset)}")

        loops_num = math.ceil(min(args.queries_per_epoch,len(train_ds.qu_dataset)) / args.cache_refresh_rate)

        if optimizer is not None:
            for loop_num in range(loops_num):
                logging.debug(f"Cache: {loop_num} / {loops_num}")

                # Compute triplets to use in the triplet loss
                train_ds.is_inference = True
                if args.separate_branch:
                    train_ds.compute_triplets(args, model=thr_model, model_db=rgb_model)
                else:
                    train_ds.compute_triplets(args, model=thr_model)
                train_ds.is_inference = False

                batch_sampler = IntraDatasetBatchSampler(train_ds.triplet_idx_to_dataset,args.train_batch_size)

                triplets_dl = DataLoader(
                    dataset=train_ds,
                    num_workers=args.num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=triplet_collate_fn,
                )



                #PARV_TODO make a intrabatch sampler for this
                if rgb_model is not None:
                    rgb_model.train()
                thr_model.train()

                # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=512, W=512
                # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
                for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):
                    torch.cuda.empty_cache()
                    # Flip all triplets or none
                    if args.horizontal_flip:
                        images = transforms.RandomHorizontalFlip()(images)

                    # Compute features of all images (images contains queries, positives and negatives)
                    if args.separate_branch:
                        # model is for query and model_db is for database
                        # query1 + pos1 + neg1s(neg_num) + query2 + pos2 + neg2(neg_num) + ...
                        # Extract query image
                        query_images_index = np.arange(0, len(images), 1 + 1 + args.negs_num_per_query)
                        images_index = np.arange(0, len(images))
                        database_images_index = np.setdiff1d(images_index, query_images_index, assume_unique=True)
                        query_images = images[query_images_index]
                        database_images = images[database_images_index]
                        if args.DA.startswith('DANN'):
                            database_feature, database_reverse_x = model_db(database_images.to(args.device), is_train=True, alpha=alpha)
                            positive_images_index_local = np.arange(0, len(database_reverse_x), 1 + args.negs_num_per_query)
                            if args.DA_only_positive:
                                database_reverse_x = database_reverse_x[positive_images_index_local]
                            database_domain_label = domain_classifier(database_reverse_x)
                            query_feature, query_reverse_x = model(query_images.to(args.device), is_train=True, alpha=alpha)
                            query_domain_label = domain_classifier(query_reverse_x)
                        else:
                            #PARV_TODO: Add is_train when adding DANN to the network
                            database_feature = rgb_model.extract_feature(database_images.to(args.device), test=False)
                            query_feature = thr_model.extract_feature(query_images.to(args.device), test=False)
                        features = torch.empty((len(images), query_feature.shape[1])).to(args.device)
                        features[query_images_index] = query_feature
                        features[database_images_index] = database_feature
                        del database_feature, query_feature
                    else:
                        if args.DA.startswith('DANN'):
                            images_index = np.arange(0, len(images))
                            query_images_index = np.arange(0, len(images), 1 + 1 + args.negs_num_per_query)
                            database_images_index = np.setdiff1d(images_index, query_images_index, assume_unique=True)
                            positive_images_index = np.arange(1, len(images), 1 + 1 + args.negs_num_per_query)
                            features, reverse_x = model(images.to(args.device), is_train=True)
                            if args.DA_only_positive:
                                database_reverse_x = reverse_x[positive_images_index]
                            else:
                                database_reverse_x = reverse_x[database_images_index]
                            query_reverse_x = reverse_x[query_images_index]
                            database_domain_label = domain_classifier(database_reverse_x)
                            query_domain_label = domain_classifier(query_reverse_x)
                        else:
                            features = thr_model.extract_feature(images.to(args.device), test=False)
                    loss_triplet = 0

                    triplet_counter = 0

                    if args.criterion == "triplet":
                        triplets_local_indexes = torch.transpose(
                            triplets_local_indexes.view(
                                -1, args.negs_num_per_query, 3
                            ),
                            1,
                            0,
                        )
                        for triplets in triplets_local_indexes:
                            queries_indexes, positives_indexes, negatives_indexes = triplets.T
                            for i in range(len(queries_indexes)):
                                # queries_indexes[i] = queries_indexes[i].item()
                                # positives_indexes[i] = positives_indexes[i].item()
                                # negatives_indexes[i] = negatives_indexes[i].item()
                                temp_loss_triplet = criterion_triplet(
                                    features[queries_indexes[i].item()],
                                    features[positives_indexes[i].item()],
                                    features[negatives_indexes[i].item()],
                                )
                                if temp_loss_triplet > 0:
                                    triplet_counter += 1
                                loss_triplet += temp_loss_triplet
                    elif args.criterion == "sare_joint":
                        # sare_joint needs to receive all the negatives at once
                        triplet_index_batch = triplets_local_indexes.view(
                            args.train_batch_size, 10, 3
                        )
                        for batch_triplet_index in triplet_index_batch:
                            q = features[batch_triplet_index[0, 0]].unsqueeze(
                                0
                            )  # obtain query as tensor of shape 1xn_features
                            p = features[batch_triplet_index[0, 1]].unsqueeze(
                                0
                            )  # obtain positive as tensor of shape 1xn_features
                            n = features[
                                batch_triplet_index[:, 2]
                            ]  # obtain negatives as tensor of shape 10xn_features
                            loss_triplet += criterion_triplet(q, p, n)
                    elif args.criterion == "sare_ind":
                        for triplet in triplets_local_indexes:
                            # triplet is a 1-D tensor with the 3 scalars indexes of the triplet
                            q_i, p_i, n_i = triplet
                            loss_triplet += criterion_triplet(
                                features[q_i: q_i + 1],
                                features[p_i: p_i + 1],
                                features[n_i: n_i + 1],
                            )

                    del features
                    print("loss_triplet befor normalisation", loss_triplet.item())
                    loss_triplet /= triplet_counter
                    print("loss_triplet after normalisation", loss_triplet.item()," triplet_counter", triplet_counter)

                    if args.DA.startswith('DANN'):
                        query_target_label = torch.zeros(query_domain_label.shape[0]).long().to(args.device)
                        if args.DA_only_positive:
                            # Positive sample num = query sample num
                            database_target_label = torch.ones(query_domain_label.shape[0]).long().to(args.device)
                        else:
                            database_target_label = torch.ones(database_domain_label.shape[0]).long().to(args.device)
                        loss_DA = criterion_DA(query_domain_label, query_target_label) + \
                                criterion_DA(database_domain_label, database_target_label)
                        loss_DA /= query_domain_label.shape[0] + database_domain_label.shape[0]
                        loss = loss_triplet + args.lambda_DA * loss_DA
                    else:
                        loss = loss_triplet

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Keep track of all losses by appending them to epoch_losses
                    triplet_loss = loss_triplet.item()
                    batch_loss = loss.item()
                    epoch_triplet_losses = np.append(epoch_triplet_losses, triplet_loss)
                    epoch_losses = np.append(epoch_losses, batch_loss)
                    if args.DA != 'none':
                        DA_loss = loss_DA.item()
                        epoch_DA_losses = np.append(epoch_DA_losses, DA_loss)
                    del loss

                debug_str = f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): "+ \
                    f"current batch sum loss = {batch_loss:.4f}, "+ \
                    f"average epoch sum loss = {epoch_losses.mean():.4f}, "+ \
                    f"current batch triplet loss = {triplet_loss:.4f}, "+ \
                    f"average epoch triplet loss = {epoch_triplet_losses.mean():.4f}, "

                if args.DA != 'none':
                    debug_str+= f"current batch DA loss = {DA_loss:.4f}, "+ \
                    f"average epoch DA loss = {epoch_DA_losses.mean():.4f}, "

                logging.debug(debug_str)
        
                del triplets_dl
        
        info_str = f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "+ \
            f"average epoch sum loss = {epoch_losses.mean():.4f}, "+ \
            f"average epoch triplet loss = {epoch_triplet_losses.mean():.4f}, "
        if args.DA != 'none':
            info_str += f"average epoch DA loss = {epoch_DA_losses.mean():.4f}, "

        logging.info(info_str)

        

        # Compute recalls on validation set
        train_recalls = None
        train_recalls_str = None
        if args.separate_branch:
            if args.log_train_recall:
                train_recalls, train_recalls_str = test(
                    args, train_recall_ds, thr_model, rgb_model)
                logging.info(f"Recalls on train set {train_recall_ds}: {train_recalls_str}")
            recalls, recalls_str = test(args, val_ds, thr_model, rgb_model)
        else:
            if args.log_train_recall:
                train_recalls, train_recalls_str = test(
                    args, train_recall_ds, thr_model)
                logging.info(f"Recalls on train set {train_recall_ds}: {train_recalls_str}")
            recalls, recalls_str = test(args, val_ds, thr_model)
        logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

        is_best = recalls[1] > best_r5

        # Save checkpoint, which contains all training parameters
        save_checkpoint(
            args,
            epoch_num,
            {
                "epoch_num": epoch_num,
                "model_state_dict": thr_model.state_dict(),
                "model_db_state_dict": rgb_model.state_dict() if args.separate_branch else None,
                "DA_state_dict": domain_classifier.state_dict() if domain_classifier is not None else None,
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "recalls": recalls,
                "best_r5": best_r5,
                "not_improved_num": not_improved_num,
            },
            is_best,
            filename="last_model.pth",
        )

        if args.DA != 'none':
            wandb.log({
                    "epoch_num": epoch_num,
                    "recall1": recalls[0],
                    "recall5": recalls[1],
                    "best_r5": recalls[1] if is_best else best_r5,
                    "sum_loss": epoch_losses.mean(),
                    "triplet loss": epoch_triplet_losses.mean(),
                    "DA loss": epoch_DA_losses.mean()
                },)
        else:
            log_dict = {
                    "epoch_num": epoch_num,
                    "val/recall1": recalls[0],
                    "val/recall5": recalls[1],
                    "val/best_r5": recalls[1] if is_best else best_r5,
                    "sum_loss": epoch_losses.mean(),
                    "triplet loss": epoch_triplet_losses.mean(),
                    "DA loss": 0
                }
        
            if train_recalls is not None:
                log_dict.update({
                    "train/recall1": train_recalls[0],
                    "train/recall5": train_recalls[1],
                })
            wandb.log(log_dict)


        # If recall@5 did not improve for "many" epochs, stop training
        if is_best:
            logging.info(
                f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}"
            )
            best_r5 = recalls[1]
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(
                f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}"
            )
            if not_improved_num >= args.patience:
                logging.info(
                    f"Performance did not improve for {not_improved_num} epochs. Stop training."
                )
                break


    logging.info(f"Best R@5: {best_r5:.1f}")
    logging.info(
        f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}"
    )

    # # Test best model on test set
    # best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))[
    #     "model_state_dict"
    # ]
    # model.load_state_dict(best_model_state_dict)
    # if args.separate_branch:
    #     best_model_db_state_dict = torch.load(join(args.save_dir, "best_model.pth"))[
    #         "model_db_state_dict"
    #     ]
    #     model_db.load_state_dict(best_model_db_state_dict)

    # if args.separate_branch:
    #     recalls, recalls_str = test.test(
    #         args, test_ds, model, model_db=model_db, test_method=args.test_method)
    # else:
    #     recalls, recalls_str = test.test(
    #         args, test_ds, model, model_db=model, test_method=args.test_method)

    # wandb.log({
    #         "final_recall1": recalls[0],
    #         "final_recall5": recalls[1],
    #     },)
            
    # logging.info(f"Recalls on {test_ds}: {recalls_str}")
