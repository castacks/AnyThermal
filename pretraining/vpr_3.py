from vpr_2 import *
from custom_datasets.vpr_dataloader import *

# (optional) worker seeding for dataset/augmentations (worker-aware)
from torch.utils.data import get_worker_info
def worker_init_fn(worker_id):
    wi = get_worker_info()
    # PyTorch gives a deterministic base seed per worker via wi.seed
    base_seed = 42
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    
    # if (base_seed + rank * 1000 + worker_id) > 2**32 - 1:
    #     print("Base seed combined with rank and worker_id exceeds 32-bit integer limit.", base_seed, rank, worker_id)
        
    np.random.seed(base_seed + rank * 1000 + worker_id)
    torch.manual_seed(base_seed + rank * 1000 + worker_id)

def faiss_search_in_qchunks(index, xq, k, qbs=64):
    """
    Perform FAISS search in smaller query batches to avoid GPU OOM/cuBLAS errors.

    Args:
        index: FAISS index (CPU or GPU).
        xq (ndarray): Query vectors (nq, d).
        k (int): Number of nearest neighbors.
        qbs (int): Query batch size.

    Returns:
        D (ndarray): Distances (nq, k).
        I (ndarray): Indices (nq, k).
    """
    nq = xq.shape[0]
    Dall = np.empty((nq, k), dtype='float32')
    Iall = np.empty((nq, k), dtype='int64')

    r0 = 0
    while r0 < nq:
        r1 = min(r0 + qbs, nq)
        D, I = index.search(xq[r0:r1], k)
        Dall[r0:r1] = D
        Iall[r0:r1] = I
        r0 = r1
    return Dall, Iall


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
    _,indices   = faiss_search_in_qchunks(index, query_feats, max(ks)+1, qbs=64)
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


def recall_dataloader(args,model_dict, dataloader, device, epoch, train=False):
    db_modality = args.teacher_modality
    q_modality = args.student_modality
    mode = "train" if train else "val"
    positive_index_per_query = np.array(dataloader.dataset.get_hard_positives_per_query(), dtype=object)
    all_rgb_feats = torch.zeros((len(dataloader.dataset), args.features_dim))
    all_thr_feats = torch.zeros((len(dataloader.dataset), args.features_dim))
    all_ground_truth = [[] for _ in range(len(dataloader.dataset))]

    with torch.no_grad():
        for batch_item in tqdm(dataloader, desc=f"{mode.capitalize()} Recall Epoch {epoch}"):
            batch, _ = batch_item["item"]
            indices = batch_item["batch_id"].tolist()
            log_dict = {}
            feats_rgb = model_dict["rgb"].extract_feature(batch[db_modality].to(device), test=False)
            feats_thr = model_dict["thr"].extract_feature(batch[q_modality].to(device), test=False)
            all_rgb_feats[indices] = feats_rgb.cpu()
            all_thr_feats[indices] = feats_thr.cpu()
            import gc; gc.collect()  # Clear memory after processing each batch
            torch.cuda.empty_cache()  # Clear CUDA memory after processing each batch
            torch.cuda.ipc_collect()
            for batch_idx in indices:
                if all_ground_truth[batch_idx] != []:
                    print(f"Overwriting ground truth for index {batch_idx} in {mode} epoch {epoch}")
                    import pdb; pdb.set_trace()  # Debugging line to inspect the ground truth

                all_ground_truth[batch_idx] = positive_index_per_query[batch_idx]
    
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
    else:
        print(f"All ground truth indices are already populated for {mode} epoch {epoch}. No remaining indices to process.")
    for i, gt in enumerate(all_ground_truth):
        if len(gt) == 0:
            print(f"Warning: No ground truth for index {i} in {mode} epoch {epoch}. This might affect recall metrics.")
            import pdb; pdb.set_trace()  # Debugging line to inspect the ground truth


    recall_metrics = compute_recall_at_k(all_thr_feats, all_rgb_feats, positive_index_per_query, exclude_self=True)

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
    
    return recall_metrics


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
    
    # all_ground_truth = [[] for _ in range(len(dataloader.dataset))]
    # all_rgb_feats = torch.zeros((len(dataloader.dataset), args.features_dim))
    # all_thr_feats = torch.zeros((len(dataloader.dataset), args.features_dim))


    db_modality = args.teacher_modality
    q_modality = args.student_modality
    positive_index_per_query = np.array(dataloader.dataset.get_hard_positives_per_query(), dtype=object)
    extra_margin_positive_index_per_query = np.array(dataloader.dataset.get_extra_margin_soft_positives(), dtype=object)


    modality_to_view_map = {"rgb":RGB, "thr":THR}
    db_modality_view = modality_to_view_map[db_modality]
    q_modality_view = modality_to_view_map[q_modality]
    
    # scaler = torch.cuda.amp.GradScaler()
    for batch_item in tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}"):
        images   = batch_item["image"].to(device, non_blocking=True)     # (B, C, H, W)
        indices = batch_item["base_idx"].to(device)                     # (B,)
        view_id  = batch_item["view_id"].to(device)                     # (B,)

        optimizer.zero_grad()

        rgb_idx = torch.where(view_id == RGB)[0]
        rgb_indices = indices[rgb_idx]
        if len(rgb_idx) > 0:
            rgb = images[rgb_idx].to(device)
        else:
            raise ValueError("No RGB images found in the batch")
        thermal_idx = torch.where(view_id == THR)[0]
        thermal_indices = indices[thermal_idx]
        if len(thermal_idx) > 0:
            thermal = images[thermal_idx].to(device)
        else:
            raise ValueError("No Thermal images found in the batch")

        log_dict = {}
    

        with torch.no_grad() if (not train or epoch ==0) else nullcontext():
            with nullcontext():
                feats_rgb = model_dict["rgb"].extract_feature(rgb, test=False)
                feats_thr = model_dict["thr"].extract_feature(thermal, test=False)
            
                if "allign" in args.loss_type:
                    allignmnet_loss = 1 - F.cosine_similarity(feats_rgb, feats_thr, dim=1).mean()

                    log_dict.update({f"{mode}/allignmnet_loss": allignmnet_loss.item()})

                
                feats = torch.cat([feats_rgb, feats_thr], dim=0)
                cat_indices  = torch.cat([torch.tensor(rgb_indices, device=device), torch.tensor(thermal_indices, device=device)], dim=0)

                if "triplet" in args.loss_type or "hard_triplet" in args.loss_type:
                    
                    with torch.no_grad():
                        triplets = get_all_triplets(cat_indices, positive_index_per_query, extra_margin_positive_index_per_query,feats.device)

                        if triplets and "hard_triplet" in args.loss_type:
                            triplets = get_top_n_hardest_triplets_cosine(
                                triplets, feats, current_margin, args.num_negatives_per_positive,verbose=False,hard_frac = args.hard_frac)
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

            


        # assert torch.all(all_rgb_feats[indices] == 0), "all_rgb_feats should be zero before filling"
        # assert torch.all(all_thr_feats[indices] == 0), "all_thr_feats should be zero before filling"

        # all_rgb_feats[rgb_indices] = feats_rgb.cpu().detach()
        # all_thr_feats[thermal_indices] = feats_thr.cpu().detach()
        # for batch_idx in indices:

        #     # if all_ground_truth[batch_idx] != []:
        #     #     print(f"Overwriting ground truth for index {batch_idx} in {mode} epoch {epoch}")
        #     #     import pdb; pdb.set_trace()  # Debugging line to inspect the ground truth

        #     all_ground_truth[batch_idx] = positive_index_per_query[batch_idx]
        
        import gc; gc.collect()  # Clear memory after processing each batch
        torch.cuda.empty_cache()  # Clear CUDA memory after processing each batch
        torch.cuda.ipc_collect()

        print("Cuda memory , after batch: ", torch.cuda.memory_allocated(device) / 1e6, "MB")

    
    # rgb_remaining_indices = [i for i, feats in enumerate(all_rgb_feats) if len(feats) == 0]
    # thr_remaining_indices = [i for i, feats in enumerate(all_thr_feats) if len(feats) == 0]

    # remaining_indices = list(set(rgb_remaining_indices) | set(thr_remaining_indices))
    
    # if remaining_indices:
    #     with torch.no_grad():
    #         remaining_dataset = Subset(dataloader.dataset, remaining_indices)
    #         remaining_dataset.idx_to_dataset = dataloader.dataset.idx_to_dataset[remaining_indices]
    #         sampler = IntraDatasetBatchSampler(remaining_dataset.idx_to_dataset,batch_size=args.eval_batch_size)
    #         remaining_dataloader = DataLoader(remaining_dataset, num_workers=args.eval_num_workers,batch_sampler = sampler)
    #         for batch_item in tqdm(remaining_dataloader, desc=f"{mode.capitalize()} Remaining Epoch {epoch}"):
    #             batch, _ = batch_item["item"]
    #             indices = batch_item["batch_id"].tolist()
    #             rgb = batch[db_modality].to(device)
    #             thermal = batch[q_modality].to(device)
    #             feats_rgb = model_dict["rgb"].extract_feature(rgb, test=False)
    #             feats_thr = model_dict["thr"].extract_feature(thermal, test=False)
    #             all_rgb_feats[indices] = feats_rgb.cpu()
    #             all_thr_feats[indices] = feats_thr.cpu()
    #             # for batch_idx in indices:
    #             #     if all_ground_truth[batch_idx] != []:
    #             #         print(f"Overwriting ground truth for index {batch_idx} in {mode} epoch {epoch}")
    #             #         import pdb; pdb.set_trace()
    #             #     all_ground_truth[batch_idx] = positive_index_per_query[batch_idx]
    
    # for i, gt in enumerate(positive_index_per_query):
    #     if not gt:
    #         print(f"Warning: No ground truth for index {i} in {mode} epoch {epoch}. This might affect recall metrics.")
    #         import pdb; pdb.set_trace()  # Debugging line to inspect the ground truth

    # all_rgb_feats = F.normalize(all_rgb_feats, dim=1)
    # all_thr_feats = F.normalize(all_thr_feats, dim=1)
    # recall_metrics = compute_recall_at_k(all_thr_feats, all_rgb_feats, positive_index_per_query, exclude_self=True)

    # # Optional retrieval visualization
    # if args.debug_viz:
    #     sim = torch.matmul(all_thr_feats, all_rgb_feats.T)
    #     top_k = torch.topk(sim, k=5, dim=1).indices
    #     for idx in random.sample(range(len(all_thr_feats)), min(5, len(all_thr_feats))):
    #         q_img = dataloader.dataset.__getitem__(idx)['thr'].permute(1, 2, 0)
    #         retrieved_imgs = [dataloader.dataset.__getitem__(j)['rgb'].permute(1, 2, 0) for j in top_k[idx]]
    #         fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    #         axs[0].imshow(q_img)
    #         axs[0].set_title("Query")
    #         for i in range(5):
    #             axs[i + 1].imshow(retrieved_imgs[i])
    #             axs[i + 1].set_title(f"Top-{i + 1}")
    #         wandb.log({f"{mode}/retrieval_{idx}": wandb.Image(fig)})
    #         plt.close(fig)
    

    return total_loss_vpr / total_loss_vpr_updated , total_loss_allignment / total_loss_allignment_updated


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


def build_dataloader(dataset,args):
    view_dataset = ViewIndexingDataset(dataset, rgb_key="rgb", thr_key="thr")

    sampler = IntraDatasetViewBatchSamplerV2(
        wrapper=dataset,
        anchors_per_batch=args.anchors_per_batch,   # anchors per batch (each adds (a,RGB)+(a,THR))
        k_hard_pos=args.hard_pos_per_anchor,           # n hard extra positives per anchor
        k_soft_pos=args.soft_pos_per_anchor,           # m soft extra positives per anchor
        neg_pool=args.neg_pos_per_anchor,            # ring negatives per anchor
        steps_per_epoch=args.steps_per_epoch,   # define your epoch length in #batches
        dataset_mix=None,       # or {"MS2Dataset":0.5, "CityGPS":0.5}
        seed=42,
        pos_view_policy="balanced",
        neg_view_policy="balanced",
    )

    loader = DataLoader(
        view_dataset,                 # <- the adapter dataset
        batch_sampler=sampler,        # <- IMPORTANT: use batch_sampler, not sampler
        num_workers=args.train_num_workers,      # set as you like
        pin_memory=False,
        persistent_workers=False,      # optional, saves worker startup time
        worker_init_fn=worker_init_fn # ensures dataset/aug randomness differs per worker
    )

    return loader
    

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
    if args.hard_frac < 1.0:
        wandb_name += f"_hard_frac_{args.hard_frac}"

    wandb.init(project="mm_vpr", name=wandb_name)
    args.save_dir = os.path.join(args.save_dir, time.strftime("%Y-%m-%d_%H-%M-%S")+wandb_name)
    os.makedirs(args.save_dir, exist_ok=True)

    agg_dict = build_head_dict(args.head_arch)
    args.agg_dict = agg_dict

    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recall_train_dataloader, recall_val_dataloader = build_dataset(args)

    train_dataloader = build_dataloader(recall_train_dataloader.dataset, args)
    val_dataloader = build_dataloader(recall_val_dataloader.dataset, args)

    
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

        print(f"Starting epoch {epoch}...")
        train_dataloader.batch_sampler.set_epoch(epoch)
        val_dataloader.batch_sampler.set_epoch(epoch)

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
        if epoch % args.save_interval == 0:

            save_dict = {"thermal_state_dict": thr_model.state_dict()}
            if not args.same_backbone:
                save_dict["rgb_state_dict"] = rgb_model.state_dict()

            torch.save(save_dict, os.path.join(args.save_dir, f"model_{epoch}.pth"))


        train_loss_vpr, train_loss_align = run(
            args,model_dict, train_dataloader if args.use_vpr_dataloader else recall_train_dataloader, optimizer, device, epoch, train=True, current_margin=current_margin
        )
        train_recall_metrics = recall_dataloader(args,model_dict, recall_train_dataloader, device, epoch,train=True)
        log_dict = {"epoch": epoch, "train/avg_loss_vpr": train_loss_vpr, "train/avg_loss_align": train_loss_align,"sched/current_margin": current_margin}
        for k, v in train_recall_metrics.items():
            log_dict.update({f"train/{k}": v})
        wandb.log(log_dict)
        
        with torch.no_grad():
            log_dict = {"epoch": epoch}
            if args.log_val_triplet_loss:
                val_loss_vpr, val_loss_align = run(
                    args,model_dict, val_dataloader if args.use_vpr_dataloader else recall_val_dataloader, optimizer, device, epoch, train=False, current_margin=current_margin
                )
                log_dict.update({"val/avg_loss_vpr": val_loss_vpr, "val/avg_loss_align": val_loss_align})

            val_recall_metrics = recall_dataloader(args,model_dict, recall_val_dataloader, device, epoch)
            for k, v in val_recall_metrics.items():
                log_dict.update({f"val/{k}": v})
            wandb.log(log_dict)
            last_val_metrics = val_recall_metrics  # for metric-based curriculum if enabled

        print(f"Epoch {epoch} - Train Loss: {train_loss_vpr+train_loss_align:.4f}")    
        
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
    
    parser.add_argument('--hard_frac', type=float, default=0.5,
                        help='Fraction of hard triplets to use in each batch. Only used if hard_triplet loss is selected.')

    parser.add_argument('--anchors_per_batch', type=int, default=32,
                        help='Fraction of hard triplets to use in each batch. Only used if hard_triplet loss is selected.')
    parser.add_argument('--hard_pos_per_anchor', type=int, default=2,
                        help='Fraction of hard triplets to use in each batch. Only used if hard_triplet loss is selected.')
    parser.add_argument('--soft_pos_per_anchor', type=int, default=0,
                        help='Fraction of hard triplets to use in each batch. Only used if hard_triplet loss is selected.')
    parser.add_argument('--neg_pos_per_anchor', type=int, default=6,
                        help='Fraction of hard triplets to use in each batch. Only used if hard_triplet loss is selected.')
    parser.add_argument('--steps_per_epoch', type=int, default=200,
                        help='Fraction of hard triplets to use in each batch. Only used if hard_triplet loss is selected.')
    parser.add_argument('--log_val_triplet_loss', action='store_true', help='Disable shuffling of dataset')
    parser.add_argument('--use_vpr_dataloader', type = bool, default = True, help='Use VPR dataloader for training and validation')


    args = parser.parse_args()

    args.eval_batch_size = args.batch_size if args.eval_batch_size == -1 else args.eval_batch_size

    args.frozen_backbone = True if args.un_frozen_layer_index == [] else False
    if args.un_frozen_layer_index != []:
        args.un_frozen_layer_index.append("norm")
 
    assert args.conv_output_dim<0 or args.fc_output_dim<0, "conv_output_dim and fc_output_dim cannot be both set."
    
    args.dataset = sorted(args.dataset)
    args.eval_dataset = sorted(args.eval_dataset) if args.eval_dataset else []
    main(args)
