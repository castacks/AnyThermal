import sys
import torch
from .base_model import *
from .utils import default_preprocess_tensor

# Add MixVPR repo to path
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition")
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/R2Former")
# Recreate the dataclass after kernel reset
from dataclasses import dataclass, field
from typing import Optional, List
from copy import deepcopy
from R2Former.util import resume_train
from R2Former.model import network
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import faiss.contrib.torch_utils
import time
@dataclass
class MockArgs:
    train_batch_size: int = 4
    infer_batch_size: int = 16
    rerank_batch_size: int = 4
    criterion: str = 'triplet'
    margin: float = 0.1
    epochs_num: int = 50
    patience: int = 3
    lr: float = 0.00001
    warmup: int = -1
    lr_crn_layer: float = 5e-3
    lr_crn_net: float = 5e-4
    optim: str = "adam"
    cos: bool = False
    fix: int = 1
    freeze: int = 0
    save_best: int = 1
    finetune: int = 0
    test: bool = False
    hypercolumn: int = 0
    reg_top: int = 5
    rerank_loss: str = 'ce'
    rerank_model: str = 'r2former'
    schedule: List[int] = field(default_factory=lambda: [60, 80])
    cache_refresh_rate: int = 1000
    queries_per_epoch: int = 5000
    negs_num_per_query: int = 10
    neg_samples_num: int = 1000
    neg_hardness: int = 10
    num_pairs: int = 5
    local_dim: int = 128
    num_local: int = 500
    mining: str = "partial"

    backbone: str = "resnet18conv4"
    l2: str = "before_pool"
    aggregation: str = "netvlad"
    netvlad_clusters: int = 64
    pca_dim: Optional[int] = None
    num_non_local: int = 1
    non_local: bool = False
    channel_bottleneck: int = 128
    fc_output_dim: Optional[int] = None
    pretrain: str = "imagenet"
    off_the_shelf: str = "imagenet"
    trunc_te: Optional[int] = None
    freeze_te: Optional[int] = None

    seed: int = 0
    resume: Optional[str] = None
    device: str = "cuda"
    num_workers: int = 8
    resize: List[int] = field(default_factory=lambda: [480, 640])
    test_method: str = "hard_resize"
    majority_weight: float = 0.01
    efficient_ram_testing: bool = False
    val_positive_dist_threshold: int = 25
    train_positives_dist_threshold: int = 10
    recall_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 100])

    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    hue: Optional[float] = None
    rand_perspective: Optional[float] = None
    horizontal_flip: bool = False
    random_resized_crop: Optional[float] = None
    random_rotation: Optional[float] = None

    datasets_folder: str = "/path/to/datasets"
    dataset_name: str = "pitts30k"
    pca_dataset_folder: Optional[str] = None
    save_dir: str = "default"

class R2FormerFeatureExtractor(BaseFeatureExtractor):
    def build_model(self):
        # Load backbone (ResNet50) and MixVPR head
        args = MockArgs()
        args.resume = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/R2Former/pretrained_models/CVPR23_DeitS_Rerank.pth"
        args.backbone="deit"
        args.aggregation="gem"
        args.fc_output_dim= 256
        args.rerank_model="r2former"
        args.rerank_batch_size= 4
        args.test = True
        args.infer_batch_size = 32
        self.args = args
        model = network.GeoLocalizationNetRerank(args)
        for name, param in model.named_parameters():
            if name.startswith('module.backbone.blocks') and int(name[23]) < args.freeze:
                param.requires_grad = False
            if args.fix and (not 'local_head' in name) and (not 'Reranker' in name):
                param.requires_grad = False
            if param.requires_grad:
                print(name)
        model, _, best_r5, start_epoch_num, not_improved_num = resume_train(args, model, strict=False)

        self.own_recall_method = True
        return model

    def preprocess(self, images, keep_ratio=False,resize=True):
        #PARV_TODO fill this 
        size = self.args.resize
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio, size=size)

    def recall(self,pos_per_query, predictions, top_k_vals, exclude_exact_query_in_db=False):
        recalls = {k: 0 for k in top_k_vals}
        for i, retrieved in enumerate(predictions):
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
        global_recalls = {k: v / total for k, v in recalls.items()}
        return global_recalls
    def evaluate_retrieval(self,db_dataset, qu_dataset, pos_per_query, top_k_vals, use_gpu=False,exclude_exact_query_in_db=False):
        num_local=self.args.num_local
        rerank_dim = self.args.local_dim + 3
        rerank_top = 100
        rerank_bs=2
        save = None
        reg_top = self.args.reg_top
        ransac = False
        threshold = 0
        debug = False

        def extract_all_features(model, dataset, batch_size):
            # Extract features from the database dataset
            # import pdb; pdb.set_trace()
            # all_features = np.empty((len(dataset), self.args.features_dim), dtype="float32")
            # all_features_rerank = np.empty((len(dataset), num_local, rerank_dim), dtype="float32")  
            all_features = []
            all_features_rerank = []
            database_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for inputs, indices in tqdm(database_dataloader, ncols=100):
                features, re_features = model(inputs.to(self.device))
                features = features.cpu().numpy()
                re_features = re_features.cpu().numpy()
                # import pdb; pdb.set_trace()
                all_features.append(features)
                all_features_rerank.append(re_features)
            all_features = np.concatenate(all_features, axis=0).astype(np.float32)
            all_features_rerank = np.concatenate(all_features_rerank, axis=0).astype(np.float32)
            return all_features, all_features_rerank
        # import pdb; pdb.set_trace()
        database_features, database_re_features = extract_all_features(self.model, db_dataset, self.args.infer_batch_size)
        queries_features, queries_re_features = extract_all_features(self.model, qu_dataset, self.args.infer_batch_size)

        faiss_index = faiss.IndexFlatL2(self.args.features_dim)
        faiss_index.add(database_features)
        distances, predictions = faiss_index.search(queries_features, rerank_top)  # max(args.recall_values)

        global_recalls = self.recall(pos_per_query, predictions, top_k_vals, exclude_exact_query_in_db)
        global_indices = deepcopy(predictions)
        # import pdb; pdb.set_trace()
        Reranker = self.model.Reranker
        Reranker.eval()
        sm = torch.nn.Softmax(dim=1)
        similarity = 1. - distances/2.
        ranks = np.array(predictions).copy()
        new_rank = np.copy(ranks)
        rerank_time = 0
        with torch.no_grad():
            for query_index in tqdm(range(0, predictions.shape[0], rerank_bs), ncols=100):
                # print(query_index)
                query_inputs = queries_features[query_index:min(predictions.shape[0], query_index + rerank_bs)]
                query_inputs_expand = np.tile(np.expand_dims(query_inputs, 1), [1, rerank_top, 1]).reshape(
                    [-1, queries_features.shape[-1]])
                query_re_inputs = queries_re_features[query_index:min(predictions.shape[0], query_index + rerank_bs)]
                query_re_inputs_expand = np.tile(np.expand_dims(query_re_inputs, 1), [1, rerank_top, 1, 1]).reshape(
                    [-1, queries_re_features.shape[-2], queries_re_features.shape[-1]])

                candidate_index = predictions[query_index:min(predictions.shape[0], query_index + rerank_bs), :rerank_top]
                candidate_re_inputs = database_re_features[candidate_index.reshape(-1)]
                candidate_inputs = database_features[candidate_index.reshape(-1)]
                # =============================================================================
                query_inputs_cuda = torch.tensor(query_inputs_expand).cuda()
                query_re_inputs_cuda = torch.tensor(query_re_inputs_expand.astype(np.float32)).cuda()
                candidate_inputs_cuda = torch.tensor(candidate_inputs).cuda()
                candidate_re_inputs_cuda = torch.tensor(candidate_re_inputs.astype(np.float32)).cuda()
                # =============================================================================
                time_s = time.time()
                rerank_score_ori, final_score = Reranker(query_inputs_cuda, query_re_inputs_cuda,
                                        candidate_inputs_cuda, candidate_re_inputs_cuda)
                rerank_score = sm(rerank_score_ori)[:, 1]
                rerank_score = torch.reshape(rerank_score,[query_re_inputs.shape[0], rerank_top]).detach()#.cpu().numpy()  # .softmax(dim=1)

                for id, candidates in enumerate(candidate_index):
                    global_score = torch.tensor(similarity[query_index + id]).cuda()
                    rerank_order = torch.argsort(-rerank_score[id]).cpu().numpy()
                    new_rank[query_index + id, :rerank_top] = ranks[query_index + id, :rerank_top][rerank_order]
                rerank_time += (time.time() - time_s)
        new_rerank_recalls = self.recall(pos_per_query, new_rank, top_k_vals, exclude_exact_query_in_db)
        return new_rerank_recalls, new_rank