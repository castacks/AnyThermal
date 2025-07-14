import torch
import torch.nn as nn
import torch.nn.functional as torch_func
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
import torch.optim as optim
import time
import os
import argparse
import wandb
from tqdm import tqdm
import sys
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc')
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/custom_datasets") # Add the custom models directory to the path

from contextlib import nullcontext
from itertools import chain
from custom_datasets.multi_dataset_loader import *
import gc
from utilities import DinoV2ExtractFeatures
from torchvision.utils import save_image
from custom_models.mmdistill_dinov2_model import MMDistillDinov2
from sklearn.decomposition import PCA
import yaml
import cv2
import uuid
from losses.loss_combiner import LossManager
viz_freq = 10
torch.backends.cudnn.benchmark = True  # Provides a speedup
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student_model = None
teacher_model = None
student_patch_size = None
teacher_patch_size = None
optimizer = None
scheduler = None

def save_viz_debug_images(images_dict, index, epoch, batch_num, split, save_root):
    """
    Save RGB and thermal images to disk for visualization/debugging.
    """
    for modality, tensor in images_dict.items():
        # Shape: (B, C, H, W)
        save_dir = os.path.join(save_root,"raw_viz", modality)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(tensor.shape[0]):
            if i % viz_freq != 0:
                continue
            filename = f"{index[i] if index is not None else i}.png"
            path = os.path.join(save_dir, filename)
            save_image(unnormalise_images(tensor[i], modality), path)
            

def viz_attention_pca(args,teacher_attention_maps,student_attention_maps,teacher_prediction,teacher_prediction_on_student,student_prediction, student_dual_prediction,save_dir, epoch, batch_index,image_shape):
    """
    Visualize attention maps using PCA and save them to disk.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    assert student_dual_prediction is None, "Student dual prediction is not supported yet. Please disable the student_dual_prediction argument."

    save_dir = os.path.join(save_dir)
    if not os.path.exists(os.path.join(save_dir, "attention_map")):
        os.makedirs(os.path.join(save_dir, "attention_map"), exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, "pca")):
        os.makedirs(os.path.join(save_dir, "pca"), exist_ok=True)


    def calculate_pca_map(patch_tokens, patch_tokens_shape, image_shape, pca=None):
        reshaped_patch_tokens = patch_tokens.permute(1, 2, 0).reshape(-1, patch_tokens.shape[0]).cpu().detach().numpy()  # Reshape to (-1, D)
        own_pca = PCA(n_components=3).fit(reshaped_patch_tokens)
        
        pca_img, own_pca_img = None, None

        if pca is None:
            pca_list = [own_pca]
        else:
            pca_list = [own_pca,pca]

        output_list = []
        
        for cur_pca in pca_list:

            temp_pca_img = cur_pca.transform(reshaped_patch_tokens)
            temp_pca_img -= temp_pca_img.min(0)
            temp_pca_img /= temp_pca_img.max(0)
            temp_pca_img = temp_pca_img.reshape(patch_tokens_shape[0], patch_tokens_shape[1], 3)
            temp_pca_img = cv2.resize(temp_pca_img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)
            output_list.append(temp_pca_img)

        own_pca_img = output_list[0]
        if len(output_list) > 1:
            pca_img = output_list[1]
        else:
            pca_img = None            
        return own_pca,own_pca_img, pca_img
    
    

    for i, (teacher_map,student_map) in enumerate(zip(teacher_attention_maps, student_attention_maps)):
        if i % viz_freq != 0:
            continue
        # attn_map shape: (Heads, N, N)
        for layer in teacher_prediction.keys():
            is_shared_specific = False
            for discard_k in ["shared", "specific"]:
                if discard_k in layer:
                    is_shared_specific = True
                    break
            if is_shared_specific:
                break
            patch_tokens = {
                "teacher": {},
                "student": {},
                "teacher_on_student": {}
            }
            patch_tokens["teacher"]["pred"] = teacher_prediction[layer][0][i]
            patch_tokens["student"]["pred"] = student_prediction[layer][0][i]
            patch_tokens["teacher_on_student"]["pred"] = teacher_prediction_on_student[layer][0][i]

            patch_tokens["teacher"]["dim_size"] = teacher_prediction[layer][0][i].shape[0]
            patch_tokens["student"]["dim_size"] = student_prediction[layer][0][i].shape[0]
            patch_tokens["teacher_on_student"]["dim_size"] = teacher_prediction_on_student[layer][0][i].shape[0]

            patch_tokens_shape = patch_tokens["student"]["pred"].shape[1:]
            
            teacher_pca,teacher_pca_img, _ = calculate_pca_map(patch_tokens["teacher"]["pred"], patch_tokens_shape, image_shape)
            _,student_pca_img, student_pca_img_teacher_pca = calculate_pca_map(patch_tokens["student"]["pred"], patch_tokens_shape, image_shape, pca=teacher_pca)
            _,teacher_on_student_pca_img, teacher_on_student_pca_img_teacher_pca = calculate_pca_map(patch_tokens["teacher_on_student"]["pred"], patch_tokens_shape, image_shape, pca=teacher_pca)

            teacher_pca_img_list = [teacher_pca_img,teacher_pca_img]
            student_pca_img_list = [student_pca_img,student_pca_img_teacher_pca]
            teacher_prediction_on_student_pca_img_list = [teacher_on_student_pca_img,teacher_on_student_pca_img_teacher_pca]

            
            for j, (temp_teacher_pca_img, temp_student_pca_img, temp_teacher_on_student_pca_img) in enumerate(zip(teacher_pca_img_list, student_pca_img_list, teacher_prediction_on_student_pca_img_list)):
                if j == 0:
                    layer_name = "individual"
                elif j == 1:
                    layer_name = "common"
                else:
                    raise ValueError("Invalid index for layer name. Expected 0 or 1, got {}".format(j))
                fig, axs = plt.subplots(1, 3, figsize=(20, 5))


                axs[0].imshow(temp_teacher_pca_img)
                axs[0].axis("off")
                axs[0].set_title(f"Teacher PCA of patch tokens with ({layer_name}) PCA")
                axs[1].imshow(temp_student_pca_img)
                axs[1].axis("off")
                axs[1].set_title(f"Student PCA of patch tokens with ({layer_name}) PCA")
                axs[2].imshow(temp_teacher_on_student_pca_img)
                axs[2].axis("off")
                axs[2].set_title(f"Teacher on Student PCA of patch tokens with ({layer_name}) PCA")
                plt.tight_layout()
                file_name = os.path.join(save_dir, "pca",layer,f"{batch_index[i]:06d}_{layer_name}.png")
                os.makedirs(os.path.join(save_dir, "pca",layer), exist_ok=True)
                plt.savefig(file_name)
                plt.close(fig)

def init_model(args,modality,un_frozen_layer_index):
    if un_frozen_layer_index != []:
        if args.unfreeze_patch_embed:
            un_frozen_layer_index = ["patch_embed"] + un_frozen_layer_index
        if not args.not_unfreeze_final_norm:
            un_frozen_layer_index.append("norm")
        if args.unfreeze_cls_token:
            un_frozen_layer_index.append("cls_token")
        if args.unfreeze_pos_embed:
            un_frozen_layer_index.append("pos_embed")
        if args.unfreeze_register_tokens:
            un_frozen_layer_index.append("register_tokens")
    
    if args.proj_head:
        un_frozen_layer_index.append("proj_head")

    model = MMDistillDinov2(args.model_name, modality=modality,un_frozen_layer_index= un_frozen_layer_index,layers_to_hook=args.loss_object.layers,proj_head=args.proj_head)

    return model, model.patch_size

def transform_images(modality,images,patch_size=14):
    # Resize the images to the required input size
    images = transforms.functional.normalize(images, mean=[0.481, 0.457, 0.408], std=[0.269, 0.26, 0.275])
    width = images.shape[-2]
    height = images.shape[-1]
    new_width = (width // patch_size) * patch_size
    new_height = (height // patch_size) * patch_size
    images = torch_func.interpolate(images, size=(new_width, new_height), mode='bilinear', align_corners=False)
    return images

def unnormalise_images(images, modality):
    if modality == "rgb":
        mean = torch.tensor([0.481, 0.457, 0.408], device=images.device).view(-1, 1, 1)
        std = torch.tensor([0.269, 0.260, 0.275], device=images.device).view(-1, 1, 1)
    elif modality == "thr" or modality == "thr_dual":
        mean = torch.tensor([0.481, 0.457, 0.408], device=images.device).view(-1, 1, 1)
        std = torch.tensor([0.269, 0.260, 0.275], device=images.device).view(-1, 1, 1)

    images = images * std + mean
    return images


def generate_positive_masks(index_dict,batch_indices,device):
    B = len(batch_indices)
    neg_mask = torch.ones((B, B), dtype=torch.bool, device=device)
    neg_mask.fill_diagonal_(False)  # don't count positives

    if index_dict is not None and batch_indices is not None:
        for i in range(B):
            anchor_idx = batch_indices[i]
            positives = set(index_dict[anchor_idx])
            for j in range(B):
                if j == i:
                    continue
                other_idx = batch_indices[j]
                if other_idx in positives:
                    # import pdb; pdb.set_trace()
                    neg_mask[i, j] = False  # filter out false negatives
    # import pdb; pdb.set_trace()
    return ~neg_mask

def cosine_positive_negative_loss(student_embed: torch.Tensor,
                                   teacher_embed: torch.Tensor,
                                   positive_mask: torch.Tensor,):
    """
    Computes positive and negative cosine similarity losses.
    Filters false negatives using `index_dict` and `batch_indices`.

    Args:
        student_embed: (B, D) student (thermal) embeddings
        teacher_embed: (B, D) teacher (RGB) embeddings
        index_dict: dict[int, list[int]], mapping dataset index → list of true positives
        batch_indices: list[int] of dataset indices for the current batch, aligned with embeddings

    Returns:
        pos_loss: average cosine distance for true positives
        neg_loss: average cosine similarity for filtered negatives
    """
    # Normalize
    student_embed = torch_func.normalize(student_embed, dim=1)
    teacher_embed = torch_func.normalize(teacher_embed, dim=1)

    B = student_embed.size(0)
    sim_matrix = torch.matmul(student_embed, teacher_embed.T)  # (B, B)

    # Positive (diagonal)
    pos_loss = 1 - sim_matrix[positive_mask].mean()
    neg_loss = 1-sim_matrix[~positive_mask].mean()

    return pos_loss.item(), neg_loss.item()

def print_model_gradients(model):
    for name,param in model.named_parameters():
        if param.requires_grad:
            print(f"{name},{param.grad.norm() if param.grad is not None else 'No gradient'}")

def detect_embedding_collapse(embedding: torch.Tensor, threshold: float = 1e-3):
    """
    Detects collapse in a batch of embeddings.

    Args:
        embedding (torch.Tensor): Tensor of shape (B, D), where B = batch size, D = embedding dimension
        threshold (float): Threshold for determining active dimensions

    Returns:
        dict: Dictionary of collapse statistics
    """
    B, D = embedding.shape

    # L2 Norm stats (optional)
    norms = torch.norm(embedding, dim=1)
    norm_mean = norms.mean().item()
    norm_std = norms.std().item()

    # Feature-wise std across the batch (main signal)
    batch_std = torch.std(embedding, dim=0)  # shape: (D,)
    dim_std_mean = batch_std.mean().item()
    dim_std_min = batch_std.min().item()
    active_dim_ratio = (batch_std > threshold).float().mean().item()

    # Optional: Cosine similarity collapse
    with torch.no_grad():
        emb_norm = torch_func.normalize(embedding, dim=1)  # (B, D)
        cos_sim_matrix = emb_norm @ emb_norm.T   # (B, B)
        mask = ~torch.eye(B, dtype=torch.bool, device=embedding.device)
        cosine_sim_mean = cos_sim_matrix[mask].mean().item()

    return {
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "dim_std_mean": dim_std_mean,
        "dim_std_min": dim_std_min,
        "active_dim_ratio": active_dim_ratio,
        "cosine_sim_mean": cosine_sim_mean,
        "collapse_detected": (
            dim_std_mean < threshold or
            active_dim_ratio < 0.1 or
            cosine_sim_mean > 0.99
        )
    }

def inference(args,teacher_model,student_model, teacher_modality,student_modality,images,batch_indices,test=False,soft_positives_per_query=None, epoch =0):
    with (torch.inference_mode() if test else nullcontext()):
        if soft_positives_per_query is not None:
            pos_masks = generate_positive_masks(soft_positives_per_query, batch_indices, device='cuda')
        else:
            pos_masks = None

        pos_masks = torch.eye(len(batch_indices), dtype=torch.bool, device=global_device)
        if args.unfreeze_teacher:
            teacher_output = teacher_model.forward_train(images[teacher_modality])
        else:
            #PARV_DEBUG - remove torch inference modes if you have to leanr projection heads proj_head 
            with torch.inference_mode():
                teacher_output = teacher_model.forward_train(images[teacher_modality])
                teacher_output_on_student = teacher_model.forward_train(images[student_modality])

        
        student_output = student_model.forward_train(images[student_modality])
        if args.student_modality_dual is not None:
            student_dual_output = student_model.forward_train(images[args.student_modality_dual])
        else:
            student_dual_output = None
        
        total_loss, individual_losses = args.loss_object.compute(student_output, student_dual_output,teacher_output)
        if not test and not args.dry_run and epoch != 0:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        del total_loss

        output_list = [teacher_output, teacher_output_on_student, student_output, student_dual_output]
        for i in range(len(output_list)):
            if output_list[i] is None:
                continue
            for name, output in output_list[i].items():
                if output_list[i][name][0] is not None:
                    output_list[i][name] = (output_list[i][name][0].cpu(), output_list[i][name][1].cpu())
        
        mode = "val" if test else "train"
        if args.wandb_use:
            log_dict = {}
            for modalty,output in zip(["teacher", "student"], [teacher_output, student_output]):
                for name in output.keys():
                    if 'final' not in name:
                        continue
                    if output[name][1] is None:
                        continue
                    collapse_dict = detect_embedding_collapse(output[name][1])
                    for key, value in collapse_dict.items():
                        log_dict[f"collapse check for {mode} {modalty} {name}/{key}"] = value
            
            wandb.log(log_dict)
                
        return teacher_output,teacher_output_on_student,student_output,student_dual_output,individual_losses,None,None

def dataloader_loop(args,save_path, teacher_model, student_model,dataloader, test=False, epoch=0, soft_positives_per_query=None):
    """
    Loop through the dataloader and perform inference or training.
    """

    total_individual_losses = {}
    for name in args.loss_object.losses.keys():
        for layer in args.loss_object.layers:
            total_individual_losses[f"{name}_layer_{layer}"] = 0.0

    total_positive_cosine_loss = 0.0
    total_negative_loss = 0.0
    num_batches = len(dataloader)
    split = "val" if test else "train"

    data_iter = iter(dataloader)


    for batch_number, batch in enumerate(tqdm(data_iter)):
        viz_debug_dir = os.path.join(save_path, "viz_debug",f"epoch_{epoch:03d}", split) if args.viz_debug else None

        images_dict,_ = batch["item"]
        index = batch["batch_id"]
        images = {}
        for k,v in images_dict.items():
           patch_size = teacher_patch_size if k == args.teacher_modality else student_patch_size
           images[k] = transform_images(k, v, patch_size).to(global_device)

        if args.viz_debug:
            save_viz_debug_images(images, index, epoch, batch_number, split, viz_debug_dir)

        
        if args.viz_attention_pca:
            teacher_attention_maps = []
            student_attention_maps = []

            def teacher_hook_fn_forward(module, input, output):
                teacher_attention_maps.append(output)  # output shape: (B, Heads, N, N)
            
            def student_hook_fn_forward(module, input, output):
                student_attention_maps.append(output)

            teacher_hook = teacher_model.model.blocks[-1].attn.attn_drop.register_forward_hook(teacher_hook_fn_forward)
            student_hook = student_model.model.blocks[-1].attn.attn_drop.register_forward_hook(student_hook_fn_forward)
            

        teacher_output,teacher_output_on_student, student_output,student_dual_output,individual_losses, positive_loss, negative_loss = inference(
            args, teacher_model, student_model,
            args.teacher_modality, args.student_modality,
            images=images, batch_indices=index.tolist(),test=test,
            soft_positives_per_query=soft_positives_per_query, epoch=epoch)
        
        if args.viz_attention_pca:
            teacher_hook.remove()
            student_hook.remove()
            viz_attention_pca(args,teacher_attention_maps[0],student_attention_maps[0], teacher_output,teacher_output_on_student,student_output, student_dual_output,viz_debug_dir, epoch, index, images[args.teacher_modality].shape[-2:])

        del teacher_output, student_output, student_dual_output, teacher_output_on_student

        for name, loss in individual_losses.items():
            if name not in total_individual_losses:
                raise ValueError(f"Loss name {name} not found in total_individual_losses dictionary. Available keys: {list(total_individual_losses.keys())}")
            total_individual_losses[name] += loss
        if positive_loss is not None and negative_loss is not None:
            total_positive_cosine_loss += positive_loss
            total_negative_loss += negative_loss

        if (batch_number + 1) % 10 == 0:
            debug_str = f"[Epoch {epoch}, Batch {batch_number+1}] "
            for name, loss in total_individual_losses.items():
                debug_str += f"{name}={loss/(batch_number+1):.4f}, "
            print(debug_str)

        torch.cuda.empty_cache()
        gc.collect()

    for name, loss in total_individual_losses.items():
        total_individual_losses[name] /= num_batches
    avg_positive_cosine_loss = total_positive_cosine_loss / num_batches if total_positive_cosine_loss > 0 else None
    avg_negative_cosine_loss = total_negative_loss / num_batches if total_negative_loss > 0 else None

    return total_individual_losses, avg_positive_cosine_loss, avg_negative_cosine_loss


def parse_yaml_to_args(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    args_list = []
    for key, value in config.items():
        key = f"--{key}"
        if isinstance(value, bool):
            if value:
                args_list.append(key)
        else:
            args_list.extend([key, str(value)])
    return args_list


def rewrite_args_dict(old_args, new_args):
    """
    Rewrite old_args with new_args, keeping the old args if not present in new_args.
    """
    for key, value in new_args.items():
        if key in old_args:
            old_args[key] = value
        else:
            raise ValueError(f"Key {key} not found in old_args. Available keys: {list(old_args.keys())}")
    return old_args

def dict_to_args(parser,config):
    args_list = []
    for k, v in config.items():
        if "__passed" in k:
            continue
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                args_list.append(key)
        elif isinstance(v, list):
            if not v:
                continue  # skip empty lists (esp. for nargs='+')
            args_list.append(key)
            args_list.extend(str(item) for item in v)
        else:

            args_list.extend([key, str(v)])
    
    return parser.parse_args(args_list)

class StoreWithFlag(argparse.Action):
    def __init__(self, *args, **kwargs):
        self._seen = False
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Handle store_true behavior (flag with no value)
        if self.nargs == 0:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, values)
        setattr(namespace, f"__passed_{self.dest}", True)

def was_passed(namespace, name):
    return getattr(namespace, f"__passed_{name}", False)

def parse_args_and_get_passed(parser):
    temp_args = parser.parse_args()
    all_args = vars(temp_args)
    passed_args = {
        k: v for k, v in all_args.items()
        if was_passed(temp_args, k)
    }
    return all_args, passed_args

    
def train():

    global student_model, teacher_model, student_patch_size, teacher_patch_size, optimizer, scheduler
    all_args, passed_args = parse_args_and_get_passed(parser)
    print(passed_args)
    start_epoch = all_args['resume_epoch_num']
    save_path = all_args['save_path']
    resume = all_args['resume']

    # Load the checkpoint if resuming training
    loss_file = None
    if resume:
        #Loading older args
        with open(os.path.join(save_path, "args.yaml"), 'r') as f:
            old_args = yaml.safe_load(f)
        args = rewrite_args_dict(old_args, passed_args)
        args = dict_to_args(parser,args)



        checkpoint = torch.load(os.path.join(save_path, "model" +str(start_epoch-1) + '.pth'))
        loss_file = os.path.join(save_path, "loss_config.yaml")
        if args.unfreeze_teacher:
            teacher_weights = checkpoint['teacher_model_state_dict']
        else:
            # Load the teacher model weights if unfreezing is not enabled
            teacher_weights = None

        student_weights = checkpoint['student_model_state_dict']
        
        optimizer_dict = checkpoint['optimizer_state_dict']
        assert checkpoint['epoch'] + 1 == start_epoch, f"Checkpoint epoch {checkpoint['epoch']+1} does not match start_epoch {start_epoch}."
        if args.lr_scheduler:
            for e in range(start_epoch):
                scheduler.step()
        print("Resuming training from epoch {}".format(start_epoch))
    else:
        args = dict_to_args(parser,all_args)
        wandb_id = str(uuid.uuid4())
        if args.wandb_id != "":
            raise ValueError("wandb_id should not be set when not resuming. It will be automatically generated.")
        args.wandb_id = wandb_id
        save_path = args.save_path
        dataset_name = "_".join(args.dataset)
        if args.eval_dataset:
            dataset_name += "_eval_" + "_".join(args.eval_dataset)

        wandb_name = args.model_name + "_" + dataset_name + "_" + args.student_modality + "_distill"
        if args.wandb_name != "":
            wandb_name += "_" + args.wandb_name

        save_path = os.path.join(save_path,dataset_name, f"{args.teacher_modality}_{args.student_modality}")
        #append currnet date and time to the save path
        save_path = os.path.join(save_path, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{wandb_name}')
        loss_file = args.loss_file

        # Create the save directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path,'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)
        # dump loss yaml file 
        with open(os.path.join(save_path,'loss_config.yaml'), 'w') as f:
            # copy the loss config file - args.loss_file
            with open(args.loss_file, 'r') as loss_f:
                loss_config = yaml.safe_load(loss_f)
            yaml.dump(loss_config, f)
    
    args.crop_images = not args.no_crop_images
    args.intra_dataset_batch = not args.no_intra_dataset_batch

    args.viz_debug = args.viz_debug or args.viz_attention_pca
    if loss_file is None:
        raise ValueError("Loss file is not specified. Please provide a valid loss file path.")
    args.loss_object = LossManager(loss_file)
    if args.loss_object.get_second_view:
        assert args.student_modality =='thr', "Second view student output is required for losses that use it. Please set student_modality to 'thr' for thermal modality."
        args.student_modality_dual = args.student_modality+"_dual"
    else:
        args.student_modality_dual = None
    teacher_model,teacher_patch_size = init_model(args,modality=args.teacher_modality,un_frozen_layer_index=[])
    student_model,student_patch_size = init_model(args,modality=args.student_modality,un_frozen_layer_index=args.un_frozen_layer_index)
    
    if args.resume:
        if teacher_weights is not None:
            teacher_model.load_model(teacher_weights)
        student_model.load_model(student_weights)
    
    if args.wandb_use:
        if args.resume:
            wandb.init(project="multiloc", id=args.wandb_id, resume="must")  # will resume if args.wandb_id exists, else start new
        else:
            wandb.init(project="multiloc", name=wandb_name, id=args.wandb_id, resume="allow")
    
    if args.unfreeze_teacher:
        teacher_model.model.train()
    else:
        # Freeze the RGB model, so that only the thermal model is trained
        teacher_model.model.eval()
        for param in teacher_model.model.parameters():
            param.requires_grad = False

    student_model.train()

    if args.unfreeze_teacher:
        raise ValueError("Unfreezing teacher model is not supported yet. Please use the default frozen teacher model. Else enable freezing and unfreew layers through optimisation")
    else:
        from itertools import chain

        unfrozen_layers = student_model.un_frozen_layer_index
        print("Unfrozen layers for optimisation: ", unfrozen_layers)

        if args.unfreeze_patch_embed:
            raise ValueError("Unfreezing patch embed is not supported yet. Please use the default frozen patch embed. Else enable freezing and unfreew layers through optimisation take into account other params like cls_token, pos_embed, norm, etc.")
            param_list = [
                    {"params": student_model.model.patch_embed.parameters(), "lr": args.learning_rate * 0.1},  # 0.0001
                    {"params": student_model.model.blocks.parameters(), "lr": args.learning_rate},             # 0.001
                    {"params": student_model.model.norm.parameters(), "lr": args.learning_rate},            # 0.001
                ]
            if student_model.modality_specific_proj_head is not None:
                param_list.append({"params": student_model.modality_specific_proj_head.parameters(), "lr": args.learning_rate})
                param_list.append({"params": student_model.modality_shared_proj_head.parameters(), "lr": args.learning_rate})

        else:
            param_list =[{"params":chain(student_model.unfrozen_parameters()), "lr": args.learning_rate}]

        # Create the optimizer
        if args.optimizer == "adamw":
            optimizer = optim.AdamW(param_list, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(param_list, weight_decay=args.weight_decay)
        else:
            raise ValueError("Invalid optimizer name. Please use 'adamw' or 'sgd'.")

        if args.resume:
            optimizer.load_state_dict(optimizer_dict)
            print("Optimizer state loaded from checkpoint.")


    for name,param in student_model.unfrozen_named_parameters():
        if param.requires_grad and not any(param is p for group in optimizer.param_groups for p in group['params']):
            print(f"{name} is not in the optimizer but requires_grad=True!")

    if args.lr_scheduler:
        raise ValueError("LR scheduler is not supported yet since resuming from checkpoint is not implemented yet. Please disable the lr_scheduler argument.")

    train_dataloader, val_dataloader = build_dataset(args,m2p2_rgb_only=True) #PARV_DEBUG using only M2P2 RGB
    print("Train dataset size: ", len(train_dataloader.dataset))
    print("Val dataset size: ", len(val_dataloader.dataset))
    val_soft_positives_per_query = getattr(val_dataloader.dataset, 'soft_positives', None)
    train_soft_positives_per_query = getattr(train_dataloader.dataset, 'soft_positives', None)
    train_pos_cosine_loss = 0.0
    train_neg_cosine_loss = 0.0
    val_pos_cosine_loss = 0.0
    val_neg_cosine_loss = 0.0

    for epoch in range(start_epoch, args.epochs+1):

        start_time = time.time()

        with torch.no_grad() if ((args.dry_run or epoch==0) and not args.debug) else nullcontext():
            train_individual_losses, train_pos_cosine_loss, train_neg_cosine_loss = dataloader_loop(args,save_path, teacher_model, student_model, train_dataloader, test=False, epoch=epoch, soft_positives_per_query=train_soft_positives_per_query)

        val_individual_losses, val_pos_cosine_loss, val_neg_cosine_loss = dataloader_loop(args, save_path,teacher_model, student_model, val_dataloader, test=True, epoch=epoch, soft_positives_per_query=val_soft_positives_per_query)

        if args.wandb_use:
            log_dict = {
                "epoch": epoch,
            }
            for name, loss in train_individual_losses.items():
                log_dict[f"train/{name}"] = loss
            for name, loss in val_individual_losses.items():
                log_dict[f"val/{name}"] = loss
            if val_pos_cosine_loss is not None:
                log_dict["avg_val_positive_cosine_loss"] = val_pos_cosine_loss
                log_dict["avg_val_negative_cosine_loss"] = val_neg_cosine_loss
            if train_pos_cosine_loss is not None:
                log_dict["avg_train_positive_cosine_loss"] = train_pos_cosine_loss
                log_dict["avg_train_negative_cosine_loss"] = train_neg_cosine_loss
            if args.lr_scheduler:
                log_dict["lr"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict)
        
        print(f"[Epoch {epoch}] Train losses: {train_individual_losses}, Val losses: {val_individual_losses}")

        if not args.dry_run and epoch!=0:
            save_dict = {
                'epoch': epoch,
                'student_model_type': args.model_name,
                'student_model_state_dict': student_model.return_model_dict_for_saving(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            for name, loss in train_individual_losses.items():
                save_dict[f"train_{name}"] = loss
            for name, loss in val_individual_losses.items():
                save_dict[f"val_{name}"] = loss

            torch.save(save_dict, os.path.join(save_path, "model" + str(epoch) + '.pth'))

        print("Epoch {} of {} took {:.3f}s\n".format(epoch, args.epochs, time.time() - start_time))


parser = argparse.ArgumentParser(description='Fine-tuning DINOv2 on ImageNet')
parser.add_argument('--dataset', type=str, nargs='+',
                    help='List of datasets to use in training and eval',action=StoreWithFlag)
parser.add_argument('--eval_dataset', default=[],type=str, nargs='+',
                    help='List of datasets to use in training and eval',action=StoreWithFlag)
parser.add_argument('--teacher_modality', default='rgb', type=str, help='modality which will be frozen unless "unfreeze teacher" is true',action=StoreWithFlag)
parser.add_argument('--student_modality', default='thr', type=str, help='modality for which encoder has to be trained',action=StoreWithFlag)
parser.add_argument('--unfreeze_teacher',nargs=0,default=False, help='modality for which encoder has to be trained',action=StoreWithFlag)
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training',action=StoreWithFlag)
parser.add_argument('--eval_batch_size', default=256, type=int, help='Batch size for training',action=StoreWithFlag)
parser.add_argument('--train_num_workers', type=int, default=16, help='Number of workers for training dataloader',action=StoreWithFlag)
parser.add_argument('--eval_num_workers', type=int, default=16, help='Number of workers for evaluation dataloader',action=StoreWithFlag)
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train for',action=StoreWithFlag)
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate',action=StoreWithFlag)
parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight decay',action=StoreWithFlag)
parser.add_argument('--save_path', default='./checkpoints', type=str, help='Path to save the checkpoints',action=StoreWithFlag)
parser.add_argument('--resume', nargs=0,default=False, help='Resume training from a checkpoint',action=StoreWithFlag)
parser.add_argument('--resume_epoch_num', default=0, type=int, help='Epoch number to resume training from',action=StoreWithFlag)
parser.add_argument('--loss_type', default="ce", type=str, help='Loss type: mse or similarity',action=StoreWithFlag)
parser.add_argument('--loss_file', default="loss_config.yaml", type=str, help='Loss type: mse or similarity',action=StoreWithFlag)
parser.add_argument('--wandb_use',nargs=0,default=True, help='Use wandb for logging',action=StoreWithFlag)
parser.add_argument('--model_name', default='dinov2_vitb14', type=str, help='Name of the encoder model',action=StoreWithFlag)
parser.add_argument('--viz_debug', nargs=0,default=False, help='Save train and val images for debugging',action=StoreWithFlag)
parser.add_argument('--lr_scheduler', nargs=0,default=False, help='Save train and val images for debugging',action=StoreWithFlag)
parser.add_argument('--augment', nargs=0,default=False, help='Add augmentation in training',action=StoreWithFlag)
parser.add_argument('--train_easy', nargs=0,default=False, help='Easy traing split in training',action=StoreWithFlag)
parser.add_argument('--wandb_name', default="",type=str, help='Name to append to wandb run',action=StoreWithFlag)
parser.add_argument('--un_frozen_layer_index', type=int, nargs='+', default=[-1],
                    help='List of layer indices to unfreeze',action=StoreWithFlag)
parser.add_argument('--train', default=True,nargs=0, help='Mode to build datasets and dataloaders',action=StoreWithFlag)
parser.add_argument('--use_odom', default=False,nargs=0, help='Mode to build datasets and dataloaders',action=StoreWithFlag)
parser.add_argument('--rescale_during_crop', default=False,nargs=0, help='Rescale images during cropping',action=StoreWithFlag)
parser.add_argument('--vpr_test', default=False,nargs=0, help='Rescale images during cropping',action=StoreWithFlag)
parser.add_argument('--no_shuffle', nargs=0,default=False, help='Rescale images during cropping',action=StoreWithFlag)
parser.add_argument('--no_crop_images', default=False, nargs=0,help='Rescale images during cropping',action=StoreWithFlag)
parser.add_argument('--subsample', default=10, type=int,help='Rescale images during cropping',action=StoreWithFlag)
parser.add_argument('--viz_attention_pca',nargs=0,default=False, help='Visualize attention map and PCA',action=StoreWithFlag)
parser.add_argument('--no_intra_dataset_batch', nargs=0,default=False, help='Visualize attention map and PCA',action=StoreWithFlag)
parser.add_argument('--dry_run', nargs=0,default=False, help='No training, but calculates loss and saves images',action=StoreWithFlag)
parser.add_argument('--unfreeze_patch_embed', nargs=0,default=False, help='Unfreeze the patch embedding layer of the student model',action=StoreWithFlag)
parser.add_argument('--not_unfreeze_final_norm', nargs=0,default=False, help='Unfreeze the final normalization layer of the student model',action=StoreWithFlag)
parser.add_argument('--thermal_aug_list', default=["blur"],type=str, nargs='+', choices=["crop", "rotate", "brightness", "contrast", "clahe", "blur"],
                    help='List of datasets to use in training and eval',action=StoreWithFlag)
parser.add_argument('--proj_head', default="",type=str, choices=["", "linear"],
                    help='List of datasets to use in training and eval',action=StoreWithFlag)
parser.add_argument('--wandb_id', default="",type=str, help='Placeholder for wandb id. This is not used in CLI',action=StoreWithFlag)
parser.add_argument('--cart_split', default='vpr',type=str, help='Task to run, currently only vpr is supported')
parser.add_argument('--debug', nargs=0,default=False, help='Rescale images during cropping',action=StoreWithFlag)
parser.add_argument('--subsample_val', default=10, type=int, help='Batch size for training',action=StoreWithFlag)
parser.add_argument('--unfreeze_cls_token', nargs=0,default=False, help='Unfreeze the patch embedding layer of the student model',action=StoreWithFlag)
parser.add_argument('--unfreeze_pos_embed', nargs=0,default=False, help='Unfreeze the patch embedding layer of the student model',action=StoreWithFlag)
parser.add_argument('--unfreeze_register_tokens', nargs=0,default=False, help='Unfreeze the patch embedding layer of the student model',action=StoreWithFlag)
parser.add_argument('--optimizer', default='sgd', type=str, help='modality which will be frozen unless "unfreeze teacher" is true',action=StoreWithFlag)
parser.add_argument('--sampling_weight', default='equal', type=str, help='Sampling weight for the dataset',action=StoreWithFlag)
parser.add_argument('--sampling_temperature', default=1., type=float, help='Sampling temperature for the dataset',action=StoreWithFlag)
if __name__ == "__main__":
    train()
