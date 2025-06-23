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

from losses.loss_combiner import LossManager
viz_freq = 10

global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def viz_attention_pca(args,teacher_attention_maps,student_attention_maps,teacher_prediction,student_prediction, save_dir, epoch, batch_index,image_shape):
    """
    Visualize attention maps using PCA and save them to disk.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    save_dir = os.path.join(save_dir)
    if not os.path.exists(os.path.join(save_dir, "attention_map")):
        os.makedirs(os.path.join(save_dir, "attention_map"), exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, "pca")):
        os.makedirs(os.path.join(save_dir, "pca"), exist_ok=True)

    
    

    for i, (teacher_map,student_map) in enumerate(zip(teacher_attention_maps, student_attention_maps)):
        if i % viz_freq != 0:
            continue
        # attn_map shape: (Heads, N, N)
        # import pdb; pdb.set_trace()
        patch_tokens = {
            "teacher": {},
            "student": {},
        }
        patch_tokens["teacher"]["pred"] = teacher_prediction[0][i]
        patch_tokens["student"]["pred"] = student_prediction[0][i]

        patch_tokens["teacher"]["dim_size"] = teacher_prediction[0][i].shape[0]
        patch_tokens["student"]["dim_size"] = student_prediction[0][i].shape[0]

        patch_tokens_shape = patch_tokens["student"]["pred"].shape[1:]
        

        # for key in ["teacher", "student"]:
        #     dim_size = patch_tokens[key]["pred"].shape[0]
        #     patch_tokens_shape = patch_tokens[key]["pred"].shape[1:]  # (H, W)
        #     attention_map = teacher_map if key == "teacher" else student_map
        #     num_heads = attention_map.shape[0]  # Number of attention heads
            
        #     cls_attn = attention_map[:, 0, 1:]  # (Heads, Patches)
        #     cls_attn = cls_attn.reshape(num_heads, *(patch_tokens_shape))  # Assume 14x14 patches for ViT-B/14

        #     upsampled_attn = [cv2.resize(h.cpu().numpy(), (image_shape[1],image_shape[0]), interpolation=cv2.INTER_CUBIC) for h in cls_attn]

        #     # Plot
        #     fig, axs = plt.subplots(1, num_heads, figsize=(20, 5))
        #     for j, attention_map in enumerate(upsampled_attn):
        #         axs[j].imshow(attention_map, cmap="inferno")
        #         axs[j].axis("off")
        #         axs[j].set_title(f"Head {j}")
        #     plt.tight_layout()
        #     if not os.path.exists(os.path.join(save_dir, "attention_map",key)):
        #         os.makedirs(os.path.join(save_dir, "attention_map",key), exist_ok=True)
        #     plt.savefig(os.path.join(save_dir, "attention_map",key, f"head_{batch_index[i]:06d}.png"))
        #     plt.close(fig)


        reshaped_teacher_patch_tokens = patch_tokens["teacher"]["pred"].permute(1,2,0).reshape(-1,patch_tokens["teacher"]["dim_size"]).cpu().detach().numpy()  # Reshape to (-1, D)
        reshaped_student_patch_tokens = patch_tokens["student"]["pred"].permute(1,2,0).reshape(-1,patch_tokens["student"]["dim_size"]).cpu().detach().numpy()  # Reshape to (-1, D)


        if args.viz_attention_pca == "combined_pca":
            # import pdb;pdb.set_trace()
            foreground_pca = PCA(n_components=1)
            # Apply PCA to reduce to 3D
            pca_teacher = foreground_pca.fit_transform(reshaped_teacher_patch_tokens)  # Shape: (196, 1)
            pca_student = foreground_pca.fit_transform(reshaped_student_patch_tokens)  # Shape: (196, 1)

            foreground_mask_student = np.arange(reshaped_student_patch_tokens.shape[0])  # Use all patches for student
            foreground_mask_teacher = np.arange(reshaped_teacher_patch_tokens.shape[0])  # Use all patches for teacher

            # Concatenate teacher and student patch tokens
            concat_patch_tokens = np.concatenate((reshaped_teacher_patch_tokens[foreground_mask_teacher], reshaped_student_patch_tokens[foreground_mask_student]), axis=0)  # Shape: (2*N, D), N: Number of patches in the image
            pca = PCA(n_components=3)
            concat_pca_feats = pca.fit_transform(concat_patch_tokens)  # Shape: (196, 3)

            # Normalize to 0-1
            concat_pca_feats -= concat_pca_feats.min(0)
            concat_pca_feats /= concat_pca_feats.max(0)

            teacher_pca_img = np.zeros((patch_tokens_shape[0], patch_tokens_shape[1], 3), dtype=np.float32)
            student_pca_img = np.zeros((patch_tokens_shape[0], patch_tokens_shape[1], 3), dtype=np.float32)

            # Fill the PCA features into the image
            for idx, teacher_idx in enumerate(foreground_mask_teacher):
                teacher_pca_img[teacher_idx // patch_tokens_shape[1], teacher_idx % patch_tokens_shape[1]] = concat_pca_feats[idx]
            for idx, student_idx in enumerate(foreground_mask_student):
                student_pca_img[student_idx // patch_tokens_shape[1], student_idx % patch_tokens_shape[1]] = concat_pca_feats[len(foreground_mask_teacher) + idx]
        elif args.viz_attention_pca == "teacher_pca":
            pca = PCA(n_components=3).fit(reshaped_teacher_patch_tokens)
            teacher_pca_img = pca.transform(reshaped_teacher_patch_tokens)  # Shape: (N, 3)
            student_pca_img = pca.transform(reshaped_student_patch_tokens)  # Shape: (N, 3)

            teacher_pca_img -= teacher_pca_img.min(0)
            teacher_pca_img /= teacher_pca_img.max(0)
            student_pca_img -= student_pca_img.min(0)
            student_pca_img /= student_pca_img.max(0)

            teacher_pca_img = teacher_pca_img.reshape(patch_tokens_shape[0], patch_tokens_shape[1], 3)
            student_pca_img = student_pca_img.reshape(patch_tokens_shape[0], patch_tokens_shape[1], 3)

        else:
            raise ValueError("Invalid viz_attention_pca option. Please choose 'combined_pca' or 'teacher_pca'.")


        teacher_pca_img = cv2.resize(teacher_pca_img, (image_shape[1],image_shape[0]), interpolation=cv2.INTER_NEAREST)
        student_pca_img = cv2.resize(student_pca_img, (image_shape[1],image_shape[0]), interpolation=cv2.INTER_NEAREST)

        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        axs[0].imshow(teacher_pca_img)
        axs[0].axis("off")
        axs[0].set_title(f"Teacher PCA of patch tokens")
        axs[1].imshow(student_pca_img)
        axs[1].axis("off")
        axs[1].set_title(f"Student PCA of patch tokens")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pca",f"{batch_index[i]:06d}.png"))
        plt.close(fig)

def init_model(args,modality,un_frozen_layer_index):
    if args.model_name == "dinov2_vits14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if global_device.type == 'cuda':
            model = model.cuda()
        patch_size = 14
    elif args.model_name == "dinov2_vitb14":
        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
        patch_size = 14
        # import pdb; pdb.set_trace()
        if un_frozen_layer_index == [-1]:
            unfreeze_layers = list(range(0, 12))
        else:
            unfreeze_layers = un_frozen_layer_index
        if unfreeze_layers != []:
            if args.unfreeze_patch_embed:
                unfreeze_layers = ["patch_embed"] + unfreeze_layers
            if args.unfreeze_final_norm:
                unfreeze_layers.append("norm")

        model = MMDistillDinov2(args.model_name, modality=modality,un_frozen_layer_index= unfreeze_layers)
    elif args.model_name == "dinov2_vitb16":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb16').cuda()
        patch_size = 16
    else:
        raise ValueError("Invalid model name")
    return model, patch_size

def transform_images(modality,images,patch_size=14):
    # Resize the images to the required input size
    if modality == "rgb":
        #normalise the image using VIT mean 
        images = transforms.functional.normalize(images, mean=[0.481, 0.457, 0.408], std=[0.269, 0.26, 0.275])
    elif modality == "thr":
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
    elif modality == "thr":
        mean = torch.tensor([0.481, 0.457, 0.408], device=images.device).view(-1, 1, 1)
        std = torch.tensor([0.269, 0.260, 0.275], device=images.device).view(-1, 1, 1)

    images = images * std + mean
    return images

parser = argparse.ArgumentParser(description='Fine-tuning DINOv2 on ImageNet')
# parser.add_argument('--dataset_path', default='', type=str, help='Path to the ImageNet dataset')
parser.add_argument('--dataset', type=str, nargs='+',
                    help='List of datasets to use in training and eval')
parser.add_argument('--eval_dataset', default=[],type=str, nargs='+',
                    help='List of datasets to use in training and eval')
parser.add_argument('--teacher_modality', default='rgb', type=str, help='modality which will be frozen unless "unfreeze teacher" is true')
parser.add_argument('--student_modality', default='thr', type=str, help='modality for which encoder has to be trained')
parser.add_argument('--unfreeze_teacher',action="store_true", help='modality for which encoder has to be trained')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--train_num_workers', type=int, default=4)
parser.add_argument('--eval_num_workers', type=int, default=4)
parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train for')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--save_path', default='./checkpoints', type=str, help='Path to save the checkpoints')
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--resume_epoch_num', default=0, type=int, help='Epoch number to resume training from')
parser.add_argument('--loss_type', default="ce", type=str, help='Loss type: mse or similarity')
parser.add_argument('--loss_file', default="loss_config.yaml", type=str, help='Loss type: mse or similarity')
parser.add_argument('--wandb_use',default=False, type=bool, help='Use wandb for logging')
parser.add_argument('--model_name', default='dinov2_vitb14', type=str, help='Name of the encoder model')
parser.add_argument('--viz_debug', action='store_true', help='Save train and val images for debugging')
parser.add_argument('--lr_scheduler', action='store_true', help='Save train and val images for debugging')
parser.add_argument('--augment', action='store_true', help='Add augmentation in training')
parser.add_argument('--train_easy', action='store_true', help='Easy traing split in training')
parser.add_argument('--wandb_name', default="",type=str, help='Name to append to wandb run')
parser.add_argument('--un_frozen_layer_index', type=int, nargs='+', default=[-1],
                    help='List of layer indices to unfreeze')
parser.add_argument('--train', default=True,type=bool, help='Mode to build datasets and dataloaders')
parser.add_argument('--use_odom', default=False,type=bool, help='Mode to build datasets and dataloaders')
parser.add_argument('--rescale_during_crop', default=False,type=bool, help='Rescale images during cropping')
parser.add_argument('--vpr_test', default=False,type=bool, help='Rescale images during cropping')
parser.add_argument('--no_shuffle', action='store_true', help='Rescale images during cropping')
parser.add_argument('--crop_images', default=True, type=bool,help='Rescale images during cropping')
parser.add_argument('--subsample', default=1, type=int,help='Rescale images during cropping')
parser.add_argument('--viz_attention_pca',default="",type=str, help='Visualize attention map and PCA')
parser.add_argument('--intra_dataset_batch', action='store_true', help='Visualize attention map and PCA')
parser.add_argument('--dry_run', action='store_true', help='No training, but calculates loss and saves images')
parser.add_argument('--unfreeze_patch_embed', action='store_true', help='Unfreeze the patch embedding layer of the student model')
parser.add_argument('--unfreeze_final_norm', action='store_true', help='Unfreeze the final normalization layer of the student model')


args = parser.parse_args()
print(args)

#Initialize wandb

dataset_name = "_".join(args.dataset)
wandb_name = args.model_name + "_" + dataset_name + "_" + args.student_modality + "_distill"
if args.wandb_name != "":
    wandb_name += "_" + args.wandb_name

if args.wandb_use:
    wandb.init(project="multiloc", name=wandb_name)

# Load models
# import pdb; pdb.set_trace()
teacher_model,teacher_patch_size = init_model(args,modality=args.teacher_modality,un_frozen_layer_index=[])
student_model,student_patch_size = init_model(args,modality=args.student_modality,un_frozen_layer_index=args.un_frozen_layer_index)


if args.unfreeze_teacher:
    teacher_model.train()
else:
    teacher_model.eval()
    # Freeze the RGB model, so that only the thermal model is trained
    for param in teacher_model.model.parameters():
        param.requires_grad = False


student_model.train()

if args.unfreeze_teacher:
    raise ValueError("Unfreezing teacher model is not supported yet. Please use the default frozen teacher model. Else enable freezing and unfreew layers through optimisation")
    optimizer = optim.SGD(
        chain(student_model.model.blocks[:].parameters(), teacher_model.model.blocks[:].parameters()),
        lr=args.learning_rate,
            weight_decay=args.weight_decay
    )
else:
    from itertools import chain

    unfrozen_layers = student_model.un_frozen_layer_index
    print("Unfrozen layers for optimisation: ", unfrozen_layers)

    # Collect parameters from specified transformer blocks
    # import pdb; pdb.set_trace()
    params = chain(student_model.unfrozen_parameters())


    # Create the optimizer
    optimizer = optim.SGD(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # print("Params in optimizer:")
    # for group in optimizer.param_groups:
    #     for p in group['params']:
    #         print(p.shape, p.requires_grad, p.grad is not None)

for name, param in student_model.model.named_parameters():
    if param.requires_grad and not any(param is p for group in optimizer.param_groups for p in group['params']):
        print(f"{name} is not in the optimizer but requires_grad=True!")

from torch.optim.lr_scheduler import LambdaLR
import math
initial_lr = args.learning_rate
def warmup_cosine_lr_lambda(current_epoch):
    if current_epoch < 10:
        print("Warmup for the first 10 epochs, linear increase, currently lr is: ", initial_lr * (current_epoch / 10.0))
        return initial_lr * (current_epoch / 10.0)  # linear warmup for the first 10 epochs
    else:
        # cosine decay from base_lr to base_lr / 1000
        progress = (current_epoch - 10) / max(1, args.epochs - 10)
        return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - 1/1000) + (1/1000)

if args.lr_scheduler:
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr_lambda)
def loss_fn_mse(outputs, targets):
    return torch_func.mse_loss(outputs, targets)

def loss_fn_global_contrastive(teacher_embed, student_embed, temperature=0.07, positive_mask=None):
    teacher_embed_global =torch_func.normalize(teacher_embed[1].detach(), dim=-1)
    student_embed_global =torch_func.normalize(student_embed[1], dim=-1)

    logits = torch.matmul(teacher_embed_global, student_embed_global.T) / temperature

    logits = logits - logits.max(dim=1, keepdim=True)[0]  # numerical stability
    exp_logits = torch.exp(logits)
    log_probs = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    if positive_mask is None:
        positive_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    # if positive_mask.sum() > positive_mask.shape[0]:
    #     import pdb; pdb.set_trace()

    loss = -(log_probs * positive_mask).sum() / (positive_mask.sum() + 1e-8)
    return loss


def loss_fn_global_cosine(teacher_embed,student_embed,return_mean=True):
    # with torch.no_grad():
    teacher_embed_global =torch_func.normalize(teacher_embed[1].detach(), dim=-1)
    student_embed_global =torch_func.normalize(student_embed[1], dim=-1)
    cos_sim = torch_func.cosine_similarity(student_embed_global, teacher_embed_global, dim=-1)  # shape: [B]
    loss = 1 - cos_sim  # shape: [B]
    return loss.mean()

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

def inference(args,teacher_model,student_model, teacher_modality,student_modality,images,batch_indices,test=False,soft_positives_per_query=None):
    with (torch.inference_mode() if test else nullcontext()):
        if soft_positives_per_query is not None:
            pos_masks = generate_positive_masks(soft_positives_per_query, batch_indices, device='cuda')
        else:
            pos_masks = None

        pos_masks = torch.eye(len(batch_indices), dtype=torch.bool, device=global_device)
        if args.unfreeze_teacher:
            teacher_output = teacher_model.forward_train(images[teacher_modality],return_local_features=True)
        else:
            with torch.inference_mode():
                teacher_output = teacher_model.forward_train(images[teacher_modality],return_local_features=True)
        
        student_output = student_model.forward_train(images[student_modality],return_local_features=True, preprocess=True)
        assert isinstance(student_output, tuple) and len(student_output) ==2, "Student model should return a tuple of (local_features,global_features)"
        assert isinstance(teacher_output, tuple) and len(teacher_output) ==2, "Teacher model should return a tuple of (local_features,global_features)"
        # if args.loss_type == "mse":
        #     loss = loss_fn_mse(student_output, teacher_output)
        # elif args.loss_type == "ce":
        #     contrastive_loss_output = loss_fn_global_contrastive(teacher_output, student_output,positive_mask=pos_masks)
        #     cosine_loss_output = loss_fn_global_cosine(teacher_output,student_output)
        #     loss = contrastive_loss_output + cosine_loss_output
        #     contrastive_item = contrastive_loss_output.item()
        #     cosine_loss_item = cosine_loss_output.item()
        # else:   
        #     raise ValueError("Invalid loss type. Please choose 'mse' or 'ce'.")
        
        total_loss, individual_losses = args.loss_object.compute(student_output, teacher_output)
        if not test and not args.dry_run:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        teacher_output_norm_mean, teacher_output_norm_std = torch.mean(torch.norm(teacher_output[1],dim=-1)), torch.std(torch.norm(teacher_output[1],dim=-1))
        student_output_norm_mean, student_output_norm_std = torch.mean(torch.norm(student_output[1],dim=-1)), torch.std(torch.norm(student_output[1],dim=-1))

        mode = "val" if test else "train"
        if args.wandb_use:
            wandb.log({
                f"{mode}/teacher_output_norm_mean": teacher_output_norm_mean.item(),
                f"{mode}/teacher_output_norm_std": teacher_output_norm_std.item(),
                f"{mode}/student_output_norm_mean": student_output_norm_mean.item(),
                f"{mode}/student_output_norm_std": student_output_norm_std.item(),
            })
        if soft_positives_per_query is None:
            return teacher_output,student_output,individual_losses,None,None
        else:
            positive_loss, negative_loss = cosine_positive_negative_loss(student_output, teacher_output, 
                                                                        positive_mask=pos_masks)
            return teacher_output,student_output,individual_losses,positive_loss, negative_loss


def dataloader_loop(args,save_path, teacher_model, student_model,dataloader, test=False, epoch=0, soft_positives_per_query=None):
    """
    Loop through the dataloader and perform inference or training.
    """

    total_individual_losses = {}
    for name in args.loss_object.losses.keys():
        total_individual_losses[name] = 0.0

    total_positive_cosine_loss = 0.0
    total_negative_loss = 0.0
    num_batches = len(dataloader)
    split = "val" if test else "train"



    for batch_number, batch in enumerate(tqdm(dataloader)):
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
            

        teacher_output, student_output,individual_losses, positive_loss, negative_loss = inference(
            args, teacher_model, student_model,
            args.teacher_modality, args.student_modality,
            images=images, batch_indices=index.tolist(),test=test,
            soft_positives_per_query=soft_positives_per_query)
        
        if args.viz_attention_pca:
            teacher_hook.remove()
            student_hook.remove()
            viz_attention_pca(args,teacher_attention_maps[0],student_attention_maps[0], teacher_output,student_output, viz_debug_dir, epoch, index, images[args.teacher_modality].shape[-2:])


        for name, loss in individual_losses.items():
            if name not in total_individual_losses:
                raise ValueError(f"Loss name {name} not found in total_individual_losses dictionary. Available keys: {list(total_individual_losses.keys())}")
            total_individual_losses[name] += loss
        if positive_loss is not None and negative_loss is not None:
            total_positive_cosine_loss += positive_loss
            total_negative_loss += negative_loss

        if (batch_number + 1) % 10 == 0:
            debug_str = f"[Epoch {epoch+1}, Batch {batch_number+1}] "
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
    
def train_better():
    # Load the dataset

    args.loss_object = LossManager(args.loss_file)

    train_dataloader, val_dataloader = build_dataset(args)
    print("Train dataset size: ", len(train_dataloader.dataset))
    print("Val dataset size: ", len(val_dataloader.dataset))

    
    # Load the checkpoint if resuming training
    start_epoch = args.resume_epoch_num
    if args.resume:
        save_path = args.save_path
        checkpoint = torch.load(os.path.join(args.save_path, "model" +str(start_epoch) + '.pth'))
        if args.unfreeze_teacher:
            teacher_model.model.load_state_dict(checkpoint['teacher_model_state_dict'])
        student_model.model.load_state_dict(checkpoint['student_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if args.lr_scheduler:
            for e in range(start_epoch):
                scheduler.step()
        print("Resuming training from epoch {}".format(start_epoch))
    else:
        save_path = args.save_path
        save_path = os.path.join(save_path,dataset_name, f"{args.teacher_modality}_{args.student_modality}")
        #append currnet date and time to the save path
        save_path = os.path.join(save_path, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{wandb_name}')

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

    val_soft_positives_per_query = getattr(val_dataloader.dataset, 'soft_positives', None)
    train_soft_positives_per_query = getattr(train_dataloader.dataset, 'soft_positives', None)
    train_pos_cosine_loss = 0.0
    train_neg_cosine_loss = 0.0
    val_pos_cosine_loss = 0.0
    val_neg_cosine_loss = 0.0

    for epoch in range(start_epoch, args.epochs):

        start_time = time.time()
        with torch.no_grad() if args.dry_run else nullcontext():
            train_individual_losses, train_pos_cosine_loss, train_neg_cosine_loss = dataloader_loop(args,save_path, teacher_model, student_model, train_dataloader, test=False, epoch=epoch, soft_positives_per_query=train_soft_positives_per_query)
        val_individual_losses, val_pos_cosine_loss, val_neg_cosine_loss = dataloader_loop(args, save_path,teacher_model, student_model, val_dataloader, test=True, epoch=epoch, soft_positives_per_query=val_soft_positives_per_query)

        if args.wandb_use:
            log_dict = {
                "epoch": epoch,
            }
            for name, loss in train_individual_losses.items():
                log_dict[f"avg_train_{name}"] = loss
            for name, loss in val_individual_losses.items():
                log_dict[f"avg_val_{name}"] = loss
            if val_pos_cosine_loss is not None:
                log_dict["avg_val_positive_cosine_loss"] = val_pos_cosine_loss
                log_dict["avg_val_negative_cosine_loss"] = val_neg_cosine_loss
            if train_pos_cosine_loss is not None:
                log_dict["avg_train_positive_cosine_loss"] = train_pos_cosine_loss
                log_dict["avg_train_negative_cosine_loss"] = train_neg_cosine_loss
            if args.lr_scheduler:
                log_dict["lr"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict)
        
        print(f"[Epoch {epoch+1}] Train losses: {train_individual_losses}, Val losses: {val_individual_losses}")

        save_dict = {
            'epoch': epoch,
            'student_model_type': args.model_name,
            'student_model_state_dict': student_model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        for name, loss in train_individual_losses.items():
            save_dict[f"train_{name}"] = loss
        for name, loss in val_individual_losses.items():
            save_dict[f"val_{name}"] = loss

        torch.save(save_dict, os.path.join(save_path, "model" + str(epoch) + '.pth'))

        print("Epoch {} of {} took {:.3f}s\n".format(epoch+1, args.epochs, time.time() - start_time))


if __name__ == "__main__":
    train_better()
