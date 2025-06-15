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

global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def save_viz_debug_images(images_dict, index, epoch, batch_num, split, save_root):
    """
    Save RGB and thermal images to disk for visualization/debugging.
    """
    for modality, tensor in images_dict.items():
        # Shape: (B, C, H, W)
        save_dir = os.path.join(save_root, f"epoch_{epoch:03d}", f"iter_{batch_num:04d}", split, modality)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(tensor.shape[0]):
            filename = f"{index[i] if index is not None else i}.png"
            path = os.path.join(save_dir, filename)
            save_image(tensor[i], path)

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
    width = images.shape[-2]
    height = images.shape[-1]
    new_width = (width // patch_size) * patch_size
    new_height = (height // patch_size) * patch_size
    images = torch_func.interpolate(images, size=(new_width, new_height), mode='bilinear', align_corners=False)
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
parser.add_argument('--eval_num_workers', type=int, default=0)

parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train for')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--save_path', default='./checkpoints', type=str, help='Path to save the checkpoints')
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--resume_epoch_num', default=0, type=int, help='Epoch number to resume training from')
parser.add_argument('--loss_type', default="ce", type=str, help='Loss type: mse or similarity')
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

args = parser.parse_args()
print(args)

#Initialize wandb

dataset_name = "_".join(args.dataset)
name = args.model_name + "_" + dataset_name + "_" + args.student_modality + "_distill"
if args.wandb_name != "":
    name += "_" + args.wandb_name
if args.wandb_use:
    wandb.init(project="multiloc", name=name)

# Load models

teacher_model,teacher_patch_size = init_model(args,modality=args.teacher_modality,un_frozen_layer_index=[])
student_model,student_patch_size = init_model(args,modality=args.student_modality,un_frozen_layer_index=args.un_frozen_layer_index)


if args.unfreeze_teacher:
    teacher_model.train()
else:
    teacher_model.eval()
    # Freeze the RGB model, so that only the thermal model is trained
    for param in teacher_model.parameters():
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
    params = chain(*[student_model.model.blocks[i].parameters() for i in unfrozen_layers])

    # Create the optimizer
    optimizer = optim.SGD(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # print("Params in optimizer:")
    # for group in optimizer.param_groups:
    #     for p in group['params']:
    #         print(p.shape, p.requires_grad, p.grad is not None)
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

def contrastive_loss(teacher_embed, student_embed, temperature=0.07, positive_mask=None):
    teacher_embed =torch_func.normalize(teacher_embed, dim=-1)
    student_embed =torch_func.normalize(student_embed, dim=-1)

    logits = torch.matmul(teacher_embed, student_embed.T) / temperature

    logits = logits - logits.max(dim=1, keepdim=True)[0]  # numerical stability
    exp_logits = torch.exp(logits)
    log_probs = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    if positive_mask is None:
        positive_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    # if positive_mask.sum() > positive_mask.shape[0]:
    #     import pdb; pdb.set_trace()

    loss = -(log_probs * positive_mask).sum() / (positive_mask.sum() + 1e-8)
    return loss


def cosine_loss(student_embed, teacher_embed,return_mean=True):
    # with torch.no_grad():
    student_embed = torch_func.normalize(student_embed, dim=-1)
    teacher_embed = torch_func.normalize(teacher_embed, dim=-1)
    cos_sim = torch_func.cosine_similarity(student_embed, teacher_embed, dim=-1)  # shape: [B]
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

def inference(args,teacher_model,student_model, teacher_modality,student_modality,batch_item,test=False,soft_positives_per_query=None):
    with (torch.inference_mode() if test else nullcontext()):
        images, _ = batch_item["item"]
        batch_indices = batch_item["batch_id"].tolist()
        # print("Batch indices: ", batch_indices)
        # if soft_positives_per_query is not None:
        #     pos_masks = generate_positive_masks(soft_positives_per_query, batch_indices, device='cuda')
        # else:
        #     pos_masks = None
        pos_masks = torch.eye(len(batch_indices), dtype=torch.bool, device=global_device)
        # if test:
        #     import pdb; pdb.set_trace()
        img_teacher = transform_images(teacher_modality,images[teacher_modality],teacher_patch_size).to(global_device)
        img_student = transform_images(student_modality,images[student_modality],student_patch_size).to(global_device)

        if args.unfreeze_teacher:
            teacher_output = teacher_model.forward_train(img_teacher,return_local_features=False)
        else:
            with torch.inference_mode():
                teacher_output = teacher_model.forward_train(img_teacher,return_local_features=False).detach()
        
        student_output = student_model.forward_train(img_student,return_local_features=False)
        # import pdb; pdb.set_trace()
        if args.loss_type == "mse":
            loss = loss_fn_mse(student_output, teacher_output)
        elif args.loss_type == "ce":
            contrastive_loss_output = contrastive_loss(teacher_output, student_output,positive_mask=pos_masks)
            cosine_loss_output = cosine_loss(student_output, teacher_output)
            loss = contrastive_loss_output + cosine_loss_output
            contrastive_item = contrastive_loss_output.item()
            cosine_loss_item = cosine_loss_output.item()
        else:   
            raise ValueError("Invalid loss type. Please choose 'mse' or 'ce'.")
        if not test:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        teacher_output_norm_mean, teacher_output_norm_std = torch.mean(torch.norm(teacher_output,dim=-1)), torch.std(torch.norm(teacher_output,dim=-1))
        student_output_norm_mean, student_output_norm_std = torch.mean(torch.norm(student_output,dim=-1)), torch.std(torch.norm(student_output,dim=-1))

        mode = "val" if test else "train"
        if args.wandb_use:
            wandb.log({
                f"{mode}/teacher_output_norm_mean": teacher_output_norm_mean.item(),
                f"{mode}/teacher_output_norm_std": teacher_output_norm_std.item(),
                f"{mode}/student_output_norm_mean": student_output_norm_mean.item(),
                f"{mode}/student_output_norm_std": student_output_norm_std.item(),
            })
        if soft_positives_per_query is None:
            del img_teacher, img_student, teacher_output, student_output, loss, contrastive_loss_output, cosine_loss_output
            return contrastive_item,cosine_loss_item,None,None
        else:
            positive_loss, negative_loss = cosine_positive_negative_loss(student_output, teacher_output, 
                                                                        positive_mask=pos_masks)
            del img_teacher, img_student, teacher_output, student_output, loss, contrastive_loss_output, cosine_loss_output

            return contrastive_item, cosine_loss_item,positive_loss, negative_loss
    
def train():
    # Load the dataset

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
        save_path = os.path.join(save_path, time.strftime("%Y-%m-%d_%H-%M-%S"))

        # Create the save directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        import yaml
        with open(os.path.join(save_path,'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)
    viz_debug_dir = os.path.join(save_path, "viz_debug") if args.viz_debug else None

    val_soft_positives_per_query = getattr(val_dataloader.dataset, 'soft_positives', None)
    train_soft_positives_per_query = getattr(train_dataloader.dataset, 'soft_positives', None)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_running_contrastive_loss = 0.0
        train_running_cosine_loss = 0.0
        train_running_cosine_loss_positive = 0.0
        train_running_cosine_loss_negative = 0.0

        batch_contrastive_loss = 0.0
        batch_cosine_loss = 0.0
        batch_cosine_loss_positive = 0.0
        batch_cosine_loss_negative = 0.0

        for i, train_item in enumerate(tqdm(train_dataloader)):
            images, _ = train_item["item"]
            train_index = train_item["batch_id"]
            if args.viz_debug:
                save_viz_debug_images(images, train_index, epoch, i, "train", viz_debug_dir)
            
            train_contrastive_loss, train_cosine_loss, train_pos_loss, train_neg_loss = inference(
                args, teacher_model, student_model, args.teacher_modality, args.student_modality,
                train_item, soft_positives_per_query=train_soft_positives_per_query)

            train_running_contrastive_loss += train_contrastive_loss
            train_running_cosine_loss += train_cosine_loss
            if train_pos_loss is not None and train_neg_loss is not None:
                train_running_cosine_loss_positive += train_pos_loss
                train_running_cosine_loss_negative += train_neg_loss

            batch_contrastive_loss += train_contrastive_loss
            batch_cosine_loss += train_cosine_loss
            if train_pos_loss is not None and train_neg_loss is not None:
                batch_cosine_loss_positive += train_pos_loss
                batch_cosine_loss_negative += train_neg_loss

            if (i + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Running Losses: Cosine={batch_cosine_loss/10:.4f}, Contrastive={batch_contrastive_loss/10:.4f}")
                batch_contrastive_loss = 0.0
                batch_cosine_loss = 0.0
                batch_cosine_loss_positive = 0.0
                batch_cosine_loss_negative = 0.0

            torch.cuda.empty_cache()

        # ==== Run validation over the full val_dataloader ====
        val_total_contrastive = 0.0
        val_total_cosine = 0.0
        val_total_pos = 0.0
        val_total_neg = 0.0
        val_batches = 0

        for j, val_item in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")):
            images, _ = val_item["item"]
            val_index = val_item["batch_id"]
            if args.viz_debug:
                save_viz_debug_images(images, val_index, epoch, j, "val", viz_debug_dir)

            val_contrastive_loss, val_cosine_loss, val_pos_loss, val_neg_loss = inference(
                args, teacher_model, student_model,
                args.teacher_modality, args.student_modality,
                val_item, test=True, soft_positives_per_query=val_soft_positives_per_query)
            val_total_contrastive += val_contrastive_loss
            val_total_cosine += val_cosine_loss
            if val_pos_loss is not None:
                val_total_pos += val_pos_loss
                val_total_neg += val_neg_loss
            val_batches += 1

        val_avg_contrastive = val_total_contrastive / val_batches
        val_avg_cosine = val_total_cosine / val_batches
        val_avg_pos = val_total_pos / val_batches if val_total_pos > 0 else None
        val_avg_neg = val_total_neg / val_batches if val_total_neg > 0 else None
                
        if args.wandb_use:
            log_dict = {
                        "epoch": epoch,
                "avg_train_contrastive_loss": train_running_contrastive_loss / len(train_dataloader),
                "avg_train_cosine_loss": train_running_cosine_loss / len(train_dataloader),
                "avg_val_contrastive_loss": val_avg_contrastive,
                "avg_val_cosine_loss": val_avg_cosine,
            }
            if val_avg_pos is not None:
                log_dict["avg_val_positive_cosine_loss"] = val_avg_pos
                log_dict["avg_val_negative_cosine_loss"] = val_avg_neg
            if train_pos_loss is not None:
                log_dict["avg_train_positive_cosine_loss"] = train_running_cosine_loss_positive / len(train_dataloader)
                log_dict["avg_train_negative_cosine_loss"] = train_running_cosine_loss_negative / len(train_dataloader)
                if args.lr_scheduler:
                    log_dict["lr"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict)
        
        print(f"[Epoch {epoch+1}] Avg Train Cosine: {train_running_cosine_loss/len(train_dataloader):.4f}, Avg Val Cosine: {val_avg_cosine:.4f}")

        torch.save({
            'epoch': epoch,
            'student_model_type': args.model_name,
            'student_model_state_dict': student_model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "train_contrastive_loss": train_running_contrastive_loss,
            "train_cosine_loss": train_running_cosine_loss,
            "val_contrastive_loss": val_avg_contrastive,
            "val_cosine_loss": val_avg_cosine
        }, os.path.join(save_path, "model" + str(epoch) + '.pth'))

        print("Epoch {} of {} took {:.3f}s\n".format(epoch+1, args.epochs, time.time() - start_time))

if __name__ == "__main__":
    train()
