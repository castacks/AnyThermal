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

from custom_datasets.ms2_dataset import *
from custom_datasets.cart_dataset import *
# from datasets.wisard_dataset import Wisard_Dataset
# from datasets.cart_dataset import CART_dataset
# from datasets.tartanair_dataset import *
import gc
from utilities import DinoV2ExtractFeatures
from torchvision.utils import save_image

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

def init_model(model_name):
    if model_name == "dinov2_vits14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        patch_size = 14
    elif model_name == "dinov2_vitb14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
        patch_size = 14
    elif model_name == "dinov2_vitb16":
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
parser.add_argument('--dataset', default='ms2', type=str, help='dataset name')
parser.add_argument('--teacher_modality', default='rgb', type=str, help='modality which will be frozen unless "unfreeze teacher" is true')
parser.add_argument('--student_modality', default='thr', type=str, help='modality for which encoder has to be trained')
parser.add_argument('--unfreeze_teacher',action="store_true", help='modality for which encoder has to be trained')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
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

args = parser.parse_args()
print(args)

#Initialize wandb
name = args.model_name + "_" + args.dataset + "_" + args.student_modality + "_distill"
if args.wandb_name != "":
    name += "_" + args.wandb_name
if args.wandb_use:
    wandb.init(project="multiloc", name=name)

# Load models

teacher_model,teacher_patch_size = init_model(args.model_name)
student_model,student_patch_size = init_model(args.model_name)
teacher_model = teacher_model.cuda()
student_model = student_model.cuda()

if args.unfreeze_teacher:
    teacher_model.train()
else:
    teacher_model.eval()
    # Freeze the RGB model, so that only the thermal model is trained
    for param in teacher_model.parameters():
        param.requires_grad = False


student_model.train()

if args.unfreeze_teacher:
    optimizer = optim.SGD(
        chain(student_model.blocks[:].parameters(), teacher_model.blocks[:].parameters()),
        lr=args.learning_rate,
            weight_decay=args.weight_decay
    )
else:
    optimizer = optim.SGD(student_model.blocks[:].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
if args.lr_scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)

def loss_fn_mse(outputs, targets):
    return torch_func.mse_loss(outputs, targets)

def contrastive_loss(teacher_embed, student_embed, temperature=0.07):
    # Normalize embeddings
    teacher_embed = torch_func.normalize(teacher_embed, dim=-1)
    student_embed = torch_func.normalize(student_embed, dim=-1)
    
    # Compute cosine similarity between embeddings
    similarity = torch.matmul(teacher_embed, student_embed.T) / temperature
    
    # Generate mask to exclude similarity of same embeddings
    mask = torch.eye(similarity.size(0), dtype=torch.bool).cuda()
    
    # Compute logits for positive and negative pairs
    positive_pairs = torch.diag(similarity)  # similarity of same embeddings
    negative_pairs = similarity[~mask].view(similarity.size(0), -1)  # similarity of different embeddings
    # Compute log probabilities
    logits = torch.cat([positive_pairs.unsqueeze(1), negative_pairs], dim=1)
    # Apply log-softmax to logits
    log_probs = torch_func.log_softmax(logits, dim=1)
    # Negative log probability of the true pairs (positive pairs)
    loss = -log_probs[:, 0].mean()
    
    return loss

def cosine_loss(student_embed, teacher_embed):
    # with torch.no_grad():
    student_embed = torch_func.normalize(student_embed, dim=-1)
    teacher_embed = torch_func.normalize(teacher_embed, dim=-1)
    cos_sim = torch_func.cosine_similarity(student_embed, teacher_embed, dim=-1)  # shape: [B]
    loss = 1 - cos_sim  # shape: [B]
    return loss.mean()

def inference(args,teacher_model,student_model, teacher_modality,student_modality,images,test=False):
    with (torch.inference_mode() if test else nullcontext()):
        img_teacher = transform_images(teacher_modality,images[teacher_modality],teacher_patch_size).to('cuda')
        img_student = transform_images(student_modality,images[student_modality],student_patch_size).to('cuda')

        if args.unfreeze_teacher:
            teacher_output = teacher_model(img_teacher)
        else:
            with torch.inference_mode():
                teacher_output = teacher_model(img_teacher).detach()
        
        student_output = student_model(img_student)
        if args.loss_type == "mse":
            loss = loss_fn_mse(student_output, teacher_output)
        elif args.loss_type == "ce":
            contrastive_loss_output = contrastive_loss(teacher_output, student_output)
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
        
        del img_teacher, img_student, teacher_output, student_output, loss, contrastive_loss_output, cosine_loss_output

        return contrastive_item,cosine_loss_item

def train():
    # Load the dataset
    teacher_modality = args.teacher_modality
    student_modality = args.student_modality
    if args.dataset == "ms2":
        print("Using MS2 dataset")
        train_seq_list = return_ms2_split("train")
        val_seq_list = return_ms2_split("val")
        data_root = "/ocean/projects/cis220039p/mdt2/datasets/MS2_full"
        train_dataset = MS2(db_modality=teacher_modality,q_modality=student_modality,datasets_folder=data_root,seq=train_seq_list, augment=args.augment)
        val_dataset = MS2(db_modality=teacher_modality,q_modality=student_modality,datasets_folder=data_root,seq=val_seq_list, augment=False) #no augmentation for val dataset
    elif args.dataset == "wisard":
        print("Using Wisard dataset")
        dataset = Wisard_Dataset(args.dataset_path)
    elif args.dataset == "cart":
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
        train_dataset = CART(root_frame_dir=frame_list_root,db_modality=teacher_modality,q_modality=student_modality,datasets_folder=data_root,seq=train_seq_list, augment=args.augment)
        val_dataset = CART(root_frame_dir=frame_list_root,db_modality=teacher_modality,q_modality=student_modality,datasets_folder=data_root,seq=val_seq_list, augment=False) #no augmentation for val dataset

    elif args.dataset == "tartanair":
        print("Using TartanAir dataset")
        modality_map = {
            "rgb": "image",
            "depth": "depth"
        }
        teacher_modality = modality_map[teacher_modality]
        student_modality = modality_map[student_modality]
        dataset = Custom_TartanAirDataset(tartanair_data_root = tartanair_data_root,modalities=[teacher_modality, student_modality],
                                            # envs=['ModernCityDowntown','Gascola','Supermarket','Rome'],
                                            # difficulties=['easy'],
                                            frame_skip = 10     
                                        )
    else:
        raise ValueError("Invalid dataset name. Please choose 'ms2', 'wisard', 'cart' or 'tartanair'.")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    
    
    # Load the checkpoint if resuming training
    start_epoch = args.resume_epoch_num
    if args.resume:
        save_path = args.save_path
        checkpoint = torch.load(os.path.join(args.save_path, "model" +str(start_epoch) + '.pth'))
        if args.unfreeze_teacher:
            teacher_model.load_state_dict(checkpoint['teacher_model_state_dict'])
        student_model.load_state_dict(checkpoint['student_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if args.lr_scheduler:
            for e in range(start_epoch):
                scheduler.step()
        print("Resuming training from epoch {}".format(start_epoch))
    else:
        save_path = args.save_path
        save_path = os.path.join(save_path, args.dataset, f"{args.teacher_modality}_{args.student_modality}")
        #append currnet date and time to the save path
        save_path = os.path.join(save_path, time.strftime("%Y-%m-%d_%H-%M-%S"))

        # Create the save directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        import yaml
        with open(os.path.join(save_path,'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)
    viz_debug_dir = os.path.join(save_path, "viz_debug") if args.viz_debug else None


    # Train the model
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        num_batches = len(train_dataloader)
        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader)

        reset_training_loss = True
        for i in tqdm(range(len(train_dataloader))):
            if reset_training_loss:
                train_running_contrastive_loss = 0.0
                train_running_cosine_loss = 0.0
                reset_training_loss = False
            images,index = next(train_iterator)
            if args.viz_debug:
                save_viz_debug_images(images, index, epoch, i, "train", viz_debug_dir)
            train_contrastive_loss, train_cosine_loss = inference(args,teacher_model,student_model, teacher_modality,student_modality,images)
            train_running_contrastive_loss += train_contrastive_loss
            train_running_cosine_loss += train_cosine_loss
            torch.cuda.empty_cache()
            if i % 10 == 9:
                train_running_contrastive_loss = train_running_contrastive_loss / 10
                train_running_cosine_loss = train_running_cosine_loss / 10
                try:
                    val_images, val_index = next(val_iterator)
                    if args.viz_debug:
                        save_viz_debug_images(val_images, val_index, epoch, i, "val", viz_debug_dir)

                    val_contrastive_loss, val_cosine_loss = inference(args,teacher_model,student_model, teacher_modality,student_modality,val_images,test=True)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    val_images, val_index = next(val_iterator)
                    if args.viz_debug:
                        save_viz_debug_images(val_images, val_index, epoch, i, "val", viz_debug_dir)

                    val_contrastive_loss, val_cosine_loss = inference(args,teacher_model,student_model, teacher_modality,student_modality,val_images,test=True)
                print('[Epoch: %d, Batch: %d/%d] train loss: %.3f val loss: %.3f\n' % (epoch + 1, i + 1,num_batches, train_running_cosine_loss, val_cosine_loss))
                
                if args.wandb_use:
                    wandb.log({"train_contrastive_loss": train_running_contrastive_loss,
                                "train_cosine_loss": train_running_cosine_loss,
                                "val_contrastive_loss": val_contrastive_loss,
                                "val_cosine_loss": val_cosine_loss,
                    })
                reset_training_loss = True
            gc.collect()
            torch.cuda.empty_cache()
        
        del images, index, train_iterator, val_iterator
        if args.lr_scheduler:
            scheduler.step()

        # Save the model
        torch.save({
            'epoch': epoch,
            'student_model_type': args.model_name,
            'student_model_state_dict': student_model.state_dict(),
            'teacher_model_state_dict': teacher_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "train_contrastive_loss": train_running_contrastive_loss,
            "train_cosine_loss": train_running_cosine_loss,
            "val_contrastive_loss": val_contrastive_loss,
            "val_cosine_loss": val_cosine_loss
        }, os.path.join(save_path, "model" +str(epoch) + '.pth'))

        print("Epoch {} of {} took {:.3f}s\n".format(epoch+1, args.epochs, time.time() - start_time))

if __name__ == "__main__":
    train()