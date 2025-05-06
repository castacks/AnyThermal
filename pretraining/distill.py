import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
import torch.optim as optim
import time
import os
import argparse
import wandb
from tqdm import tqdm

from datasets.custom_dataset_loader import Custom_MS2Dataset
from datasets.wisard_dataset import Wisard_Dataset
from datasets.cart_dataset import CART_dataset
from datasets.tartanair_dataset import *
import gc
from utilities import DinoV2ExtractFeatures


def init_model(model_name):
    if model_name == "dinov2_vits14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    elif model_name == "dinov2_vitb16":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb16').cuda()
    else:
        raise ValueError("Invalid model name")
    return model

parser = argparse.ArgumentParser(description='Fine-tuning DINOv2 on ImageNet')
parser.add_argument('--dataset_path', default='/storage2/datasets/ms2_full/', type=str, help='Path to the ImageNet dataset')
parser.add_argument('--dataset', default='ms2', type=str, help='dataset name')
parser.add_argument('--teacher_modality', default='rgb', type=str, help='modality which will be frozen unless "unfreeze teacher" is true')
parser.add_argument('--student_modality', default='thermal', type=str, help='modality for which encoder has to be trained')
parser.add_argument('--unfreeze_teacher',action="store_true", help='modality for which encoder has to be trained')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--save_path', default='./checkpoints_ce/dinov2_ms2_checkpoints_thermal_global_bigger_denser_no_night', type=str, help='Path to save the checkpoints')
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--resume_epoch_num', default=0, type=int, help='Epoch number to resume training from')
parser.add_argument('--loss_type', default="ce", type=str, help='Loss type: mse or similarity')
parser.add_argument('--wandb_use',default=False, type=bool, help='Use wandb for logging')
parser.add_argument('--model_name', default='dinov2_vits14', type=str, help='Name of the encoder model')
args = parser.parse_args()
print(args)

#Initialize wandb
name = args.model_name + "_" + args.dataset + "_" + args.student_modality + "_distill"
if args.wandb_use:
    wandb.init(project="multiloc", name=name)

# Load models

teacher_model = init_model(args.model_name).cuda()
student_model = init_model(args.model_name).cuda()

if args.unfreeze_teacher:
    teacher_model.train()
else:
    teacher_model.eval()
    # Freeze the RGB model, so that only the thermal model is trained
    for param in teacher_model.parameters():
        param.requires_grad = False


student_model.train()

#PARV_TODO: Not being used
temperature = 1
embedding_dim = 384

if args.unfreeze_teacher:
    optimizer = optim.SGD(student_model.blocks[:].parameters() + teacher_model.blocks[:].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD(student_model.blocks[:].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def loss_fn_mse(outputs, targets):
    return F.mse_loss(outputs, targets)

def contrastive_loss(teacher_embed, student_embed, temperature=0.07):

    #PARV_TODO: Improve speed of this

    # Normalize embeddings
    teacher_embed = F.normalize(teacher_embed, dim=-1)
    student_embed = F.normalize(student_embed, dim=-1)
    
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
    log_probs = F.log_softmax(logits, dim=1)
    # Negative log probability of the true pairs (positive pairs)
    loss = -log_probs[:, 0].mean()
    
    return loss

def train():
    # Load the dataset
    teacher_modality = args.teacher_modality
    student_modality = args.student_modality
    if args.dataset == "ms2":
        print("Using MS2 dataset")
        modality_map = {
            "rgb": "rgb1",
            "thermal": "thermal1"
        }
        teacher_modality = modality_map[teacher_modality]
        student_modality = modality_map[student_modality]
        dataset = Custom_MS2Dataset(args.dataset_path)
    elif args.dataset == "wisard":
        print("Using Wisard dataset")
        dataset = Wisard_Dataset(args.dataset_path)
    elif args.dataset == "cart":
        print("Using CART dataset")
        dataset = CART_dataset(args.dataset_path)
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

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,persistent_workers=True)

    
    
    # Load the checkpoint if resuming training
    start_epoch = args.resume_epoch_num
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save_path, "model" +str(start_epoch) + '.pth'))
        if args.unfreeze_teacher:
            teacher_model.load_state_dict(checkpoint['teacher_model_state_dict'])
        student_model.load_state_dict(checkpoint['student_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
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
    

    # Train the model
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        running_loss = 0.0
        running_timing ={}
        for k in ['data_cuda_transfer', 'forward_pass', 'loss', 'optimizer', "dataloader_extra","dataloader_total","batch_time"]:
            running_timing[k] = 0.0
        num_batches = len(train_dataloader)

        # last_time_total = time.time()
        # last_time = time.time()

        iterator = iter(train_dataloader)

        torch.cuda.synchronize()
        # last_time_total = time.time()

        for i in tqdm(range(len(train_dataloader))):
            torch.cuda.synchronize()
            start_time = time.time()

            # Fetch batch
            images = next(iterator)

            fetch_done_time = time.time()

            dataloader_fetch_time = fetch_done_time - start_time
            running_timing['dataloader_extra'] = running_timing['dataloader_extra'] + dataloader_fetch_time
            # running_timing['dataloader_total'] = running_timing['dataloader_total'] + dataloader_total_time
            # last_time_total = time.time()
            # if i!=0:
                # st = time.time()
                # del teacher_images, student_images, teacher_output, student_output, loss
                
                # print("Garbage collection time: ", time.time() - st)

            st = time.time()
            
            teacher_images = images[teacher_modality].cuda()
            student_images = images[student_modality].cuda()
            et = time.time()
            running_timing['data_cuda_transfer'] = running_timing['data_cuda_transfer'] + et - st
            # Forward pass
            torch.cuda.synchronize()

            st = time.time()
            if args.unfreeze_teacher:
                teacher_output = teacher_model(teacher_images)
            else:
                with torch.inference_mode():
                    teacher_output = teacher_model(teacher_images)
            torch.cuda.synchronize()

            teacher_time = time.time() - st
            torch.cuda.synchronize()

            st = time.time()
            student_output = student_model(student_images)
            torch.cuda.synchronize()

            student_time = time.time() - st
            
            running_timing['forward_pass'] = running_timing['forward_pass'] + teacher_time + student_time
            
            # print("Forward pass time: ", teacher_time + student_time)
            # Compute loss
            torch.cuda.synchronize()

            st = time.time()
            if args.loss_type == "mse":
                loss = loss_fn_mse(student_output, teacher_output)
            elif args.loss_type == "ce":
                loss = contrastive_loss(teacher_output, student_output)
            torch.cuda.synchronize()

            end = time.time()
            running_timing['loss'] = running_timing['loss'] + end - st
            # print("Loss time: ", time.time() - st)
            # Backpropagation
            torch.cuda.synchronize()

            st = time.time()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            optimizer_time = time.time() - st
            running_timing['optimizer'] = running_timing['optimizer'] + optimizer_time

            # print("Optimizer time: ", optimizer_time)


            # if args.wandb_use:
                # st = time.time()
            #     wandb.log({"loss": loss.item()})
            #     wandb_time = time.time() - st
            #     print("Wandb time: ", wandb_time)
            running_loss += loss.item()

               
            
            # st_end = time.time()
            # print("Batch time: ", st_end - st_all)
            batch_time = time.time() - start_time
            running_timing['batch_time'] = running_timing['batch_time'] + batch_time
            if i % 10 == 9:
                running_loss /= 10
                for k in running_timing.keys():
                    running_timing[k] = running_timing[k] / 10
                print('[Epoch: %d, Batch: %d/%d] loss: %.3f\n' % (epoch + 1, i + 1,num_batches, running_loss))
                if args.wandb_use:
                    wandb.log({"running_loss": running_loss})
                    for k in running_timing.keys():
                        wandb.log({"time/"+k: running_timing[k]})
                running_loss = 0.0
                for k in running_timing.keys():
                    running_timing[k] = 0.0
                gc.collect()
            # last_time = time.time()

        scheduler.step()

        # Save the model
        torch.save({
            'epoch': epoch,
            'student_model_state_dict': student_model.state_dict(),
            'teacher_model_state_dict': teacher_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, os.path.join(save_path, "model" +str(epoch) + '.pth'))

        print("Epoch {} of {} took {:.3f}s\n".format(epoch+1, args.epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}\n".format(loss))

if __name__ == "__main__":
    train()