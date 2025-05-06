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
import matplotlib.pyplot as plt
import numpy as np



def train():
    # Load the dataset
    teacher_modality = "rgb"
    student_modality = "depth"
    batch_size = 32
    
    print("Using TartanAir dataset")
    modality_map = {
        "rgb": "image",
        "depth": "depth"
    }
    teacher_modality = modality_map[teacher_modality]
    student_modality = modality_map[student_modality]
    dataset = Custom_TartanAirDataset(tartanair_data_root = tartanair_data_root,modalities=[teacher_modality, student_modality],
                                        envs=['ModernCityDowntown','Gascola','Supermarket','Rome'],
                                        difficulties=['easy'],
                                        frame_skip = 10     
                                    )
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    save_path = os.path.join("tartanair_depth")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    for i, images in tqdm(enumerate(train_dataloader)):
        # st_all = time.time()
        # if i!=0:
            # st = time.time()
            # del teacher_images, student_images, teacher_output, student_output, loss
            
            # print("Garbage collection time: ", time.time() - st)
        
        # teacher_images = images[teacher_modality].cuda()
        # student_images = images[student_modality].cuda()

        rgb = images['image'] /255.0
        depth_500 = images['depth'] /255.0
        depth_200 = images['depth_200'] /255.0
        depth_1000 = images['depth_1000'] /255.0

        #write a code to display image using matplotlib


        # rgb = rgb[0].numpy()
        # depth_500 = depth_500[0].numpy()
        # depth_200 = depth_200[0].numpy()
        # depth_1000 = depth_1000[0].numpy()
        print(rgb.shape)
        print(depth_500.shape)
        print(depth_200.shape)
        print(depth_1000.shape)
        rgb = np.transpose(rgb, (0,2, 3, 1))
        depth_500 = np.transpose(depth_500, (0,2, 3, 1))
        depth_200 = np.transpose(depth_200, (0,2, 3, 1))
        depth_1000 = np.transpose(depth_1000, (0,2, 3, 1))
        for j in range(rgb.shape[0]):
            plt.subplot(1, 4, 1)
            plt.imshow(rgb[j])
            plt.title('RGB')
            plt.subplot(1, 4, 2)
            plt.imshow(depth_500[j])
            plt.title('Depth 500')
            plt.subplot(1, 4, 3)
            plt.imshow(depth_200[j])
            plt.title('Depth 200')
            plt.subplot(1, 4, 4)
            plt.imshow(depth_1000[j])
            plt.title('Depth 1000')
            plt.savefig('tartanair_depth/image' + str(i*batch_size+j) + '.png')




            # # Forward pass
            # # st = time.time()
            # if args.unfreeze_teacher:
            #     teacher_output = teacher_model(teacher_images)
            # else:
            #     with torch.inference_mode():
            #         teacher_output = teacher_model(teacher_images)
            # # teacher_time = time.time() - st
            # # st = time.time()
            # student_output = student_model(student_images)
            # # student_time = time.time() - st
            # # print("Forward pass time: ", teacher_time + student_time)
            # # Compute loss

            # # st = time.time()
            # if args.loss_type == "mse":
            #     loss = loss_fn_mse(student_output, teacher_output)
            # elif args.loss_type == "ce":
            #     loss = contrastive_loss(teacher_output, student_output)
            # # print("Loss time: ", time.time() - st)
            # # Backpropagation
            # # st = time.time()
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # # optimizer_time = time.time() - st

            # # print("Optimizer time: ", optimizer_time)


            # # if args.wandb_use:
            # #     st = time.time()
            # #     wandb.log({"loss": loss.item()})
            # #     wandb_time = time.time() - st
            # #     print("Wandb time: ", wandb_time)
            # running_loss += loss.item()

               
            # if i % 10 == 9:
            #     running_loss /= 10
            #     print('[Epoch: %d, Batch: %d/%d] loss: %.3f\n' % (epoch + 1, i + 1,num_batches, running_loss))
            #     if args.wandb_use:
            #         wandb.log({"running_loss": running_loss})
            #     running_loss = 0.0
            #     gc.collect()
            # # st_end = time.time()
            # # print("Batch time: ", st_end - st_all)

        # scheduler.step()

        # # Save the model
        # torch.save({
        #     'epoch': epoch,
        #     'student_model_state_dict': student_model.state_dict(),
        #     'teacher_model_state_dict': teacher_model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss
        # }, os.path.join(save_path, "model" +str(epoch) + '.pth'))

        # print("Epoch {} of {} took {:.3f}s\n".format(epoch+1, args.epochs, time.time() - start_time))
        # print("  training loss (in-iteration): \t{:.6f}\n".format(loss))

if __name__ == "__main__":
    train()