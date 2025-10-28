import os
import torch
import numpy as np
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from sklearn.metrics import confusion_matrix
import tyro
from dataclasses import dataclass, field
from typing import List, Literal, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import sys 
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc") # Add the parent directory to the path
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/custom_datasets") # Add the custom models directory to the path

from utils.segment_utils import label_to_rgb

from custom_datasets.cart_dataset import *
from custom_datasets.mfnet_dataset import *
from custom_models.str_to_cls import get_model_from_string
import gc

def compute_iou(conf_matrix):
    intersection = np.diag(conf_matrix)
    ground_truth_set = conf_matrix.sum(axis=1)
    predicted_set = conf_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / np.clip(union, a_min=1e-6, a_max=None)
    return IoU

def evaluate_segmentation(model, dataloader, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_images, all_preds, all_targets = [], [], []
    db_modality = dataloader.dataset.db_modality
    with torch.no_grad():
        for batch, index in tqdm(dataloader, desc="Evaluating"):
            images, masks = batch[db_modality].to(model.device), batch["seg_mask"].to(model.device).long()
            masks = masks.squeeze(-3)
            # import pdb; pdb.set_trace()
            valid_mask = (masks >= 0).cpu().numpy()
            preds = model.forward(images)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            all_images.extend(batch[db_modality].cpu().numpy())
            all_preds.extend(preds)
            all_targets.extend(targets)

            # import pdb; pdb.set_trace()

            for p, t,m in zip(preds, targets, valid_mask):
                conf_matrix += confusion_matrix(t.flatten()[m.flatten()], p.flatten()[m.flatten()], labels=list(range(num_classes)))
            gc.collect()
            torch.cuda.empty_cache()

    correct_per_class = np.diag(conf_matrix)
    total_per_class = conf_matrix.sum(axis=1)
    per_class_acc = correct_per_class / np.clip(total_per_class, a_min=1e-6, a_max=None)
    mean_acc = np.nanmean(per_class_acc)

    pixel_acc = correct_per_class.sum() / conf_matrix.sum()
    per_class_iou = compute_iou(conf_matrix)
    miou = np.nanmean(per_class_iou)

    freq = total_per_class / np.clip(conf_matrix.sum(), a_min=1e-6, a_max=None)
    fwiou = np.sum(freq * per_class_iou)

    return pixel_acc, mean_acc, fwiou, miou, per_class_acc, per_class_iou, total_per_class, all_images, all_targets, all_preds

@dataclass
class BenchmarkSegArgs:
    splits: List[str]
    model_names: List[str]
    dataset_name: Literal['mfnet','cart_random']
    batch_size: int = 4
    use_wandb: bool = False
    output_dir: str = './qualitative_outputs'
    # test_areas: List[Literal['socal', 'northcarolina', 'kentucky']] = field(default_factory=lambda: ['socal'])
    viz_outputs: bool = False

def save_viz_outputs(images, labels, preds, out_dir, model_name, label_to_rgb_dict, num_samples=200):
    os.makedirs(out_dir, exist_ok=True)
    # indices = np.linspace(0, len(images)-1, num=min(num_samples, len(images)), dtype=int)
    for idx in range(len(images)):
        fig, axs = plt.subplots(1, 4, figsize=(12, 4))
        axs[0].imshow(images[idx][0], cmap='gray')
        axs[0].set_title("Thermal")
        axs[1].imshow(label_to_rgb(labels[idx], label_to_rgb_dict))
        axs[1].set_title("Label")
        axs[2].imshow(label_to_rgb(preds[idx], label_to_rgb_dict))
        axs[2].set_title("Prediction")
        mask = np.logical_not(preds[idx]==labels[idx])
        incorrect_mean = mask.mean()
        axs[3].imshow(mask)
        axs[3].set_title(f"Incorrect mask {np.round(incorrect_mean,3)}")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{idx:04d}_incorrect_mean_{incorrect_mean}.png"))
        plt.close()

def main(args: BenchmarkSegArgs):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for split in args.splits:
        # for area in args.test_areas:
        # wandb_name = f"segmentation_benchmark_{args.dataset_name}_{args.model_names}_{split}"
        # wandb.init(project="Segmnetation Benchmark", config=vars(args), name=wandb_name)
    

        # if args.dataset_name == "cart":
        #     area = split.split("_")[0]
        #     temp_split = split.split("_")[1]
        #     print(f"Evaluating on area: {area}, split: {temp_split}")
        #     seq_list = return_cart_split_segmentation_geographic(temp_split, area, "thermal")
        #     dataset = CART(args=args,root_frame_dir=None, db_modality="thr_seg", q_modality="seg_mask", datasets_folder=None, seq=seq_list, augment=False, seq_as_txt="thermal",crop_images=False)
        if "cart" in args.dataset_name:
            print("Using CART dataset")

            type_of_split = args.dataset_name.split("_")[-1]
            if type_of_split == "random":
                seq_list = return_cart_split_segmentation_random(split)
            elif type_of_split == "geographic":
                raise NotImplementedError("Geographic split not implemented for segmentation yet.")
            else:
                raise ValueError("Unsupported CART split type. Use 'random' or 'geographic'.")
            dataset = CART(args=args,root_frame_dir=None, db_modality="thr", q_modality="seg_mask", datasets_folder=None, seq=seq_list, augment=False, seq_as_txt="thermal", crop_images=False)

        elif args.dataset_name == "mfnet":
            if split not in ['train','val', 'test']:
                raise ValueError("MFNet only has 'val' and 'test' splits.")
            seq_list = return_mfnet_split(split)
            dataset = MFNet(args=args,root_frame_dir=None, db_modality="thr", q_modality="seg_mask", datasets_folder=None, seq=seq_list, augment=False,crop_images=False)
        else:
            raise ValueError(f"{args.dataset_name} is not supported currently")

        num_classes = dataset.semantic_classes
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        root_out_dir = os.path.join(args.output_dir, 'segmentation', args.dataset_name, timestamp,split)
        os.makedirs(root_out_dir, exist_ok=True)

        args_file = os.path.join(root_out_dir, 'args.txt')
        with open(args_file, 'w') as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")

        mean_metrics_csv = os.path.join(root_out_dir, 'mean_metrics.csv')
        iou_csv = os.path.join(root_out_dir, 'iou_metrics.csv')
        pixel_acc_csv = os.path.join(root_out_dir, 'pixel_accuracy_metrics.csv')

        with open(mean_metrics_csv, 'w', newline='') as mm_file, \
                open(iou_csv, 'w', newline='') as iou_file, \
                open(pixel_acc_csv, 'w', newline='') as acc_file:

            mean_writer = csv.DictWriter(mm_file, fieldnames=[
                'model_name', 'pixel_accuracy', 'mean_pixel_accuracy', 'mean_iou', 'frequency_weighted_iou'
            ])
            mean_writer.writeheader()

            iou_writer = csv.DictWriter(iou_file, fieldnames=[
                'model_name'] + [f'class_{i}_iou' for i in range(num_classes)]
            )
            iou_writer.writeheader()

            acc_writer = csv.DictWriter(acc_file, fieldnames=[
                'model_name'] + 
                [f'class_{i}_pixel_acc' for i in range(num_classes)] +
                [f'class_{i}_total_instances' for i in range(num_classes)]
            )
            acc_writer.writeheader()

            for model_name in args.model_names:
                model = get_model_from_string(args,model_name, task='segmentation', num_classes=num_classes)
                model.eval()
                pixel_acc, mean_acc, fwiou, miou, per_class_acc, per_class_iou, total_per_class, all_images, all_labels, all_preds = evaluate_segmentation(
                    model, dataloader, num_classes
                )

                # wandb.log({
                #     **{f"class_{i}_total_instances": int(val) for i, val in enumerate(total_per_class)},
                #     "model": model_name,
                #     "pixel_accuracy": pixel_acc,
                #     "mean_pixel_accuracy": mean_acc,
                #     "mean_iou": miou,
                #     "frequency_weighted_iou": fwiou,
                #     **{f"class_{i}_iou": val for i, val in enumerate(per_class_iou)},
                #     **{f"class_{i}_pixel_acc": val for i, val in enumerate(per_class_acc)}
                # })

                mean_writer.writerow({
                    'model_name': model_name,
                    'pixel_accuracy': pixel_acc,
                    'mean_pixel_accuracy': mean_acc,
                    'mean_iou': miou,
                    'frequency_weighted_iou': fwiou
                })

                iou_writer.writerow({
                    'model_name': model_name,
                    **{f'class_{i}_iou': per_class_iou[i] for i in range(num_classes)}
                })

                acc_writer.writerow({
                    'model_name': model_name,
                    **{f'class_{i}_pixel_acc': per_class_acc[i] for i in range(num_classes)},
                    **{f'class_{i}_total_instances': int(total_per_class[i]) for i in range(num_classes)}
                })

                print(f"[Split: {split} | Model: {model_name}] Pixel Acc: {pixel_acc:.4f}, Mean Acc: {mean_acc:.4f}, mIoU: {miou:.4f}, FWIoU: {fwiou:.4f}")

                if args.viz_outputs:
                    model_out_dir = os.path.join(root_out_dir, model_name)
                    save_viz_outputs(all_images, all_labels, all_preds, model_out_dir, model_name, label_to_rgb_dict=dataset.semantic_id_to_rgb)

if __name__ == '__main__':
    main(tyro.cli(BenchmarkSegArgs))
