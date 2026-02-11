import os
import cv2
import yaml
import argparse
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from multiprocessing import Pool, cpu_count

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_trajectories_to_rotate(traj_config):
    rotate_trajs = {}
    for traj_name, attrs in traj_config.get("traj_list", {}).items():
        if attrs.get("rotate_180", False):
            rotate_trajs[traj_name] = attrs
    return rotate_trajs

def rotate_image_180(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to read image {img_path}")
        return
    rotated = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(img_path, rotated)

def rotate_images_in_folder(folder_path, num_workers=None):
    png_files = natsorted(glob(os.path.join(folder_path, "*.png")))
    if not png_files:
        print(f"No .png files found in {folder_path}")
        return

    print(f"Rotating {len(png_files)} images in {folder_path} using {num_workers or cpu_count()} workers")
    with Pool(processes=num_workers or cpu_count()) as pool:
        list(tqdm(pool.imap(rotate_image_180, png_files), total=len(png_files), desc=f"Rotating in {os.path.basename(folder_path)}"))

def process_trajectory(root_dir, traj_name, traj_info, num_workers=None):
    traj_path = os.path.join(root_dir, traj_name)
    thermal_only = traj_info.get("thermal_only", False)

    if thermal_only:
        folders = [os.path.join(traj_path, "thermal_color", "thermal8")]
    else:
        folders = [
            os.path.join(traj_path, "stereo_rectified", "thermal_color", "eo"),
            os.path.join(traj_path, "stereo_rectified", "thermal_color", "thermal8"),
            os.path.join(traj_path, "stereo_rectified", "thermal_color", "sxs"),
        ]

    for folder in folders:
        if os.path.exists(folder):
            rotate_images_in_folder(folder, num_workers=num_workers)
        else:
            print(f"Folder not found: {folder}")

def main():
    parser = argparse.ArgumentParser(description="Rotate trajectory images by 180 degrees.")
    parser.add_argument("--yaml_file", type=str, default="all_trajectories.yaml", help="Path to YAML file with trajectory config")
    parser.add_argument("--root_dir", type=str, default="/ocean/projects/cis220039p/mdt2/shared/CART/bag_files", help="Root directory containing trajectories")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers for image rotation")
    args = parser.parse_args()

    traj_config = load_yaml(args.yaml_file)
    rotate_trajs = get_trajectories_to_rotate(traj_config)

    print(f"Found {len(rotate_trajs)} trajectories to rotate:")

    for traj_name, traj_info in rotate_trajs.items():
        print(f"Processing trajectory: {traj_name}")
        process_trajectory(args.root_dir, traj_name, traj_info, num_workers=args.num_workers)

if __name__ == "__main__":
    main()
