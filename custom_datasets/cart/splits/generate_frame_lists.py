import os
import argparse
import glob
from natsort import natsorted
import yaml
from tqdm import tqdm
import cv2
def extract_frame_number(filename):
    # Extract the frame number from the filename
    # Assuming the filename format is like "image_color-00001.jpg"
    # import pdb; pdb.set_trace()  # Debugging line to inspect the filename parts

    parts = filename.split('-')
    if len(parts) > 1:
        return int(parts[1].split('.')[0])  # Get the number before the file extension
    return -1  # Return -1 if no valid frame number is found

def load_excluded_indices(txt_path):
    excluded = set()
    with open(txt_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                start, end = int(extract_frame_number(parts[0])), int(extract_frame_number(parts[1]))
                excluded.update(range(start, end + 1))
    return excluded

def get_image_paths(image_dir, ext):
    return natsorted(glob.glob(os.path.join(image_dir, f"*.{ext.strip('.')}")))

def subsample_paths(image_paths, target_rate):
    # p = 0
    # final_list = []
    # for i, path in enumerate(image_paths):
    #     idx = extract_frame_number(path.split("/")[-1])
    #     if idx % target_rate == 0:
    #         final_list.append(path)
    # if len(final_list) ==0:
    #     import pdb; pdb.set_trace()  # Debugging line to inspect the subsampling
    # return final_list
    return image_paths[::target_rate]

def readable_path(images_list):
    new_list = []
    for i, path in enumerate(images_list):
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist. Skipping.")
            continue
        try:
            img = cv2.imread(path)
            new_list.append(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return new_list


def process_traj(traj, root_dir, meta_dir, output_dir, sync_rate, gps_available, start_idx, end_idx, modality="thermal",stereo=True,seg_path=None):

    if modality not in ["thermal", "rgb"]:
        raise ValueError("Modality must be either 'thermal' or 'rgb'.")
    if modality == "thermal":
        folder_name = "thermal8"
    else:
        folder_name = "eo"
    if stereo:
        folder_name = os.path.join("stereo_rectified", "thermal_color",folder_name)
    else:
        folder_name = os.path.join("thermal_color",folder_name)
    # import pdb; pdb.set_trace()  # Debugging line to inspect the trajectory and parameters
    
    img_dir = os.path.join(root_dir, traj, folder_name)
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory {img_dir} does not exist for trajectory {traj}. Skipping.")
        return 0
    img_all = get_image_paths(img_dir, ext=".png")
    if len(img_all) == 0:
        import pdb; pdb.set_trace()  # Debugging line to inspect the empty image directory
    img_sub_60 = subsample_paths(img_all, target_rate=60)

    if gps_available:
        thr_idx_file = os.path.join(meta_dir, f"{traj}_thermal_frame_indices.txt")
        img_excludes = load_excluded_indices(thr_idx_file) #using thermal indices to exclude RGB frames, since after sstereo_rectify, rgb and thermal frames are synchronized
    else:
        img_excludes = set()
        if start_idx >= 0:
            img_excludes.update(range(0, start_idx))
        if end_idx >= 0:
            img_excludes.update(range(end_idx+1, len(img_all)))

    img_final = [p for p in img_sub_60 if extract_frame_number(p.split("/")[-1]) not in img_excludes]
    img_final = readable_path(img_final)
    img_out_path = os.path.join(output_dir, f"{traj}_{modality}_frame_list.txt")

    with open(img_out_path, "w") as f:
        f.write("\n".join(img_final) + "\n")
    if modality == "rgb":
        return len(img_final)
    if seg_path is not None and seg_path !="":

        seg_mask_list = natsorted(glob.glob(os.path.join(seg_path, "masks", "*.png")))
        if len(seg_mask_list) == 0:
            import pdb; pdb.set_trace()  # Debugging line to inspect the empty segmentation mask directory
        thermal_img_list = natsorted(glob.glob(os.path.join(img_dir, "*.png")))
        final_seg_mask_list = []
        for p in seg_mask_list:
            idx = extract_frame_number(p.split("/")[-1])
            if idx < 0:
                print(f"Warning: Invalid frame number in segmentation mask {p}. Skipping.")
                continue
            thermal_img = os.path.join(img_dir, f"image-{idx:05d}.png")
            if thermal_img not in thermal_img_list:
                print(f"Warning: Corresponding thermal image {thermal_img} not found for segmentation mask {p}. Skipping.")
                continue
            final_seg_mask_list.append(p)

        seg_file_out_path = os.path.join(output_dir, f"{traj}_{modality}_segmentation_frame_list.txt")
        final_seg_mask_list = readable_path(final_seg_mask_list)
        with open(seg_file_out_path, "w") as f:
            f.write("\n".join(final_seg_mask_list) + "\n")
        thermal_seg_pair = []
        for img in final_seg_mask_list:
            base_name = os.path.basename(img)
            
            thermal_seg_pair.append(os.path.join(seg_path,"thermal8", base_name))
        img_out_seg_pair_path = os.path.join(output_dir, f"{traj}_{modality}_frame_list_seg_pair.txt")
        thermal_seg_pair = readable_path(thermal_seg_pair)
        with open(img_out_seg_pair_path, "w") as f:
            f.write("\n".join(thermal_seg_pair) + "\n")
    return len(img_final)
    
    # if len(rgb_final) != len(thr_final):
    #     print(f"{traj}: {len(rgb_final)} RGB, {len(thr_final)} Thermal frames")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files")
    parser.add_argument("--trajectory_file", type=str, default = "trajectories.yaml")
    parser.add_argument("--meta_dir", type=str, default="./static_segments_output/image_filenames")
    parser.add_argument("--output_dir", type=str, default="./static_segments_output/frames")
    parser.add_argument("--sync_rate", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trajs = yaml.safe_load(open(args.trajectory_file, "r"))["traj_list"]
    total_rgb_thermal_pairs  = 0
    for traj in tqdm(trajs.keys()):
        gps_available = trajs[traj]["gps"]
        end_idx = -1
        start_idx = -1
        if "thermal_only" in trajs[traj]:
            thermal_only = trajs[traj]["thermal_only"]
        else:
            thermal_only = False
        if not gps_available:
            start_idx = trajs[traj]["start_idx"]
            end_idx = trajs[traj]["end_idx"]
        seg_path = trajs[traj]["segmentation_path"]
        thermal_len = process_traj(traj, args.root_dir, args.meta_dir, args.output_dir, args.sync_rate,gps_available=gps_available, start_idx=start_idx, end_idx=end_idx,stereo=not thermal_only,seg_path=seg_path)
            
        if not thermal_only:
            rgb_len = process_traj(traj, args.root_dir, args.meta_dir, args.output_dir, args.sync_rate,gps_available=gps_available, start_idx=start_idx, end_idx=end_idx,modality="rgb")
            if thermal_len != rgb_len:
                print(f"{traj}: {thermal_len} Thermal is not equal to {rgb_len} RGB frames")
            total_rgb_thermal_pairs += thermal_len
            print(f"{traj}: {thermal_len} pairs of frames")
    print(f"Total RGB-Thermal pairs: {total_rgb_thermal_pairs}")

if __name__ == "__main__":
    main()
