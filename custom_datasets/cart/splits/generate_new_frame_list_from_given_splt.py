import os
import argparse
import yaml

def load_trajectories(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['traj_list']

def map_trajectories(data):
    """
    Maps trajectory keys to their corresponding segmentation paths.
    """
    traj_map = {}
    for traj in data:
        key = traj
        seg_path = data[traj].get('segmentation_path', '')
        if not seg_path:
            print(f"[WARNING] No segmentation path for trajectory: {traj}")
            continue
        if key and seg_path:
            traj_map[key] = str.replace(seg_path,"/ocean/projects/cis220039p/mdt2/shared/CART/labeled_thermal_singles/","")
            traj_map[key] = str.replace(traj_map[key],"2022-05-08_BigBear","big_bear")
            traj_map[key] = str.replace(traj_map[key],"2022-04-03_Idyllwild/2022-04-03-12","caltech-coregistered-nature-dataset/ONR_2022-04-03-12")
            traj_map[key] = str.replace(traj_map[key],"2022-04-03_JoshuaTree/2022-04-03-17","caltech-coregistered-nature-dataset/ONR_2022-04-03-17")
            traj_map[key] = str.replace(traj_map[key],"2021-09-09-KentuckyRiver/flight","kentucky_river/flight")
            traj_map[key] = str.replace(traj_map[key],"2023-03-XX_Duck/ONR_2023-03-","caltech_duck/ONR_2023-03-")

    inverse_traj_map = {v: k for k, v in traj_map.items()}
        
    return inverse_traj_map

def parse_frame_index(thermal_path):
    fname = os.path.basename(thermal_path)
    rgbt_frame_prefix = "_".join(thermal_path.split('/')[:-2])  # e.g., 'thermal-09720.tiff' → 'thermal'
    thermal_frame_prefix = "/".join(thermal_path.split('/')[:-2])  # e.g., 'thermal-09720.tiff' → 'thermal-09720'
    # return fname.split('-')[-1].split('.')[0]  # 'thermal-09720.tiff' → '09720'
    return thermal_frame_prefix,rgbt_frame_prefix, fname.split('-')[-1].split('.')[0] 
def find_traj_key_from_path(path):
    """
    Extracts trajectory timestamp key (e.g., '2022-04-03-17-12-07') from the input path.
    """
    # for part in path.split('/'):
    #     if part.startswith("ONR_"):
    #         return part.replace("ONR_", "")
    #     if part.count('-') == 4:  # '2022-12-20-12-16-02'
    #         return part
    return "/".join(path.split('/')[:2])  # Fallback to last three parts
    # return None

def build_output_paths(segmentation_path, frame_prefix,frame_idx,mode):
    if mode == 'rgbt':
        root_dir = "/ocean/projects/cis220039p/mdt2/shared/CART/labeled_rgbt_pairs"
        rgb_dir = os.path.join(root_dir, 'color')
        thermal8_dir = os.path.join(root_dir, 'thermal8')
        mask_dir = os.path.join(root_dir, 'annotations')
        extension = 'jpg'
        rgb_path = os.path.join(rgb_dir, f'{frame_prefix}-{frame_idx}.{extension}')
        thermal8_path = os.path.join(thermal8_dir, f'{frame_prefix}-{frame_idx}.{extension}')
        mask_path = os.path.join(mask_dir, f'{frame_prefix}_mask-{frame_idx}.png')
        return rgb_path, thermal8_path, mask_path
    elif mode == 'thermal':
        root_dir = "/ocean/projects/cis220039p/mdt2/shared/CART/labeled_thermal_singles"
        extension = 'png'
        thermal8_path = os.path.join(root_dir, f'{frame_prefix}/thermal8/pair-{frame_idx}.png')
        mask_path = os.path.join(root_dir, f'{frame_prefix}/masks/pair-{frame_idx}.png')

        thermal8_path = thermal8_path.replace("big_bear","2022-05-08_BigBear")
        mask_path = mask_path.replace("big_bear","2022-05-08_BigBear")

        thermal8_path = thermal8_path.replace("caltech-coregistered-nature-dataset/ONR_2022-04-03-12","2022-04-03_Idyllwild/2022-04-03-12")
        mask_path = mask_path.replace("caltech-coregistered-nature-dataset/ONR_2022-04-03-12","2022-04-03_Idyllwild/2022-04-03-12")

        thermal8_path = thermal8_path.replace("caltech-coregistered-nature-dataset/ONR_2022-04-03-17","2022-04-03_JoshuaTree/2022-04-03-17")
        mask_path = mask_path.replace("caltech-coregistered-nature-dataset/ONR_2022-04-03-17","2022-04-03_JoshuaTree/2022-04-03-17")

        thermal8_path = thermal8_path.replace("kentucky_river/flight","2021-09-09-KentuckyRiver/flight")
        mask_path = mask_path.replace("kentucky_river/flight","2021-09-09-KentuckyRiver/flight")
        
        mask_path = mask_path.replace("caltech_duck/ONR_2023-03-","2023-03-XX_Duck/ONR_2023-03-")
        thermal8_path = thermal8_path.replace("caltech_duck/ONR_2023-03-","2023-03-XX_Duck/ONR_2023-03-")
        
        return thermal8_path, mask_path
        
    

def main(input_txt, traj_yaml, rgbt_output_txt,thermal_output_txt):
    traj_map = load_trajectories(traj_yaml)

    seg_to_traj_map = map_trajectories(traj_map)
    # import pdb;pdb.set_trace()

    with open(input_txt, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    rgb_thermal_output_lines = []
    thermal_only_output_lines = []
    rgbt_missing_counter = 0
    thermal_missing_counter = 0

    for line in lines:
        thermal_path, mask_path = line.split(',')

        thermal_frame_prefix,rgbt_frame_prefix,frame_idx = parse_frame_index(thermal_path)
        traj_key = find_traj_key_from_path(thermal_path)

        real_traj_key = seg_to_traj_map.get(traj_key, None)
        # import pdb;pdb.set_trace()

        if traj_key is None:
            print(f"[ERROR] Trajectory key not found in path: {thermal_path}")
            continue
        if traj_key not in seg_to_traj_map:
            print(f"[ERROR] Trajectory {traj_key} not in seg_to_traj_map")
            continue

        seg_path = traj_map[real_traj_key].get('segmentation_path', '')
        if not seg_path:
            print(f"[ERROR] No segmentation path for trajectory: {real_traj_key}")
            continue

        
        for mode in ['thermal']:

            if 'kentucky' in line and mode == 'rgbt':
                print(f"[SKIPPING] Kentucky trajectory {traj_key} does not have RGB-T data.")
                continue
            if mode == 'rgbt':            
               rgb_path, thermal8_path, mask_path_corrected = build_output_paths(seg_path, rgbt_frame_prefix,frame_idx,mode)
            elif mode == 'thermal':
                thermal8_path, mask_path_corrected = build_output_paths(seg_path, thermal_frame_prefix,frame_idx,mode)
            # Check if all files exist
            missing = []
            if mode == 'rgbt':
                for path in [rgb_path, thermal8_path, mask_path_corrected]:
                    if not os.path.exists(path):
                        missing.append(path)
            elif mode == 'thermal':
                for path in [thermal8_path, mask_path_corrected]:
                    if not os.path.exists(path):
                        missing.append(path)
                
            if missing:
                if mode == 'rgbt':
                    rgbt_missing_counter += 1
                elif mode == 'thermal':
                    thermal_missing_counter += 1
                print(f"[MISSING FILES] For frame {frame_idx} in {real_traj_key} ({mode}):")
                for path in missing:
                    print(f"  - {path}")
                continue

            if mode == 'rgbt':
                rgb_thermal_output_lines.append(f"{rgb_path},{thermal8_path},{mask_path_corrected}")
            elif mode == 'thermal':
                thermal_only_output_lines.append(f"{thermal8_path},{mask_path_corrected}")

    if rgbt_output_txt != "":
        with open(rgbt_output_txt, 'w') as f:
            for line in rgb_thermal_output_lines:
                f.write(f"{line}\n")
    with open(thermal_output_txt, 'w') as f:
        for line in thermal_only_output_lines:
            f.write(f"{line}\n")

    print(f"✅ RGB-T Output written to {rgbt_output_txt} with {len(rgb_thermal_output_lines)} valid entries, ❗ {rgbt_missing_counter}/{len(lines)} frames were missing files.")
    print(f"✅ Thermal Only Output written to {thermal_output_txt} with {len(thermal_only_output_lines)} valid entries, ❗ {thermal_missing_counter}/{len(lines)} frames were missing files.")

    if len(rgb_thermal_output_lines) == 0:
        print("[ERROR] No valid RGB-T entries found.")
        import pdb;pdb.set_trace()
    if len(thermal_only_output_lines) == 0:
        print("[ERROR] No valid Thermal entries found.")
        import pdb;pdb.set_trace()


def geographic_splits(args):

    main_root = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/thermal_splits/geographic/region/"

    for place in ["socal","northcarolina","kentucky"]:
        for split in ["train","val","test"]:
            args.input_txt = os.path.join(main_root, f"{place}/{split}.txt")

            input_txt_base = str.replace(args.input_txt, "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/thermal_splits/geographic/region/", "")
            input_txt_base = "_".join(input_txt_base.split('/'))  # e.g., 'socal_train.txt' → 'socal_train'
            input_txt_base = input_txt_base.replace(".txt", "")
            args.rgbt_output_txt = f"geographic_splits/{input_txt_base}_{args.output_rgb_txt}.txt"
            args.thermal_output_txt = f"geographic_splits/{input_txt_base}_{args.output_thermal_txt}.txt"

            main(args.input_txt, args.traj_yaml, args.rgbt_output_txt, args.thermal_output_txt)

def random_splits(args):
    main_root = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/thermal_splits/random"

    for split in ["train","val","test"]:
        args.input_txt = os.path.join(main_root, f"{split}.txt")
        args.thermal_output_txt = f"random_splits/{split}.txt"

        main(args.input_txt, args.traj_yaml, "", args.thermal_output_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_yaml", type=str, default="all_trajectories.yaml", help="Path to trajectories.yaml")
    parser.add_argument("--output_rgb_txt", type=str,default="rgbt", help="Path to write output file")
    parser.add_argument("--output_thermal_txt", type=str,default="thermal", help="Path to write output file")
    args = parser.parse_args()

    # geographic_splits(args)
    random_splits(args)
