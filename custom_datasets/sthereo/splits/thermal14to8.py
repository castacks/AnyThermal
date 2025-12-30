import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from multiprocessing import Pool

def normalize_14bit_to_8bit(image: np.ndarray, apply_clahe=False) -> np.ndarray:
    p1, p99 = np.percentile(image, (1, 99))
    stretched = np.clip((image - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        stretched = clahe.apply(stretched)

    return stretched

def process_single_trajectory(traj_tuple):
    traj, apply_clahe = traj_tuple
    # skip_list = ['valley_morning', 'kaist_evening', 'kaist_afternoon', 'valley_evening']
    # if traj.name in skip_list:
    #     print(f"Skipping {traj.name}")
    #     return

    thermal_dir = traj / "image" / "stereo_thermal_14_left"
    if not thermal_dir.exists():
        print(f"Skipping {traj.name}: no stereo_thermal_14_left directory found.")
        return

    output_dir_norm = traj / "image" / "thermal8_left"
    output_dir_clahe = traj / "image" / "thermal8_left_clahe"
    output_dir_norm.mkdir(parents=True, exist_ok=True)
    if apply_clahe:
        output_dir_clahe.mkdir(parents=True, exist_ok=True)

    image_files = sorted(thermal_dir.glob("*.png"))
    for img_path in tqdm(image_files, desc=f"Processing {traj.name}", leave=False):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16:
            print(f"⚠️ Skipping invalid image: {img_path.name}")
            continue

        img_8bit = normalize_14bit_to_8bit(img, apply_clahe=False)
        out_path_norm = output_dir_norm / img_path.name
        cv2.imwrite(str(out_path_norm), img_8bit)

        if apply_clahe:
            img_clahe = normalize_14bit_to_8bit(img, apply_clahe=True)
            out_path_clahe = output_dir_clahe / img_path.name
            cv2.imwrite(str(out_path_clahe), img_clahe)

    print(f"✅ Finished: {traj.name}")

def process_thermal_images_parallel(root_dir: Path, apply_clahe=False, num_workers=4):
    traj_dirs = list(root_dir.glob("*"))
    task_list = [(traj, apply_clahe) for traj in traj_dirs]

    with Pool(processes=num_workers) as pool:
        pool.map(process_single_trajectory, task_list)

# === Usage ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/ocean/projects/cis220039p/mdt2/datasets/STHEREO/sequences",
                        help="Path to the root folder containing trajectories")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    root_path = Path(args.root_dir)
    process_thermal_images_parallel(root_path, apply_clahe=True, num_workers=args.workers)
