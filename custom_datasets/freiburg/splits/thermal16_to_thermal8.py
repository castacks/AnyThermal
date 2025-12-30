import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os
# Add fieldscale path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
Fieldscale_DIR = os.path.join(PROJECT_ROOT, "baselines", "thermal_processing", "fieldscale","python")

assert os.path.exists(Fieldscale_DIR), f"Fieldscale directory does not exist: {Fieldscale_DIR}"

# Add it to sys.path if not already there
if Fieldscale_DIR not in sys.path:
    sys.path.append(Fieldscale_DIR)
from fieldscale import Fieldscale

# Fieldscale parameter
params_200_no_clahe = {
    'max_diff': 400, 'min_diff': 400, 'iteration': 7, 'gamma': 1.5,
    'clahe': False, 'video': False
}  # Base config, CLAHE toggled dynamically

def compute_crop_bounds(img):
    row_sum = img.sum(axis=1)
    col_sum = img.sum(axis=0)
    valid_rows = np.where(row_sum > 0)[0]
    valid_cols = np.where(col_sum > 0)[0]
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        raise ValueError("Image seems to be fully padded or blank.")
    row_start, row_end = valid_rows[0], valid_rows[-1] + 1
    col_start, col_end = valid_cols[0], valid_cols[-1] + 1
    return row_start, row_end, col_start, col_end

def normalize_16bit_to_8bit(img, use_clahe=False):
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    img_clipped = np.clip(img, p1, p99)
    img_norm = ((img_clipped - p1) / (p99 - p1 + 1e-5) * 255).astype(np.uint8)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_norm = clahe.apply(img_norm)
    return img_norm

def apply_fieldscale(img, use_clahe=False):
    params = params_200_no_clahe.copy()
    params['clahe'] = use_clahe
    fs = Fieldscale(**params)
    tmp_path = "_tmp_fieldscale_input.png"
    cv2.imwrite(tmp_path, img)
    rescaled = fs(tmp_path)
    return rescaled

def process_image(path: Path, root_dir: Path, crop_box, use_fieldscale=False, use_clahe=False):
    if not path.exists():
        print(f"⚠️ Skipping missing file: {path}")
        return
    
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return
    if img is None or img.dtype != np.uint16:
        raise ValueError(f"Failed to read 16-bit image: {path}")

    row_start, row_end, col_start, col_end = crop_box
    cropped = img[row_start:row_end, col_start:col_end]

    rel_path = path.relative_to(root_dir)
    if use_clahe:
        output_dir = "thermal8_clahe"
    elif use_fieldscale:
        output_dir = "thermal8_fieldscale"
        output_dir_clahe = "thermal8_fieldscale_clahe"
    else:
        output_dir = "thermal8_aligned_RGB"
    output_path = root_dir / rel_path.parent.parent / output_dir / rel_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    

    if use_fieldscale:
        img_rescaled = apply_fieldscale(cropped, False)
    else:
        img_rescaled = normalize_16bit_to_8bit(cropped, use_clahe)

    try:
        cv2.imwrite(str(output_path), img_rescaled)
    except Exception as e:
        print(f"⚠️ Error writing {output_path}: {e}")
        return
    if use_fieldscale:
        output_path = root_dir / rel_path.parent.parent / output_dir_clahe / rel_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_rescaled_clahe = apply_fieldscale(cropped, True)
        try:
            cv2.imwrite(str(output_path), img_rescaled_clahe)
        except Exception as e:
            print(f"⚠️ Error writing {output_path}: {e}")
            return
    
def find_global_crop_bounds(txt_folder):
    min_row_start, max_row_end = 0, float('inf')
    min_col_start, max_col_end = 0, float('inf')
    first = True

    for txt_file in tqdm(sorted(Path(txt_folder).glob("*_thermal.txt"))):
        with open(txt_file, 'r') as f:
            for line in f:
                path = Path(line.strip())
                if not path.exists():
                    print(f"⚠️ Skipping missing file: {path}")
                    continue
                try:
                    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                except Exception as e:
                    print(f"⚠️ Error reading {path}: {e}")
                    continue
                if img is None or img.dtype != np.uint16:
                    print(f"⚠️ Invalid image format for {path}, expected 16-bit: {img.dtype if img is not None else 'None'}")
                    continue
                row_start, row_end, col_start, col_end = compute_crop_bounds(img)
                if first:
                    min_row_start, max_row_end = row_start, row_end
                    min_col_start, max_col_end = col_start, col_end
                    first = False
                else:
                    min_row_start = max(min_row_start, row_start)
                    max_row_end = min(max_row_end, row_end)
                    min_col_start = max(min_col_start, col_start)
                    max_col_end = min(max_col_end, col_end)

    if max_row_end <= min_row_start or max_col_end <= min_col_start:
        raise ValueError("No common valid crop area found.")

    return min_row_start, max_row_end, min_col_start, max_col_end

def main(txt_folder, root_dir, use_fieldscale=False, use_clahe=False):
    crop_metadata_path = Path("thermal16_crop_box_metadata.txt")
    txt_folder = Path(txt_folder)
    root_dir = Path(root_dir)

    crop_box = None
    if crop_metadata_path.exists():
        with open(crop_metadata_path, 'r') as f:
            crop_box_lines = f.readlines()
            crop_box = [ int(line.split(" ")[1].strip()) for line in crop_box_lines]
        if crop_box:
            print(f"Using existing crop box: {crop_box}")
    # Save crop box metadata
    else:
        crop_box = find_global_crop_bounds(txt_folder)

        print(f"Computed crop box: {crop_box}")

        with open(crop_metadata_path, 'w') as f:
            f.write(f"row_start: {crop_box[0]}")
            f.write(f"row_end: {crop_box[1]}")
            f.write(f"col_start: {crop_box[2]}")
            f.write(f"col_end: {crop_box[3]}")
    # import pdb; pdb.set_trace()
    for txt_file in tqdm(sorted(txt_folder.glob("*_thermal.txt"))):
        with open(txt_file, 'r') as f:
            image_paths = [Path(line.strip()) for line in f if line.strip()]
        for path in tqdm(sorted(image_paths), desc=f"Processing {txt_file.name}"):
            try:
                process_image(path, root_dir, crop_box, use_fieldscale, use_clahe)
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_folder", type=str, default="absolute_frame_lists", help="Path to folder containing *_thermal.txt files")
    parser.add_argument("--root_dir", type=str, default="/ocean/projects/cis220039p/mdt2/datasets/freiburg/train", help="Root directory prefix to preserve relative structure")
    parser.add_argument("--fieldscale", action="store_true", help="Use Fieldscale normalization instead of percentile-based")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE during Fieldscale normalization")
    args = parser.parse_args()

    main(args.txt_folder, args.root_dir, use_fieldscale=args.fieldscale, use_clahe=args.clahe)
