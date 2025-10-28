#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False

FROZEN_DIR_NAME = "mmdistill_mfnet_frozen_rgb_dinov2"
THERMAL_DIR_NAME = "mmdistill_mfnet_thermal_dinov2_with_tartan_rgbt"

FNAME_RE = re.compile(
    r"^sample_(?P<idx>\d+)_incorrect_mean_(?P<mean>[0-9]*\.?[0-9]+)\.(?P<ext>[A-Za-z0-9]+)$"
)

def scan_folder(folder: Path) -> Dict[int, Tuple[Path, float]]:
    out = {}
    if not folder.exists():
        return out
    for name in os.listdir(folder):
        m = FNAME_RE.match(name)
        if not m:
            continue
        idx = int(m.group("idx"))
        mean = float(m.group("mean"))
        p = folder / name
        out[idx] = (p, mean)
    return out

def select_indices(frozen, thermal):
    selected = []
    all_idxs = sorted(set(frozen.keys()) | set(thermal.keys()))
    for i in all_idxs:
        if i in frozen and i in thermal:
            if frozen[i][1] > thermal[i][1]:
                selected.append(i)
        elif i in frozen and i not in thermal:
            selected.append(i)
    return selected

# --- Video helpers ---
class VideoWriter:
    def __init__(self, out_path: Path, fps: int, frame_size: Tuple[int, int]):
        self.out_path = str(out_path)
        self.fps = fps
        self.frame_size = frame_size
        self.backend = None
        self._writer = None
        self._init_writer()

    def _init_writer(self):
        if CV2_AVAILABLE:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self.out_path, fourcc, float(self.fps), self.frame_size
            )
            self.backend = "cv2"
        elif IMAGEIO_AVAILABLE:
            self._writer = imageio.get_writer(self.out_path, fps=self.fps)
            self.backend = "imageio"
        else:
            raise RuntimeError("Install opencv-python or imageio to enable video writing.")

    def write(self, frame_bgr):
        if self.backend == "cv2":
            self._writer.write(frame_bgr)
        else:
            self._writer.append_data(frame_bgr[:, :, ::-1])  # BGR→RGB

    def close(self):
        if self.backend == "cv2":
            self._writer.release()
        else:
            self._writer.close()

def read_image(path: Path):
    if CV2_AVAILABLE:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img
    else:
        rgb = imageio.imread(str(path))
        if rgb.ndim == 2:
            import numpy as np
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        return rgb[:, :, ::-1]  # RGB→BGR

def build_videos_for_indices(indices, frozen, thermal, out_dir: Path, fps: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get size from first valid frame
    first_img = None
    for i in indices:
        if i in frozen:
            first_img = read_image(frozen[i][0]); break
        elif i in thermal:
            first_img = read_image(thermal[i][0]); break
    if first_img is None:
        raise RuntimeError("No frames found.")

    H, W = first_img.shape[:2]
    size = (W, H)

    frozen_out = out_dir / f"frozen_rgb_{fps}fps.mp4"
    thermal_out = out_dir / f"anythermal_{fps}fps.mp4"
    stacked_out = out_dir / f"stacked_{fps}fps.mp4"

    vw_frozen = VideoWriter(frozen_out, fps=fps, frame_size=size)
    vw_thermal = VideoWriter(thermal_out, fps=fps, frame_size=size)
    vw_stacked = VideoWriter(stacked_out, fps=fps, frame_size=(W, H*2))  # vertical stack

    import numpy as np

    try:
        for i in indices:
            fr_img = read_image(frozen[i][0]) if i in frozen else None
            th_img = read_image(thermal[i][0]) if i in thermal else None
            if fr_img is None and th_img is None:
                continue
            if fr_img is None: fr_img = th_img
            if th_img is None: th_img = fr_img

            vw_frozen.write(fr_img)
            vw_thermal.write(th_img)

            # vertically stack (ensure same size)
            fr_resized = cv2.resize(fr_img, (W, H)) if CV2_AVAILABLE else fr_img
            th_resized = cv2.resize(th_img, (W, H)) if CV2_AVAILABLE else th_img
            stacked = np.vstack([fr_resized, th_resized])
            vw_stacked.write(stacked)

    finally:
        vw_frozen.close()
        vw_thermal.close()
        vw_stacked.close()

    return frozen_out, thermal_out, stacked_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", type=str, default="out_pairs")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    frozen = scan_folder(root / FROZEN_DIR_NAME)
    thermal = scan_folder(root / THERMAL_DIR_NAME)

    selected = select_indices(frozen, thermal)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pairs_file = out_dir / "pairs.txt"
    with open(pairs_file, "w") as f:
        f.write("index,frozen_mean,anythermal_mean\n")
        for i in selected:
            f.write(f"{i},{frozen.get(i,(None,float('nan')))[1]},{thermal.get(i,(None,float('nan')))[1]}\n")

    fr, th, st = build_videos_for_indices(selected, frozen, thermal, out_dir, args.fps)
    print(f"Wrote: {pairs_file}\nVideos:\n - {fr}\n - {th}\n - {st}")

if __name__ == "__main__":
    main()
