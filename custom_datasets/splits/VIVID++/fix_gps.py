import os
from pathlib import Path
import numpy as np

def apply_offset_to_gps(gps_path: Path, offset: np.ndarray, output_suffix="_absolute"):
    out_path = gps_path.parent / f"{gps_path.stem}{output_suffix}.txt"
    with open(gps_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            parts = line.strip().split(", ")
            if len(parts) != 3:
                continue
            try:
                x, y, ts = float(parts[0]), float(parts[1]), float(parts[2])
                x_abs, y_abs = x + offset[0], y + offset[1]
                fout.write(f"{x_abs:.6f} {y_abs:.6f} {ts:.6f}\n")
            except ValueError:
                continue
    print(f"✅ Written: {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apply fixed offset to all GPS files.")
    parser.add_argument("--root_dir", default="/ocean/projects/cis220039p/mdt2/datasets/VIVID++/extracted_data", type=str)
    parser.add_argument("--offset", nargs=2, type=float, default=[988118.672713562, 1818806.4875518726], help="Offset to apply as [x y]")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    offset = np.array(args.offset)

    for traj_group in sorted(root_dir.glob("*/")):
        for traj_path in traj_group.glob("*/gpslist.txt"):
            print(f"Processing: {traj_path}")
            # apply_offset_to_gps(traj_path, offset)

if __name__ == "__main__":
    main()
