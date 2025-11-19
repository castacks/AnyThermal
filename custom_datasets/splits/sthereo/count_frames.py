from pathlib import Path
from datetime import datetime
import os
import pandas as pd

# === Configuration ===
root_dir = Path("/ocean/projects/cis220039p/mdt2/datasets/STHEREO/sequences")
modal_rgb = "stereo_left"
modal_thm = "stereo_thermal_14_left"
img_subpath = Path("image")
gps_threshold_sec = 0.5     # for GPS matching
pair_threshold_sec = 0.1    # for RGB ↔ Thermal pairing
output_dir = "frame_lists"
os.makedirs(output_dir, exist_ok=True)

def extract_timestamp_ns(file: Path) -> float:
    try:
        return float(int(file.stem) / 1e9)
    except Exception:
        return None

# Find all valid sequence dirs
sequence_dirs = [p.parent.parent for p in root_dir.glob("*/image/stereo_left") if (p.parent / modal_thm).exists()]

for seq in sequence_dirs:
    print(f"Processing: {seq.name}")
    rgb_dir = seq / img_subpath / modal_rgb
    thm_dir = seq / img_subpath / modal_thm
    seq_name = str(seq.relative_to(root_dir)).replace("/", "_")

    # Load GPS CSV
    gps_path = seq / "pose" / "global_pose.csv"
    gps_entries = []
    if gps_path.exists():
        try:
            gps_df = pd.read_csv(gps_path, header=None)
            gps_entries = gps_df[[0,1,2,3]].values.tolist()
        except Exception as e:
            print(f"❌ Failed to read GPS CSV for {seq.name}: {e}")
            continue
    else:
        print(f"⚠️ No GPS file for {seq.name}")
        continue

    # Load image timestamps
    rgb_files = [(f, extract_timestamp_ns(f)) for f in rgb_dir.glob("*.png")]
    thm_files = [(f, extract_timestamp_ns(f)) for f in thm_dir.glob("*.png")]
    rgb_files = [(ts, f) for f, ts in rgb_files if ts is not None]
    thm_files = [(ts, f) for f, ts in thm_files if ts is not None]
    thm_timestamps = [ts for ts, _ in thm_files]
    thm_map = dict(thm_files)

    matched_rows = []
    no_pair_count = 0
    no_gps_count = 0
    for rgb_ts, rgb_file in sorted(rgb_files):
        # Match to closest thermal
        closest_thm_ts = min(thm_timestamps, key=lambda t: abs(t - rgb_ts), default=None)
        if closest_thm_ts is None or abs(closest_thm_ts - rgb_ts) > pair_threshold_sec:
            print(f"⚠️ Skipping RGB file {rgb_file.name} at {rgb_ts:.3f} — no thermal match within ±{pair_threshold_sec}s")
            no_pair_count += 1
            continue  # No thermal match
        

        # Match to closest GPS
        gps_match = min(gps_entries, key=lambda row: abs(row[0] - rgb_ts), default=None)
        if gps_match is None or abs(gps_match[0] - rgb_ts) > gps_threshold_sec:
            print(f"⚠️ Skipping pair at {rgb_ts:.3f} — no GPS within ±{gps_threshold_sec}s")
            no_gps_count += 1
            continue

        x, y, z = gps_match[1], gps_match[2], gps_match[3]
        matched_rows.append((rgb_file.name, thm_map[closest_thm_ts].name, x, y, z))

    # Save sorted by time
    matched_rows.sort()
    output_file = os.path.join(output_dir, f"{seq_name}_frame_pairs.txt")
    with open(output_file, "w") as f:
        for rgb_name, thm_name, x, y, z in matched_rows:
            f.write(f"{rgb_name} {thm_name} {x:.6f} {y:.6f} {z:.6f}\n")

print(f"✅ Frame lists with GPS coordinates saved to: {output_dir}")
print(f"❗ No thermal pairings: {no_pair_count}")
print(f"❗ No GPS matches: {no_gps_count}")
