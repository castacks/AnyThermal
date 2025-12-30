import os
import argparse
import statistics
from pathlib import Path
from collections import defaultdict
import csv
import yaml

def parse_timestamp(fname):
    try:
        return float(fname.replace(".png", ""))
    except ValueError:
        return None

def load_timestamps(folder):
    timestamps = []
    for fname in sorted(os.listdir(folder)):
        ts = parse_timestamp(fname)
        if ts is not None:
            timestamps.append((ts, fname))
    return timestamps

def load_or_create_thresholds(root_dir, default_thresh=0.2):
    threshold_map = {}
    yaml_path = Path(__file__).parent / "thresholds.yml"

    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            threshold_map = yaml.safe_load(f)
    else:
        print(f"📄 Creating default YAML: {yaml_path}")
        for traj_group in sorted(os.listdir(root_dir)):
            group_path = root_dir / traj_group
            if not group_path.is_dir():
                continue
            for traj_name in sorted(os.listdir(group_path)):
                traj_path = group_path / traj_name
                if traj_path.is_dir():
                    key = f"{traj_group}/{traj_name}"
                    threshold_map[key] = default_thresh

        with open(yaml_path, "w") as f:
            yaml.dump(threshold_map, f, default_flow_style=False)

    return threshold_map

def find_closest_pair(rgb_ts, thermal_ts, threshold, rgb_dir, thermal_dir):
    rgb_ts = sorted(rgb_ts, key=lambda x: x[0])
    thermal_ts = sorted(thermal_ts, key=lambda x: x[0])

    rgb_times = [ts for ts, _ in rgb_ts]
    thermal_times = [ts for ts, _ in thermal_ts]

    if not rgb_times or not thermal_times:
        return [], [], [], 0, 0, 0

    all_seconds = range(int(min(rgb_times[0], thermal_times[0])),
                        int(max(rgb_times[-1], thermal_times[-1])) + 1)

    used_rgb = set()
    used_thermal = set()
    rgb_selected, thermal_selected = [], []
    diffs = []
    skipped_threshold = 0
    skipped_no_pair = 0

    for sec in all_seconds:
        t = float(sec)

        rgb_available = [(i, x) for i, x in enumerate(rgb_ts)
                        if i not in used_rgb and abs(x[0] - t) <= 1.0]
        thermal_available = [(i, x) for i, x in enumerate(thermal_ts)
                            if i not in used_thermal and abs(x[0] - t) <= 1.0]

        if not rgb_available or not thermal_available:
            skipped_no_pair += 1
            continue

        best_pair = None
        best_diff = float('inf')
        for rgb_idx, rgb_data in rgb_available:
            for thermal_idx, thermal_data in thermal_available:
                diff = abs(rgb_data[0] - thermal_data[0])
                if diff <= threshold and diff < best_diff:
                    best_pair = (rgb_idx, rgb_data, thermal_idx, thermal_data)
                    best_diff = diff

        if best_pair is None:
            skipped_threshold += 1
            continue

        rgb_idx, rgb_data, thermal_idx, thermal_data = best_pair
        rgb_selected.append(os.path.abspath(rgb_dir / rgb_data[1]))
        thermal_selected.append(os.path.abspath(thermal_dir / thermal_data[1]))
        diffs.append(best_diff)
        used_rgb.add(rgb_idx)
        used_thermal.add(thermal_idx)

    return rgb_selected, thermal_selected, diffs, len(all_seconds), skipped_no_pair, skipped_threshold

def load_gps_file(gps_file_path):
    gps_entries = []
    # import pdb; pdb.set_trace()
    with open(gps_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(' ')
            # import pdb;pdb.set_trace()
            if len(parts) != 3:
                continue
            try:
                x, y, ts = float(parts[0]), float(parts[1]), float(parts[2])
                gps_entries.append((ts, x, y))
            except ValueError:
                print(f"[ERROR] Invalid GPS entry in {gps_file_path}: {line.strip()}")
                continue
    return sorted(gps_entries, key=lambda x: x[0])

def find_closest_gps(ts_rgb, gps_entries):
    closest = min(gps_entries, key=lambda g: abs(g[0] - ts_rgb))
    if abs(closest[0] - ts_rgb) > 1.0:
        print(f"[ERROR] No close GPS for RGB timestamp {ts_rgb:.3f}. Closest is {closest[0]:.3f} (diff {abs(closest[0] - ts_rgb):.3f}s)")
    return closest[1], closest[2], closest[0]

def process_trajectory(traj_path, out_root, traj_group, traj_name, threshold, csv_records,root_dir_str):
    rgb_dir = traj_path / "img/color"
    thermal_dir = traj_path / "img/thermal_fieldscale_clahe"

    if not rgb_dir.exists() or not thermal_dir.exists():
        print(f"Skipping {traj_group}/{traj_name} due to missing folders.")
        return

    rgb_ts = load_timestamps(rgb_dir)
    thermal_ts = load_timestamps(thermal_dir)

    if not rgb_ts or not thermal_ts:
        print(f"Skipping {traj_group}/{traj_name} due to empty RGB or thermal image list.")
        return

    rgb_list, thermal_list, diffs, total_secs, skipped_no_pair, skipped_threshold = find_closest_pair(
        rgb_ts, thermal_ts, threshold, rgb_dir, thermal_dir
    )

    gps_file_path = traj_path / "gpslist.txt"
    gps_entries = load_gps_file(gps_file_path) if gps_file_path.exists() else []

    rgb_gps_coords = []
    for r_path in rgb_list:
        ts_rgb = parse_timestamp(Path(r_path).name)
        if gps_entries:
            x, y, ts = find_closest_gps(ts_rgb, gps_entries)
            rgb_gps_coords.append((x, y, ts))
        else:
            print(f"[WARNING] No GPS entries found for {traj_group}/{traj_name}.")
            rgb_gps_coords.append((None, None, None))
    
    thermal_gps_coords = []
    for t_path in thermal_list:
        ts_thermal = parse_timestamp(Path(t_path).name)
        if gps_entries:
            x, y, ts = find_closest_gps(ts_thermal, gps_entries)
            thermal_gps_coords.append((x, y, ts))
        else:
            print(f"[WARNING] No GPS entries found for {traj_group}/{traj_name}.")
            thermal_gps_coords.append((None, None, None))

    out_dir = Path(out_root) / traj_group / traj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "rgb_framelist.txt", "w") as f_rgb, \
         open(out_dir / "thermal_framelist.txt", "w") as f_thermal, \
         open(out_dir / "rgb_gps_framelist.txt", "w") as f_rgb_gps, \
         open(out_dir / "thermal_gps_framelist.txt", "w") as f_thermal_gps:
        for r, t, rgb_gps,thermal_gps in zip(rgb_list, thermal_list, rgb_gps_coords, thermal_gps_coords):
            r = r.replace(root_dir_str+"/","")
            t = t.replace(root_dir_str+"/","")
            f_rgb.write(f"{r}\n")
            f_thermal.write(f"{t}\n")
            f_rgb_gps.write(f"{rgb_gps[0]} {rgb_gps[1]} {rgb_gps[2]}\n")
            f_thermal_gps.write(f"{thermal_gps[0]} {thermal_gps[1]} {thermal_gps[2]}\n")

    if diffs:
        min_diff = min(diffs)
        max_diff = max(diffs)
        mean_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    else:
        min_diff = max_diff = mean_diff = std_diff = None

    csv_records.append({
        "trajectory": f"{traj_group}/{traj_name}",
        "total_seconds": total_secs,
        "matched_pairs": len(diffs),
        "skipped_no_pair": skipped_no_pair,
        "skipped_threshold": skipped_threshold,
        "min_diff": min_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "max_diff": max_diff
    })

    print(f"[{traj_group}/{traj_name}] Pairs: {len(diffs)} / {total_secs} | NoPair: {skipped_no_pair}, ThresholdFail: {skipped_threshold}")

def main():
    parser = argparse.ArgumentParser(description="Generate 1Hz frame lists from RGB and Thermal images.")
    parser.add_argument("--root_dir", default="/ocean/projects/cis220039p/mdt2/datasets/VIVID++/extracted_data", type=str)
    parser.add_argument("--threshold", type=float, default=0.2, help="Default threshold if YAML not found.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_root = "frame_lists"
    script_dir = Path(__file__).parent
    csv_path = script_dir / f"pair_stats_thresholds_used.csv"

    threshold_map = load_or_create_thresholds(root_dir, args.threshold)
    csv_records = []

    for traj_group in sorted(os.listdir(root_dir)):
        group_path = root_dir / traj_group
        if not group_path.is_dir():
            continue
        for traj_name in sorted(os.listdir(group_path)):
            traj_path = group_path / traj_name
            if not traj_path.is_dir():
                continue

            traj_key = f"{traj_group}/{traj_name}"
            threshold = threshold_map.get(traj_key, args.threshold)
            process_trajectory(traj_path, out_root, traj_group, traj_name, threshold, csv_records,args.root_dir)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trajectory", "total_seconds", "matched_pairs",
            "skipped_no_pair", "skipped_threshold",
            "min_diff", "mean_diff", "std_diff", "max_diff"
        ])
        writer.writeheader()
        writer.writerows(csv_records)

    print(f"\n📄 Stats written to: {csv_path}")

if __name__ == "__main__":
    main()
