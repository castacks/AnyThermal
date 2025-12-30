import os
import argparse
from pathlib import Path

def parse_freiburg_timestamp(filename):
    parts = Path(filename).stem.split('_')
    try:
        seconds = int(parts[-2])
        nanoseconds = int(parts[-1][:9])
        return seconds + nanoseconds * 1e-9
    except Exception:
        return None

def filter_sparse_frames(timestamps, files, min_interval=1.0):
    if not timestamps:
        return []
    filtered = [(timestamps[0], files[0])]
    last_time = timestamps[0]
    for ts, f in zip(timestamps[1:], files[1:]):
        if ts - last_time >= min_interval:
            filtered.append((ts, f))
            last_time = ts
    return filtered

def analyze_frame_intervals(fl_rgb_dir, fl_thermal_dir, skip_short_intervals=True, min_interval=1.0):
    rgb_files = sorted(fl_rgb_dir.glob("*.png"))
    thermal_files = sorted(fl_thermal_dir.glob("*.png"))
    timestamps = [parse_freiburg_timestamp(f.name) for f in rgb_files]
    timestamps = [(ts, f) for ts, f in zip(timestamps, rgb_files) if ts is not None]

    if skip_short_intervals:
        timestamps = filter_sparse_frames(
            [ts for ts, _ in timestamps], [f for _, f in timestamps], min_interval=min_interval
        )

    if len(timestamps) < 2:
        return None, len(rgb_files), len(timestamps), [], []

    diffs = [timestamps[i+1][0] - timestamps[i][0] for i in range(len(timestamps)-1)]
    avg_diff = sum(diffs) / len(diffs)
    min_diff = min(diffs)
    max_diff = max(diffs)

    used_rgb_files = [f for _, f in timestamps]
    used_thermal_files = [fl_thermal_dir / f.name for f in used_rgb_files]  # Match by filename
    used_thermal_files = [Path(str(x).replace("fl_rgb", "fl_ir_aligned")) for x in used_thermal_files]
    return (avg_diff, min_diff, max_diff), len(rgb_files), len(timestamps), used_rgb_files, used_thermal_files

def save_frame_list(file_list, out_path):
    with open(out_path, 'w') as f:
        for file in file_list:
            import re
            f.write(str(file) + "\n")

def scan_train_dir(train_root, skip_short_intervals, min_interval=1.0):
    train_root = Path(train_root)
    global_diffs = []

    print(f"{'Trajectory':40s} | {'Total':>5} | {'Used':>5} | {'Avg (s)':>8} | {'Min (s)':>8} | {'Max (s)':>8}")
    print("-" * 90)

    global_total_frames = 0
    global_used_frames = 0

    for traj in sorted(train_root.glob("seq_*")):
        for sub in sorted(traj.glob("*")):
            fl_rgb_dir = sub / "fl_rgb"
            fl_thermal_dir = sub / "fl_ir_aligned"
            print(f"Processing {fl_rgb_dir} and {fl_thermal_dir}...")
            if fl_rgb_dir.is_dir() and fl_thermal_dir.is_dir():
                result, total_frames, used_frames, rgb_list, thermal_list = analyze_frame_intervals(
                    fl_rgb_dir, fl_thermal_dir, skip_short_intervals, min_interval
                )
                # import pdb; pdb.set_trace()  # Debugging breakpoint
                global_total_frames += total_frames
                global_used_frames += used_frames
                name = f"{traj.name}/{sub.name}"
                if result:
                    avg, min_d, max_d = result
                    global_diffs.append(avg)
                    print(f"{name:40s} | {total_frames:5d} | {used_frames:5d} | {avg:8.3f} | {min_d:8.3f} | {max_d:8.3f}")
                else:
                    print(f"{name:40s} | {total_frames:5d} | {used_frames:5d} | {'-':>8} | {'-':>8} | {'-':>8}")

                if used_frames > 0:
                    out_prefix = "_".join([train_root.name, traj.name, sub.name])
                    save_frame_list(rgb_list, f"absolute_frame_lists/{out_prefix}.txt")
                    save_frame_list(thermal_list, f"absolute_frame_lists/{out_prefix}_thermal.txt")

                    rgb_frame_only_list = [f.name for f in rgb_list]
                    rgb_frame_only_list = [f.replace("fl_rgb_","") for f in rgb_frame_only_list]
                    save_frame_list(rgb_frame_only_list, f"frame_list/{out_prefix}.txt")
                    # thermal_frame_only_list = [f.name for f in thermal_list]
                    # save_frame_list(thermal_frame_only_list, f"frame_list/{out_prefix}_thermal.txt")


    if global_diffs:
        global_avg = sum(global_diffs) / len(global_diffs)
        print(f"\n🌍 Global average interval (after filtering): {global_avg:.3f} s")
    else:
        print("\n⚠️ No valid intervals found.")

    print(f"Total frames across all trajectories: {global_total_frames}")
    print(f"Total used frames across all trajectories: {global_used_frames}")

# Run as script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/ocean/projects/cis220039p/mdt2/datasets/freiburg/train", help="Path to Freiburg 'train' folder")
    parser.add_argument("--skip_short_intervals", action="store_true", help="Skip frames < 1s apart")
    parser.add_argument("--min_interval", type=float, default=1.0, help="Minimum interval to consider for filtering (in seconds)")
    args = parser.parse_args()

    os.makedirs("absolute_frame_lists", exist_ok=True)
    os.makedirs("frame_list", exist_ok=True)

    scan_train_dir(args.train_dir, skip_short_intervals=args.skip_short_intervals, min_interval=args.min_interval)
