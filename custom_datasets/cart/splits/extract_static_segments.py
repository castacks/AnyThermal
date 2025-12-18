import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
def compute_velocity_magnitude(df):
    vx = df["twist.twist.linear.x"]
    vy = df["twist.twist.linear.y"]
    vz = df["twist.twist.linear.z"]
    return np.sqrt(vx**2 + vy**2 + vz**2)

def find_static_segments(times, velocities, threshold):
    is_static = velocities < threshold
    static_segments = []
    start_idx = None

    for i, static in enumerate(is_static):
        if static and start_idx is None:
            start_idx = i
        elif not static and start_idx is not None:
            static_segments.append((start_idx, i - 1))
            start_idx = None

    if start_idx is not None:
        static_segments.append((start_idx, len(times) - 1))

    # Convert index segments to time segments
    time_segments = [(times[start], times[end]) for start, end in static_segments]
    return is_static, time_segments, static_segments

def plot_and_save(times, velocities, segments, traj_name, output_dir):
    plt.figure(figsize=(12, 4))
    plt.plot(times, velocities, label="Velocity Magnitude", color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Trajectory: {traj_name}")
    plt.grid(True)

    for (start, end) in segments:
        plt.axvspan(start, end, color='red', alpha=0.3)

    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{traj_name}_velocity_plot.png")
    plt.savefig(out_path)
    plt.close()

def process_trajectory(root_dir, traj_name, threshold, gps_sampling_rate, output_dir):
    csv_path = os.path.join(root_dir, traj_name, "csv", "_gps_fix_velocity.csv")
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        return [], []

    df = pd.read_csv(csv_path)
    df = df[::gps_sampling_rate].reset_index(drop=True)

    required_cols = ["twist.twist.linear.x", "twist.twist.linear.y", "twist.twist.linear.z", "Time"]
    if not all(k in df.columns for k in required_cols):
        print(f"Required columns missing in {csv_path}")
        return [], []

    times = df["Time"].values
    velocities = compute_velocity_magnitude(df)
    _, static_time_segments, static_index_segments = find_static_segments(times, velocities, threshold)

    plot_and_save(times, velocities, static_time_segments, traj_name, output_dir)

    # # Save index segments as TXT
    # index_txt_path = os.path.join(output_dir, f"{traj_name}_static_frame_indices.txt")
    # with open(index_txt_path, "w") as f:
    #     for start, end in static_index_segments:
    #         f.write(f"{start},{end}\n")

    return static_time_segments, static_index_segments

def read_trajectories_from_file(file_path):
    # import pdb; pdb.set_trace()
    traj_list = yaml.safe_load(open(file_path, "r"))["traj_list"].keys()
    return list(traj_list)
    # with open(file_path, "r") as f:
    #     return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files")
    parser.add_argument("--trajectory_file", type=str, default = "trajectories.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gps_sampling", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./static_segments_output/velocity_segments")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trajs = read_trajectories_from_file(args.trajectory_file)
    all_segments = {}
    for traj in trajs:
        time_segments, _ = process_trajectory(
            args.root_dir,
            traj,
            args.threshold,
            args.gps_sampling,
            args.output_dir
        )
        all_segments[traj] = time_segments
        print(f"{traj}: {len(time_segments)} static segments")

    # Save dictionary of time-based segments
    with open(os.path.join(args.output_dir, "static_segments.pkl"), "wb") as f:
        pickle.dump(all_segments, f)

if __name__ == "__main__":
    main()
