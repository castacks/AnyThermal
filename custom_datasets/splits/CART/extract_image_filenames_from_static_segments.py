import os
import argparse
import pandas as pd
import pickle
import yaml
def load_csv_subset(csv_path, sampling_rate):
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df = df[::sampling_rate].reset_index(drop=True)
    return df

def find_index_for_time(df, timestamp):
    return (df["Time"] - timestamp).abs().idxmin()


def extract_static_segments(output_dir,df, static_segments,traj_name,modality="thermal"):

    lines = []
    for start, end in static_segments:
        start_idx = find_index_for_time(df, start)
        end_idx = find_index_for_time(df, end)

        if start_idx is None or end_idx is None:
            print(f"Missing data for {traj_name} at {start} - {end}")
            return []

        start_row = df.loc[start_idx]
        end_row = df.loc[end_idx]
        lines.append(f"{start_row['filename'].split('/')[-1]},{end_row['filename'].split('/')[-1]},{start_row['Time']:.6f},{end_row['Time']:.6f}")
    
    txt_file =  os.path.join(output_dir, f"{traj_name}_{modality}_frame_indices.txt")
    with open(txt_file, "w") as f:
        f.write("start_idx,end_idx,start_time,end_time\n")
        f.write("\n".join(lines) + "\n")

    

def process_trajectory(root_dir, traj_name, static_segments, rgb_rate, thermal_rate, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(root_dir, traj_name, "csv")
    rgb_csv = os.path.join(base, "_eo_color_image_color_compressed.csv")
    thermal_csv = os.path.join(base, "_boson_thermal_image_raw.csv")

    rgb_df = load_csv_subset(rgb_csv, rgb_rate)
    thr_df = load_csv_subset(thermal_csv, thermal_rate)
    
    # if rgb or thermal data is missing, return empty result
    # if rgb_df.empty or thr_df.empty:
    #     print(f"Missing data for trajectory {traj_name}: RGB or thermal CSV is empty.")
    #     return []

    if not thr_df.empty:
        extract_static_segments(output_dir,thr_df, static_segments, traj_name, modality="thermal")
    else:
        print(f"Missing thermal data for trajectory {traj_name}. Skipping thermal segment extraction.")
    
    # if not rgb_df.empty:
    # #     extract_static_segments(output_dir,rgb_df, static_segments, traj_name, modality="rgb")
    # # else:z
    #     print(f"Missing RGB data for trajectory {traj_name}. Skipping RGB segment extraction.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files")
    parser.add_argument("--segments_file", type=str, default = "./static_segments_output/velocity_segments/static_segments.pkl")
    parser.add_argument("--output_file", type=str, default = "./static_segments_output/image_filenames/image_filenames.pkl")
    parser.add_argument("--rgb_rate", type=int, default=30)
    parser.add_argument("--thermal_rate", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="./static_segments_output/image_filenames")
    args = parser.parse_args()

    with open(args.segments_file, "rb") as f:
        static_segments = pickle.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    all_filenames = {}

    for traj_name, segments in static_segments.items():
        process_trajectory(
            args.root_dir, traj_name, segments,
            args.rgb_rate, args.thermal_rate,
            args.output_dir
        )
        # all_filenames[traj_name] = result

    # with open(args.output_file, "wb") as f:
    #     pickle.dump(all_filenames, f)

if __name__ == "__main__":
    main()
