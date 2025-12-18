import os
import pandas as pd
import numpy as np
import argparse
import yaml
from pyproj import Transformer
from tqdm import tqdm
def load_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    return pd.read_csv(csv_path)

def find_closest_gps_match(gps_df, thermal_time):
    idx = (gps_df["Time"] - thermal_time).abs().idxmin()
    return gps_df.loc[idx]

def get_utm_transformer(zone_number):
    # UTM North zones are EPSG 326## (e.g., zone 33N → EPSG:32633)
    epsg_code = f"326{int(zone_number):02d}"
    return Transformer.from_crs("epsg:4326", f"epsg:{epsg_code}", always_xy=True)

def process_trajectory(traj, root_dir, zone_number):
    traj_path = os.path.join(root_dir, traj, "csv")
    output_dir = traj_path

    gps_csv = os.path.join(traj_path, "_gps_fix.csv")
    thermal_csv = os.path.join(traj_path, "_boson_thermal_image_raw.csv")

    gps_df = load_csv(gps_csv)
    thermal_df = load_csv(thermal_csv)

    if not all(k in gps_df.columns for k in ["latitude", "longitude", "altitude", "Time"]):
        print(f"GPS file {gps_csv} missing required columns")
        return

    transformer = get_utm_transformer(zone_number)

    thermal_times = thermal_df["Time"].values
    utm_coords = []
    results = []

    for thermal_idx, t in enumerate(thermal_times):
        gps_row = find_closest_gps_match(gps_df, t)
        lat, lon, alt = gps_row["latitude"], gps_row["longitude"], gps_row["altitude"]
        easting, northing = transformer.transform(lon, lat)
        utm_coords.append([easting, northing,alt])
        results.append({
            "Thermal_image_index": thermal_idx,
            "Time": t,
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "easting": easting,
            "northing": northing,
            "utm_zone": f"{zone_number}N"
        })

    out_df = pd.DataFrame(results)

    out_csv = os.path.join(output_dir, "thermal_utm_coords.csv")
    out_npy = os.path.join(output_dir, "thermal_utm_coords.npy")

    # Remove existing outputs if they exist
    if os.path.exists(out_csv):
        os.remove(out_csv)
    if os.path.exists(out_npy):
        os.remove(out_npy)

    out_df.to_csv(out_csv, index=False)
    np.save(out_npy, np.array(utm_coords))

    print(f"{traj}: saved {len(utm_coords)} coords →\n  {out_csv}\n  {out_npy}")

def read_trajectory_list(file_path):
    with open(file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    # return [(name, traj_meta["zone"]) for name, traj_meta in yaml_data["traj_list"].items()]
    return yaml_data["traj_list"]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/ocean/projects/cis220039p/mdt2/shared/CART/bag_files", help="Root directory with trajectory folders.")
    parser.add_argument("--trajectory_file", type=str, default="trajectories.yaml", help="YAML file listing trajectory names and UTM zones.")
    args = parser.parse_args()

    trajs_dict = read_trajectory_list(args.trajectory_file)

    for traj_name in tqdm(trajs_dict.keys()):
        traj_meta = trajs_dict[traj_name]
        # import pdb; pdb.set_trace()
        zone_number = traj_meta["zone"]
        gps_available = traj_meta["gps"]
        if not gps_available:
            print(f"Skipping {traj_name} due to missing GPS data.")
            continue
        process_trajectory(traj_name, args.root_dir, zone_number)

if __name__ == "__main__":
    main()
