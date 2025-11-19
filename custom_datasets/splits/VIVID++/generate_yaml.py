import os
import yaml
from pathlib import Path

root_dir = Path("/ocean/projects/cis220039p/mdt2/datasets/VIVID++/extracted_data")
default_threshold = 0.2

threshold_map = {}

for traj_group in sorted(os.listdir(root_dir)):
    group_path = root_dir / traj_group
    if not group_path.is_dir():
        continue
    for traj_name in sorted(os.listdir(group_path)):
        traj_path = group_path / traj_name
        if traj_path.is_dir():
            key = f"{traj_group}/{traj_name}"
            threshold_map[key] = float(default_threshold)

with open("thresholds.yml", "w") as f:
    yaml.dump(threshold_map, f, default_flow_style=False)

print("✅ thresholds.yml written with default threshold =", default_threshold)
