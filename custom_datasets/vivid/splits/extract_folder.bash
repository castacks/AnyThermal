#!/bin/bash

# List of folders to process
# folders=("driving_full" "driving_vision" "handheld_indoor" "handheld_outdoor")
# folders=("driving_full" "driving_vision" "handheld_outdoor")

# folders=("handheld_indoor")
folder="$1"
# Path to your Python script
# python_script="process_dvs.py"

# Iterate over folders
# for folder in "${folders[@]}"; do
echo "Processing folder: $folder"

# Find all .bag files in the folder (non-recursively; change if recursive search is needed)
find "$folder" -maxdepth 1 -name "*.bag" | while read -r bag_file; do
    bag_name=$(basename "$bag_file")
    python_script="process_img.py"
    echo "Running: python3 $python_script $folder $bag_name"
    python3 "$python_script" "$folder" "$bag_name"
    # python_script="process_dvs.py"
    # echo "Running: python3 $python_script $folder $bag_name"
    # python3 "$python_script" "$folder" "$bag_name"

done
# done
