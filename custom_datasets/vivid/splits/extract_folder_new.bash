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
python_script="process_img_parallel.py"
python3 "$python_script" "$folder" 
# python_script="process_dvs.py"
# echo "Running: python3 $python_script $folder $bag_name"
# python3 "$python_script" "$folder" "$bag_name"

