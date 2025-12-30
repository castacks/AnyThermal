#!/bin/bash

root_dir="$1"
folder="$2"
echo "Processing folder: $folder"
python_script="process_img_parallel.py"
python3 "$python_script" "$folder" "$root_dir"