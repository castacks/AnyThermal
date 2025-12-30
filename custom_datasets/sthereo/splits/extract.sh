#!/bin/bash
set -e

# =========================
# Usage check
# =========================
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ROOT_DATASET_PATH>"
    exit 1
fi

ROOT_DIR="$1"

if [ ! -d "$ROOT_DIR" ]; then
    echo "❌ Root directory does not exist: $ROOT_DIR"
    exit 1
fi

# =========================
# Configuration
# =========================
log_file="$ROOT_DIR/extract_log.txt"
echo "Extraction started at $(date)" > "$log_file"

# Expected top-level folders
DATASET_FOLDERS=(
    kaist_morning
    kaist_afternoon
    kaist_evening
    snu_afternoon
    snu_evening
    valley_morning
    valley_afternoon
    valley_evening
)

# =========================
# Extraction loop
# =========================
for dataset in "${DATASET_FOLDERS[@]}"; do
    DATASET_PATH="$ROOT_DIR/$dataset"

    if [ ! -d "$DATASET_PATH" ]; then
        echo "⚠️ Skipping missing folder: $DATASET_PATH" | tee -a "$log_file"
        continue
    fi

    echo "📂 Processing dataset: $DATASET_PATH" | tee -a "$log_file"

    # Find tar / tar.gz files recursively
    find "$DATASET_PATH" -type f \( -name "*.tar" -o -name "*.tar.gz" \) | while read -r archive; do
        echo "📦 Found archive: $archive" | tee -a "$log_file"

        filetype=$(file "$archive")

        if [[ "$archive" == *.tar.gz ]] && echo "$filetype" | grep -q "gzip compressed"; then
            echo "➡️ Extracting gzip tar: $archive" | tee -a "$log_file"
            tar -xzvf "$archive" -C "$(dirname "$archive")" >> "$log_file" 2>&1 \
                || echo "❌ Failed to extract $archive" | tee -a "$log_file"

        elif [[ "$archive" == *.tar ]] && echo "$filetype" | grep -q "tar archive"; then
            echo "➡️ Extracting tar: $archive" | tee -a "$log_file"
            tar -xvf "$archive" -C "$(dirname "$archive")" >> "$log_file" 2>&1 \
                || echo "❌ Failed to extract $archive" | tee -a "$log_file"

        else
            echo "⚠️ Skipping invalid archive: $archive ($filetype)" | tee -a "$log_file"
        fi
    done
done

echo "✅ Extraction completed at $(date)" | tee -a "$log_file"
