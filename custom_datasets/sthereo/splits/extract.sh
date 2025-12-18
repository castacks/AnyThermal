#!/bin/bash

log_file="extract_log.txt"
echo "Extraction started at $(date)" > "$log_file"

for dir in */; do
    echo "Entering directory: $dir"
    cd "$dir" || continue

    for archive in *.tar.gz *.tar; do
        [ -e "$archive" ] || continue  # Skip if no matching files

        echo "Processing: $archive"
        filetype=$(file "$archive")

        if [[ "$archive" == *.tar.gz ]] && echo "$filetype" | grep -q "gzip compressed"; then
            echo "Extracting gzip-compressed tar: $archive"
            if ! tar -xzvf "$archive" >> "../$log_file" 2>&1; then
                echo "❌ Failed to extract $archive" >> "../$log_file"
            fi

        elif [[ "$archive" == *.tar ]] && echo "$filetype" | grep -q "tar archive"; then
            echo "Extracting plain tar: $archive"
            if ! tar -xvf "$archive" >> "../$log_file" 2>&1; then
                echo "❌ Failed to extract $archive" >> "../$log_file"
            fi

        else
            echo "❗ Skipping: $archive (unrecognized or invalid format)"
            echo "⚠️ Skipped invalid archive: $archive ($filetype)" >> "../$log_file"
        fi
    done

    cd ..
done

echo "Extraction completed at $(date)" >> "$log_file"
