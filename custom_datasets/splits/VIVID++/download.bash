#!/bin/bash

# Create the output directory
mkdir -p driving_vision
cd driving_vision || exit 1

# List of URLs to download
urls=(
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_day1.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_day2.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_day2_2.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_evening.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_morning.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_morning_2.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_morning_manual.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_morning_manual_small.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_night.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/campus_night_2.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/city_day1.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/city_day2.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/city_evening.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/city_morning.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/city_morning_manual.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/city_night.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/urban_day.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/urban_evening.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/urban_evening_road.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/urban_morning.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/urban_morning_manual.bag"
"https://urserver.kaist.ac.kr/publicdata/ViViD++/driving_vision/urban_night.bag"
)

# Download each file using wget with resume support
for url in "${urls[@]}"; do
    echo "Downloading: $url"
    wget -c "$url" &
done

# Wait for all parallel downloads to finish
wait

echo "✅ All files downloaded to ./driving_vision/"
