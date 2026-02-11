#!/bin/bash

# Base URL for the S3 bucket
# BASE_URL=""https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data"

# List of .bag" files to download
bag_files=(
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2021-07-21_Beach/2021-07-21-09-27-15/2021-07-21-09-27-15.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2021-07-21_Beach/2021-07-21-09-33-49/2021-07-21-09-33-49.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2021-09-09_KentuckyRiver/2021-09-09-14-11-10/2021-09-09-14-11-10.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2021-09-09_KentuckyRiver/2021-09-09-14-31-14/2021-09-09-14-31-14.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2021-09-09_KentuckyRiver/2021-09-09-14-42-06/2021-09-09-14-42-06.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-01-08_ArroyoSeco/2022-01-08-14-32-58/2022-01-08-14-32-58.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-01-08_ArroyoSeco/2022-01-08-14-40-06/2022-01-08-14-40-06.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-01-08_ArroyoSeco/2022-01-08-14-43-57/2022-01-08-14-43-57.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-01-08_ArroyoSeco/2022-01-08-14-46-23/2022-01-08-14-46-23.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-01-08_ArroyoSeco/2022-01-08-15-21-59/2022-01-08-15-21-59.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-01-08_ArroyoSeco/2022-01-08-15-23-26/2022-01-08-15-23-26.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-01-08_ArroyoSeco/2022-01-08-15-29-41/2022-01-08-15-29-41.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-01-08_ArroyoSeco/2022-01-08-15-31-20/2022-01-08-15-31-20.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-04-03_Idyllwild/2022-04-03-12-16-33/2022-04-03-12-16-33.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-04-03_Idyllwild/2022-04-03-12-20-57/2022-04-03-12-20-57.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-04-03_JoshuaTree/2022-04-03-17-12-07/2022-04-03-17-12-07.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-04-03_JoshuaTree/2022-04-03-17-16-17/2022-04-03-17-16-17.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-08_BigBear/2022-05-08-11-23-59/2022-05-08-11-23-59.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-08_BigBear/2022-05-08-11-27-13/2022-05-08-11-27-13.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-08_BigBear/2022-05-08-11-30-40/2022-05-08-11-30-40.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-08_BigBear/2022-05-08-11-34-00/2022-05-08-11-34-00.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-08_BigBear/2022-05-08-11-37-09/2022-05-08-11-37-09.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-08_BigBear/2022-05-08-11-40-44/2022-05-08-11-40-44.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-08_BigBear/2022-05-08-11-44-33/2022-05-08-11-44-33.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-08_BigBear/2022-05-08-11-50-48/2022-05-08-11-50-48.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-15_ColoradoRiver/2022-05-15-06-00-09/2022-05-15-06-00-09.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-15_ColoradoRiver/2022-05-15-06-14-42/2022-05-15-06-14-42.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-15_ColoradoRiver/2022-05-15-06-26-50/2022-05-15-06-26-50.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-05-15_ColoradoRiver/2022-05-15-06-39-43/2022-05-15-06-39-43.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-07-26_NorthField/2022-07-26-10-39-11/2022-07-26-10-39-11.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-07-26_NorthField/2022-07-26-10-50-52/2022-07-26-10-50-52.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-07-26_NorthField/2022-07-26-11-00-21/2022-07-26-11-00-21.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-07-26_NorthField/2022-07-26-11-05-36/2022-07-26-11-05-36.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-07-26_NorthField/2022-07-26-11-22-00/2022-07-26-11-22-00.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-10-06_NorthField/2022-10-06-12-23-29/2022-10-06-12-23-29.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-10-06_NorthField/2022-10-06-12-59-02/2022-10-06-12-59-02.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-10-06_NorthField/2022-10-06-13-11-26/2022-10-06-13-11-26.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-12-20_CastaicLake/2022-12-20-11-40-28/2022-12-20-11-40-28.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-12-20_CastaicLake/2022-12-20-12-16-02/2022-12-20-12-16-02.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-12-20_CastaicLake/2022-12-20-12-48-59/2022-12-20-12-48-59.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2022-12-20_CastaicLake/2022-12-20-13-37-37/2022-12-20-13-37-37.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2023-03-XX_Duck/2023-03-21-09-59-39/2023-03-21-09-59-39.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2023-03-XX_Duck/2023-03-21-14-06-04/2023-03-21-14-06-04.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2023-03-XX_Duck/2023-03-21-18-20-21/2023-03-21-18-20-21.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2023-03-XX_Duck/2023-03-21-19-55-11/2023-03-21-19-55-11.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2023-03-XX_Duck/2023-03-22-08-44-31/2023-03-22-08-44-31.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2023-03-XX_Duck/2023-03-22-14-31-06/2023-03-22-14-31-06.bag"
"https://renc.osn.xsede.org/ini210004tommorrell/cks6g-ps927/data/2023-03-XX_Duck/2023-03-22-14-41-46/2023-03-22-14-41-46.bag"

)

# Create a directory to store downloaded bag files
mkdir -p bag_files
cd bag_files || exit

# Download each file
for file in "${bag_files[@]}"; do
    echo "Downloading $file ..."
    wget -c "$file"
done

echo "All downloads complete."
