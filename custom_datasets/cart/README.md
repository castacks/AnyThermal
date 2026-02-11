# CART

This folder contains scripts to process the CART dataset for use with AnyThermal.

## Dataset Download

Follow [this](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset/tree/main?tab=readme-ov-file#dataset-download) to download the dataset. 

To download all the rosbags at once - we aosr provide a `splits/download_bags.sh` script that uses `wget` to download all the rosbags. Place taht script in your desired download location and execute it.

The segmentation labels will ahve to be still downlaoded manually from the above link.


## Dataset Extraction 

For extracting raw rosbags see [this](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset/tree/main?tab=readme-ov-file#data-extraction)


## Dataset Processing
To process the dataset, first navigate to the `splits` directory:
```bash 
cd <AnyThermal_ROOT>/custom_datasets/cart/splits
python3 rotate180.py --root_dir /ocean/projects/cis220039p/mdt2/shared/CART/bag_files
python3 thermal_to_utm_from_gps.py --root_dir /ocean/projects/cis220039p/mdt2/shared/CART/bag_files
```

Next, to stereo rectify thermal and RGB images along with converting 16 to 8 bit for thermal follow [this](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset/tree/main?tab=readme-ov-file#rectification)


## Dataset diectory Structure

```
ROOT_PATH/TO/CART/
├── bag_files/
    ├── 2021-07-21-09-27-15.bag
    ├── 2021-07-21-09-33-49.bag
    .
    .
    .
    2022-05-15-06-14-42/
        csv/
            .
            .
            .
            thermal_utm_coords.csv
        images
        stereo_rectified/
            thermal_color/
                eo/
                overlay/
                sxs/
                thermal/
                thermal8/
            themral_mono

├── labeled_rgbt_pairs/
├── labeled_thermal_singles/
```
bag_files : contains all the raw rosbags and their repective extracted folders 


## Dataset Path for AnyThermal

When using the CART dataset with AnyThermal, set the dataset path to the root directory of the CART dataset (i.e., the directory containing the `bag_files`, `labeled_rgbt_pairs`, and `labeled_thermal_singles` folders) in the `custom_datasets/dataset_path.yaml` in the `cart` key.
