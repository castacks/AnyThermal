# OdomBeyondVision

This folder contains scripts to download and post processs the OdomBeyondVision (OBV) dataset to use for the AnyThermal project.

## Downloading the Data

The data could be originally downloaded following the [official github link](https://github.com/MAPS-Lab/OdomBeyondVision) however as of Dec 29, 2025 the original link seems to be down.

If the dataset can be downloaded again download the data to follow the following directory structure 

```
<path to OdomBeyondVision>
├── Rosbags
    ├── Handheld
    ├── UAV
    └── UGV
.
.
.
```

## Extracting the data

To extract the data, run the following command:

```bash
cd <ANYTHERMAL_ROOT_DIR>/custom_datasets/obv/splits
python3 extract_images_odom_files.py --root_dir <PATH_TO_OBV_DATASET_ROOT_FOLDER>
```

## Setting up the dataset path

Replace the `obv` path in the `<ANYTHERMAL_ROOT_DIR>/custom_datasets/dataset_path.yaml` file to point to the root location of the extracted data `<PATH_TO_OBV_DATASET_ROOT_FOLDER>/ExtractedData`