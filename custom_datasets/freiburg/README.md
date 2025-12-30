# Freigburg Post Processing

This folder contains scripts to download and post processs the freiburg dataset to use for the AnyThermal project. 

## Downloading the Data

Follow [this](https://github.com/JohanVer/heatnet?tab=readme-ov-file#dataset-preparation) to download the freiburg dataset.

Replace the `freigburg` path in the `<ANYTHERMAL_ROOT_DIR>/custom datasets/datasdet_path.yaml` file to point to the root location of the extracted data.

For example, we extract our train.zip and test.zip at `/ocean/projects/cis220039p/mdt2/datasets/freiburg/train` and `/ocean/projects/cis220039p/mdt2/datasets/freiburg/test` and set the root path to `/ocean/projects/cis220039p/mdt2/datasets/freiburg/` in <ANYTHERMAL_ROOT_DIR>/custom_datasets/dataset_path.yaml.

## Post Processing the Data

To post process the data, run the following command:

```bash
cd <ANYTHERMAL_ROOT_DIR>/custom_datasets/freiburg/splits

python3 generate_frame_list.py --train_dir <PATH_TO_FREIBURG_DATASET_TRAIN_FOLDER> --skip_short_intervals

python3 thermal16_to_thermal8.py --root_dir <PATH_TO_FREIBURG_DATASET_TRAIN_FOLDER> --clahe
```


This will generate the necessary frame lists and save 8-bit thermal in each trajectory under the `thermal8_clahe` folder.