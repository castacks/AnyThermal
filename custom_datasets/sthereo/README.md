# STheReO

This folder contains scripts to download and post processs the STheReO dataset to use for the AnyThermal project.

## Downloading the Data

Follow [this link](https://sites.google.com/view/rpmsthereo/download?authuser=0) to download the STheReO dataset.

Once downloaded use the `extract.sh` script to extract the data. For example, your directory structure should look like the following 
```
<ROOT_DATASET_PATH>/
├── kaist_morning/
├────── calibration.tar.gz 
├────── image.tar
├────── pose.tar
├────── sensor_data.tar
├── kaist_afternoon/
├── kaist_evening/
├── snu_afternoon/
├── snu_evening/
├── valley_morning/
├── valley_afternoon/
├── valley_evening/
```

if you have downloaded and extracted the dataset to `/ocean/projects/cis220039p/mdt2/datasets/sthereo/` run the following commands:

```bash
cd <ANYTHERMAL_ROOT_DIR>/custom_datasets/sthereo/splits 
bash extract.sh /ocean/projects/cis220039p/mdt2/datasets/sthereo/
```

This will extract all the zip files in their parent folder

## Post Processing the Data

Run the following command to post process the data:

```bash
cd <ANYTHERMAL_ROOT_DIR>/custom_datasets/sthereo/splits
python3 thermal14to8.py --root_dir <PATH_TO_STHEREO_DATASET_FOLDER>
```

## DATASET_PATH.YAML

Use the root directory (where you extracted the dataset) path in the `<ANYTHERMAL_ROOT_DIR>/custom_datasets/dataset_path.yaml` file to point to the root location of the extracted data. Replace the `sthereo` path in the file with your chosen path.



