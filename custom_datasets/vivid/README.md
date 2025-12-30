# Preparing VIVID++ Dataset for AnyThermal

This folder contains scripts to extract and prepare the VIVID++ dataset for use with the AnyThermal project.

## Downloading the Data

Follow the instructions on the [VIVID++ dataset website](https://visibilitydataset.github.io/4_download.html?highlight=process_img#request-downloads) to download the `driving_vision` and `driving_full` datasets. Make sure to request access and download the datasets to your desired location.

For example we use `/ocean/projects/cis220039p/mdt2/datasets/VIVID++` as the root directory to store the downloaded datasets. The datasets are stored in the following structure:

```
/your/chosen/path/VIVID++/
    driving_vision/
        <rosbag files>
    driving_full/
        <rosbag files>

## Extracting and Post Processing the Data
```bash 
source /opt/ros/noetic/setup.bash
export VIVID_ROOT_DIR=/ocean/projects/cis220039p/mdt2/datasets/VIVID++
bash extract_rosbag_folder.bash $VIVID_ROOT_DIR driving_vision
bash extract_rosbag_folder.bash $VIVID_ROOT_DIR driving_full
```

This will extract all the rosbags in the specified folder and save the extracted data in the folder `$VIVID_ROOT_DIR/extracted_data`.

## Setting the Dataset Path

Finally, update the `<ANYTHERMAL_ROOT_DIR>/custom_datasets/dataset_path.yaml` file to point to the root location of the extracted data for the `vivid` dataset.