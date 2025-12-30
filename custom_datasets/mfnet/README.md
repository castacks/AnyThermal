# MFNet

To use the MFNet dataset with AnyThermal, follow the instructions below.

## Downloading the Dataset

Download the MFNet dataset from the [official google drive folder](https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw).

Choose a location on your local machine where you want to store the dataset. We will refer to this location as `ROOT_PATH/TO/MFNET`.

After downloading and extracting the dataset, ensure that the directory structure looks like this:

```
ROOT_PATH/TO/MFNET/
├── anno_json/
├── images/
├── labels/
├── visual/
│
├── black_list.txt
├── make_flip.py
├── README.txt
│
├── train.txt
├── val.txt
├── test.txt
├── test_day.txt
└── test_night.txt
```

## Preparing the Dataset

Before using the dataset, you need to prepare it by running the provided script to create flipped images.

```bash
cd ROOT_PATH/TO/MFNET
python make_flip.py
```

This will generate flipped versions of the images and save them in the appropriate directories.

## Setting the dataset_path.yaml

When configuring AnyThermal to use the MFNet dataset, set the `mfnet` path to the path where you stored the dataset (`ROOT_PATH/TO/MFNET`).




