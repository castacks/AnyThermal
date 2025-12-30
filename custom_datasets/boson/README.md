# Boson-Nightime Dataset

There is no extra post processing is required for Boson-Nightime dataset. The dataset can be used directly after downloading.

## Downloading the Data

Download the Boson-Nightime dataset from [this link](https://huggingface.co/datasets/xjh19972/boson-nighttime/tree/main/satellite-thermal-dataset-v1)

Follow the provided README on the link to see how to extract the dataset. 

Our recommended structure after extraction is as follows:

```
</your/chosen/path/boson-nighttime/>/
├── maps/
│   └── satellite/
│       └── 20201117_BingSatellite.png
│
├── satellite_0_satellite_0/
│   └── train_database.h5
│
├── satellite_0_thermalmapping_135/
│   ├── train_database.h5
│   ├── train_queries.h5
│   ├── val_database.h5
│   ├── val_queries.h5
│   ├── test_database.h5
│   └── test_queries.h5
│
└── satellite_0_thermalmapping_135_train/
    ├── extended_database.h5
    ├── extended_queries.h5
    ├── train_database.h5
    ├── train_queries.h5
    ├── val_database.h5
    ├── val_queries.h5
    ├── test_database.h5
    └── test_queries.h5
```

Add the root directory to `<ANYTHERMAL_ROOT_DIR>/custom_datasets/dataset_path.yaml` file to point to the root location of the extracted data for the `boson` dataset.

For example, we extract our dataset at `/ocean/projects/cis220039p/mdt2/datasets/boson-nighttime/` and set the root path to `/ocean/projects/cis220039p/mdt2/datasets/boson-nighttime/` in `<ANYTHERMAL_ROOT_DIR>/custom_datasets/dataset_path.yaml`.

