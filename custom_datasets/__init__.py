import yaml
import os

dataset_yaml = os.path.join(os.path.dirname(__file__), 'dataset_path.yaml')
with open(dataset_yaml, 'r') as f:
    DATASETS = yaml.safe_load(f)

for dataset_name, dataset_path in DATASETS['data_root'].items():
    os.environ["ANYTHERMAL_"+dataset_name.upper() + '_DATA_ROOT'] = dataset_path
