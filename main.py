import configparser

from face_features_extraction import extract_dataset_to_csv
from src.tuning import run_tuning
from src.fold_assignments import k_fold_assignment

config = configparser.ConfigParser()
config.read('config.ini')

model_path = config['config1']['mediapipe_model_path']
dataset_path = config['config1']['dataset_path']
csv_path = config['config1']['extracted_features_csv_file_path']

extract_dataset_to_csv(root_dir=dataset_path, model_path=model_path, output_csv_path=csv_path)
k_fold_assignment(csv_file=csv_path)
run_tuning(csv_path)