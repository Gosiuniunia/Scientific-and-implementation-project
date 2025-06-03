import configparser

from src.utils.face_features_extraction import extract_dataset_to_csv
from src.utils.tuning import run_tuning

config = configparser.ConfigParser()
config.read('config.ini')

model_path = config['config1']['mediapipe_model_path']
dataset_path = config['config1']['dataset_path']
csv_path = config['config1']['extracted_features_csv_file_path']

extract_dataset_to_csv(root_dir=dataset_path, model_path=model_path, output_csv_path=csv_path)
run_tuning(csv_path)