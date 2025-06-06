import configparser

from src.face_features_extraction import extract_dataset_to_csv
from src.tuning import run_tuning
from src.fold_assignments import k_fold_assignment
from src.deep_learning_approach import run_deep_learning
from src.model_free_augmentation import run_augmentation

config = configparser.ConfigParser()
config.read('config.ini')

model_path = config['config1']['mediapipe_model_path']
dataset_path = config['config1']['dataset_path']
csv_path = config['config1']['extracted_features_csv_file_path']

# extract_dataset_to_csv(root_dir=dataset_path, model_path=model_path, output_csv_path=csv_path)
# k_fold_assignment(csv_file=csv_path)
# run_tuning(csv_path)

images_path = config['config2']['images_path']
model_free_images_path = config['config2']['model_free_augmented_images_path']
folds_assignment_path = config['config2']['folds_assignment_path']
model_free_folds_assignment_path = config['config2']['model_free_folds_assignment_path']
target_size = int(config['config2']['target_size'])
offset = int(config['config2']['offset'])
current_approach = config['config2']['current_approach']

# run_augmentation(images_path, model_free_images_path, target_size)
run_deep_learning(images_path, model_free_images_path, folds_assignment_path, model_free_folds_assignment_path, offset_val=offset, current_approach=current_approach)





