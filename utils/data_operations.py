import os
import shutil
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
# dataset1: Korean one
TEST_IMAGES_PATH_1 = fr"C:\Users\wdomc\Documents\personal_color_analysis\ShowMeTheColor\res\test"
TRAIN_IMAGES_PATH_1 = rf"C:\Users\wdomc\Documents\personal_color_analysis\ShowMeTheColor\res\train"

# dataset 2: Italian one
IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\raw_face_pictures"
TRAIN_IMAGES_PATH_2 = rf"{IMAGES_PATH}\train"
TEST_IMAGES_PATH_2 = rf"{IMAGES_PATH}\test"

def load_images_from_folder_showme(folder_path, class_map):
    X = []
    y = []
    for color_type_class_name in os.listdir(folder_path):
        color_type_path = os.path.join(folder_path, color_type_class_name)
        label = class_map[color_type_class_name]
        for img_filename in os.listdir(color_type_path):
            img_path = os.path.join(color_type_path, img_filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                X.append(img)
                y.append(label)
    return np.array(X), to_categorical(np.array(y), num_classes=4)

def load_images_from_folder_armocromia(folder_path, class_map):
    X = []
    y = []
    for color_type_class_name in os.listdir(folder_path):
        color_type_path = os.path.join(folder_path, color_type_class_name)
        for subclass_name in os.listdir(color_type_path):
            subclass_folder = os.path.join(color_type_path, subclass_name)
            if os.path.isdir(subclass_folder):
                label = class_map[color_type_class_name]
                for img_file in os.listdir(subclass_folder):
                    img_path = os.path.join(subclass_folder, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        X.append(img)
                        y.append(label)
    return np.array(X), to_categorical(np.array(y), num_classes=4)

