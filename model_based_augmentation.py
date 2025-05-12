import cv2
import os
import numpy as np
import shutil

BASE_DATA_PATH = rf'C:\Users\wdomc\Documents\personal_color_analysis\gan_augmented'
os.makedirs(BASE_DATA_PATH, exist_ok=True)
SHOWME_IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\ShowMeTheColor\res"
ARCRONOMIA_IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\raw_face_pictures"

for partition_path_name in os.listdir(SHOWME_IMAGES_PATH):
    if os.path.isdir(os.path.join(SHOWME_IMAGES_PATH, partition_path_name)):
        partition_path = os.path.join(SHOWME_IMAGES_PATH, partition_path_name)
        color_types = os.listdir(partition_path)
        for color_type_name in color_types:
            color_type_path = os.path.join(partition_path, color_type_name)
            dest_path = os.path.join(BASE_DATA_PATH, f"{color_type_name}")
            if not os.path.exists(dest_path):
                shutil.copytree(color_type_path, dest_path)





