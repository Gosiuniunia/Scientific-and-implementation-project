import cv2
import os
import numpy as np
import shutil


BASE_DATA_PATH = rf'C:\Users\wdomc\Documents\personal_color_analysis\gan_augmented'
os.makedirs(BASE_DATA_PATH, exist_ok=True)
IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\raw_face_pictures"

for main_folder in os.listdir(IMAGES_PATH):
    main_folder_path = os.path.join(IMAGES_PATH, main_folder)
    if os.path.isdir(main_folder_path):
        for sub_folder in os.listdir(main_folder_path):
            sub_folder_path = os.path.join(main_folder_path, sub_folder)
