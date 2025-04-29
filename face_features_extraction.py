import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pandas as pd
from utils import crop_img, apply_kmeans, get_hsv_lab_colour, get_color_between_points
from face_detection import face_detection_using_haar
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


model_path = r'C:/studia/P_nw/face_landmarker.task'
image_path = 'OIP.jpg'

def init_face_landmark(model_path):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    return FaceLandmarker, options


def get_face_landmarks(FaceLandmarker, options, img_rgb):
    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        return result.face_landmarks

def extract_iris_colour(img, face_landmarks):
    right_iris_indices = [374,476, 475, 474]
    left_iris_indices  = [469, 145, 471, 159]
    pupil_indices = [468, 473]
    for face_landmarks in face_landmarks:
        left_iris_img, left_origin = crop_img(img, face_landmarks, left_iris_indices)
        right_iris_img, right_origin = crop_img(img, face_landmarks, right_iris_indices)
        left_iris_centers, segmented_img_li = apply_kmeans(left_iris_img, k=2)
        right_iris_centers, segmented_img_ri = apply_kmeans(right_iris_img, k=2)
        left_pupil_colour = get_color_between_points(
            (face_landmarks[468].x * img.shape[1], face_landmarks[468].y * img.shape[0]), 
            (face_landmarks[468].x * img.shape[1], face_landmarks[468].y * img.shape[0]), 
            left_origin, segmented_img_li
        )
        right_pupil_colour = get_color_between_points(
            (face_landmarks[473].x * img.shape[1], face_landmarks[473].y * img.shape[0]), 
            (face_landmarks[473].x * img.shape[1], face_landmarks[473].y * img.shape[0]), 
            right_origin, segmented_img_ri
        )
        cv2.imshow('segmented_img_li cfen', segmented_img_li)
        cv2.imshow('segmented_img_ri cfen', segmented_img_ri)
        right_iris_colour = right_iris_centers[0] if np.all(right_pupil_colour == right_iris_centers[0]) else right_iris_centers[1]
        left_iris_colour = left_iris_centers[0] if np.all(left_pupil_colour == left_iris_centers[0]) else left_iris_centers[1]

        iris_colour = get_hsv_lab_colour([right_iris_colour, left_iris_colour])
        return iris_colour



def extract_skin_colour(img, face_landmarks):
    cheek_indices = [111, 117, 118, 101, 36, 203, 165, 50, 205, 206, 92, 147, 187, 207, 216, 192, 214, 212, 340, 346, 347, 330, 266, 423, 280, 425, 426, 322, 411, 427, 436, 410, 432, 434, 416]
    chin_indices = [202, 204, 194, 201, 200, 199, 421, 418, 424, 422, 262, 428, 32, 208]
    nose_indices = [168, 6, 197, 195, 5, 4, 45, 275, 120, 100, 349, 329, 167, 164, 393]
    forehead_indices = [103, 67, 109, 10, 338, 297, 332, 104, 69, 108, 151, 337, 299, 333, 9, 8]

    all_indices = forehead_indices + cheek_indices + nose_indices + chin_indices
    skin_colours = []
    for face in face_landmarks:
        for idx in all_indices:
            if idx < len(face):
                landmark = face[idx]
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    color = img[y, x]
                    skin_colours.append(color)

    skin_colour = get_hsv_lab_colour(skin_colours)

    return skin_colour

def extract_hair_colour(img, face_landmarks):
    left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53]
    right_eyebrow = [336, 296, 334, 293, 276, 283, 282, 295, 285]
    for face_landmarks in face_landmarks:
        left_eyebrow_img, left_origin = crop_img(img, face_landmarks, left_eyebrow)
        right_eyebrow_img, right_origin = crop_img(img, face_landmarks, right_eyebrow)
        left_eyebrow_centers, segmented_img_le = apply_kmeans(left_eyebrow_img, k=2)
        right_eyebrow_centers, segmented_img_re = apply_kmeans(right_eyebrow_img, k=2)

        cv2.imshow('segmented_img_le', segmented_img_le)
        cv2.imshow('segmented_img_re', segmented_img_re)
        left_eyebrow_colour = get_color_between_points(
            (face_landmarks[105].x * img.shape[1], face_landmarks[105].y * img.shape[0]), 
            (face_landmarks[65].x * img.shape[1], face_landmarks[65].y * img.shape[0]), 
            left_origin, segmented_img_le
        )

        right_eyebrow_colour = get_color_between_points(
            (face_landmarks[334].x * img.shape[1], face_landmarks[334].y * img.shape[0]), 
            (face_landmarks[295].x * img.shape[1], face_landmarks[295].y * img.shape[0]), 
            right_origin, segmented_img_re
        )
    eyebrow_colour = get_hsv_lab_colour([left_eyebrow_colour, right_eyebrow_colour])
    return eyebrow_colour

def extract_lab_hsv_values_from_photo(image_path):
    img = face_detection_using_haar(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    FaceLandmarker, options = init_face_landmark(model_path)
    face_landmarks = get_face_landmarks(FaceLandmarker, options, img_rgb)

    iris_colour = extract_iris_colour(img, face_landmarks)
    skin_colour = extract_skin_colour(img, face_landmarks)
    eyebrow_colour = extract_hair_colour(img, face_landmarks)
    return np.concatenate([iris_colour, skin_colour, eyebrow_colour])



iris_columns = [f"iris_{ch}" for ch in ["L", "a", "b", "H", "S", "V"]]
skin_columns = [f"skin_{ch}" for ch in ["L", "a", "b", "H", "S", "V"]]
eyebrow_columns = [f"eyebrow_{ch}" for ch in ["L", "a", "b", "H", "S", "V"]]
columns = iris_columns + skin_columns + eyebrow_columns

data = extract_lab_hsv_values_from_photo("JLO.jpg")
df = pd.DataFrame([data], columns=columns)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(df)