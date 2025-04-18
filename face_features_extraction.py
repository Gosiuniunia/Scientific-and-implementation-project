import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from utils import crop_img, GrayworldeWB_algoritm
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
        left_iris_img = crop_img(img, face_landmarks, left_iris_indices)
        right_iris_img = crop_img(img, face_landmarks, right_iris_indices)

        cv2.imshow("Left Eye", left_iris_img)
        cv2.imshow("Right Eye", right_iris_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def extract_skin_colour(img, face_landmarks):
    cheek_indices = [111, 117, 118, 101, 36, 203, 165, 50, 205, 206, 92, 147, 187, 207, 216, 192, 214, 212, 340, 346, 347, 330, 266, 423, 280, 425, 426, 322, 411, 427, 436, 410, 432, 434, 416]
    chin_indices = [202, 204, 194, 201, 200, 199, 421, 418, 424, 422, 262, 428, 32, 208]
    nose_indices = [168, 6, 197, 195, 5, 4, 45, 275, 120, 100, 349, 329, 167, 164, 393]
    forehead_indices = [103, 67, 109, 10, 338, 297, 332, 104, 69, 108, 151, 337, 299, 333, 9, 8]

    all_indices = forehead_indices + cheek_indices + nose_indices + chin_indices
    for face_landmarks in face_landmarks:
        for idx in all_indices:
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow('Face Landmarks', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def extract_hair_colour(img_rgb, face_landmarks):
    left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53]
    right_eyebrow = [336, 296, 334, 293, 276, 283, 282, 295, 285]
    for face_landmarks in face_landmarks:
        left_eyebrow_img = crop_img(img_rgb, face_landmarks, left_eyebrow)
        right_eyebrow_img = crop_img(img_rgb, face_landmarks, right_eyebrow)
        cv2.imshow("Left Eyebrow", left_eyebrow_img)
        cv2.imshow("Right Eyebrow", right_eyebrow_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

FaceLandmarker, options = init_face_landmark(model_path)
face_landmarks = get_face_landmarks(FaceLandmarker, options, img_rgb)

extract_iris_colour(img, face_landmarks)
extract_skin_colour(img, face_landmarks)
extract_hair_colour(img, face_landmarks)