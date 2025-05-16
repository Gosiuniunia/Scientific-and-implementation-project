import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from utils.color_utils import white_balance, crop_img, apply_kmeans, get_hsv_lab_colour, get_color_between_points
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_path = r"C:/studia/P_nw/face_landmarker.task"
image_path = "OIP.jpg"


def init_face_landmark(model_path):
    """
    Initializes the MediaPipe FaceLandmarker model for facial landmark detection.

    Args:
        model_path (str): Path to the '.task' model file.

    Returns:
        tuple: A tuple containing the FaceLandmarker class and its configuration options (FaceLandmarkerOptions).
    """
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    return FaceLandmarker, options


def get_face_landmarks(FaceLandmarker, options, img_rgb):
    """
    Detects facial landmarks from an RGB image.

    Args:
        FaceLandmarker: MediaPipe FaceLandmarker class.
        options: Configuration options for the landmark model.
        img_rgb (np.ndarray): The input image in RGB format.

    Returns:
        list: A list of detected facial landmarks.
    """
    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        return result.face_landmarks


def extract_iris_colour(img, face_landmarks):
    """
    Extracts the iris color from the image using facial landmarks.

    Args:
        img (np.ndarray): The original image in BGR format.
        face_landmarks (list): List of facial landmarks.

    Returns:
        np.ndarray: A combined LAB and HSV color vector representing the iris color.
    """
    right_iris_indices = [374, 476, 475, 474]
    left_iris_indices = [469, 145, 471, 159]
    pupil_indices = [468, 473]
    for face_landmarks in face_landmarks:
        left_iris_img, left_origin = crop_img(img, face_landmarks, left_iris_indices)
        right_iris_img, right_origin = crop_img(img, face_landmarks, right_iris_indices)
        segmented_img_li = apply_kmeans(left_iris_img)
        segmented_img_ri = apply_kmeans(right_iris_img)
        left_iris_colour = get_color_between_points(
            (
                face_landmarks[469].x * img.shape[1],
                face_landmarks[469].y * img.shape[0],
            ),
            (
                face_landmarks[145].x * img.shape[1],
                face_landmarks[145].y * img.shape[0],
            ),
            left_origin,
            segmented_img_li,
        )
        right_iris_colour = get_color_between_points(
            (
                face_landmarks[374].x * img.shape[1],
                face_landmarks[374].y * img.shape[0],
            ),
            (
                face_landmarks[476].x * img.shape[1],
                face_landmarks[476].y * img.shape[0],
            ),
            right_origin,
            segmented_img_ri,
        )

        iris_colour = get_hsv_lab_colour([right_iris_colour, left_iris_colour])
        return iris_colour


def extract_skin_colour(img, face_landmarks):
    """
    Extracts skin color by sampling predefined facial landmarks.

    Args:
        img (np.ndarray): The original image in BGR format.
        face_landmarks (list): List of facial landmarks.

    Returns:
        np.ndarray: A combined LAB and HSV color vector representing the skin tone.
    """
    skin_extraction_landmarks = [195, 5]
    skin_colours = []
    for face in face_landmarks:
        for idx in skin_extraction_landmarks:
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
    """
    Extracts eyebrow (hair) color using facial landmarks.

    Args:
        img (np.ndarray): The original image in BGR format.
        face_landmarks (list): List of facial landmarks.

    Returns:
        np.ndarray: A combined LAB and HSV color vector representing eyebrow color.
    """
    left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53]
    right_eyebrow = [336, 296, 334, 293, 276, 283, 282, 295, 285]
    for face_landmarks in face_landmarks:
        left_eyebrow_img, left_origin = crop_img(img, face_landmarks, left_eyebrow)
        right_eyebrow_img, right_origin = crop_img(img, face_landmarks, right_eyebrow)
        segmented_img_le = apply_kmeans(left_eyebrow_img)
        segmented_img_re = apply_kmeans(right_eyebrow_img)
        left_eyebrow_colour = get_color_between_points(
            (
                face_landmarks[105].x * img.shape[1],
                face_landmarks[105].y * img.shape[0],
            ),
            (face_landmarks[65].x * img.shape[1], face_landmarks[65].y * img.shape[0]),
            left_origin,
            segmented_img_le,
        )

        right_eyebrow_colour = get_color_between_points(
            (
                face_landmarks[334].x * img.shape[1],
                face_landmarks[334].y * img.shape[0],
            ),
            (
                face_landmarks[295].x * img.shape[1],
                face_landmarks[295].y * img.shape[0],
            ),
            right_origin,
            segmented_img_re,
        )
    eyebrow_colour = get_hsv_lab_colour([left_eyebrow_colour, right_eyebrow_colour])
    return eyebrow_colour


def extract_lab_hsv_values_from_photo(image_path, FaceLandmarker, options):
    """
    Loads an image, detects facial landmarks, and extracts iris, skin, and eyebrow colors.

    Args:
        image_path (str): Path to the input image.
        FaceLandmarker: MediaPipe FaceLandmarker class.
        options: Configuration options for the landmark model.

    Returns:
        list: A flattened list of LAB and HSV color features from iris, skin, and eyebrow.
    """
    img = cv2.imread(image_path)
    balanced_img = white_balance(img)
    img = (balanced_img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_landmarks = get_face_landmarks(FaceLandmarker, options, img_rgb)

    iris_colour = extract_iris_colour(img, face_landmarks)
    skin_colour = extract_skin_colour(img, face_landmarks)
    eyebrow_colour = extract_hair_colour(img, face_landmarks)
    extracted_values = np.concatenate(
        [iris_colour, skin_colour, eyebrow_colour]
    ).tolist()
    return extracted_values


def extract_dataset_to_csv(root_dir):
    """ 
    Extracts color features (in HSV and LAB color spaces) from images located in subdirectories of the root folder.
    Each subdirectory is treated as a separate class label. The final CSV file will be named after the root folder.

    Args:
        root_dir (str): Root directory containing labeled subdirectories of images.

    Returns:
        pd.DataFrame: A DataFrame where each row contains extracted features and a label.
    """
    FaceLandmarker, options = init_face_landmark(model_path)

    iris_columns = [f"iris_{ch}" for ch in ["L", "a", "b", "H", "S", "V"]]
    skin_columns = [f"skin_{ch}" for ch in ["L", "a", "b", "H", "S", "V"]]
    eyebrow_columns = [f"eyebrow_{ch}" for ch in ["L", "a", "b", "H", "S", "V"]]
    all_columns = ["id"] + iris_columns + skin_columns + eyebrow_columns + ["label"]
    df = pd.DataFrame(columns=all_columns)

    for label_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, label_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                full_path = os.path.join(class_dir, filename)
                extracted_values = extract_lab_hsv_values_from_photo(
                    full_path, FaceLandmarker, options
                )
                row = [filename] + extracted_values + [label_name]
                df.loc[len(df)] = row
    df.to_csv(root_dir + ".csv")

# primavera

df = extract_dataset_to_csv("dataset_PColA")