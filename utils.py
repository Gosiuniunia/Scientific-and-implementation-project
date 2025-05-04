import numpy as np
import cv2
from white_balancing.classes import WBsRGB as wb_srgb

def white_balance(img):
  wbModel = wb_srgb.WBsRGB()
  wb_img = wbModel.correctImage(img)
  return wb_img


def GrayworldeWB_algoritm(img):
    wb = cv2.xphoto.createGrayworldWB()
    wb_img = wb.balanceWhite(img)
    return wb_img


def crop_img(img, landmarks, indices):
    h, w, _ = img.shape
    points = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices]
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w_box, h_box = cv2.boundingRect(points)
    cropped_img = masked_img[y : y + h_box, x : x + w_box]
    return cropped_img, (x, y)


def apply_kmeans(img, k=4):
    img_data = img.reshape((-1, 3))
    img_data = np.float32(img_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(img.shape)
    return (centers, segmented_img)


def get_hsv_lab_colour(bgr_array):
    avg_bgr = np.mean(bgr_array, axis=0).astype(np.uint8)
    avg_bgr_reshaped = avg_bgr.reshape((1, 1, 3))
    avg_lab = cv2.cvtColor(avg_bgr_reshaped, cv2.COLOR_BGR2Lab)[0, 0]
    avg_hsv = cv2.cvtColor(avg_bgr_reshaped, cv2.COLOR_BGR2HSV)[0, 0]
    return np.concatenate([avg_lab, avg_hsv])


def get_color_between_points(p1, p2, crop_origin, segmented_img):
    cx = int((p1[0] + p2[0]) / 2) - crop_origin[0]
    cy = int((p1[1] + p2[1]) / 2) - crop_origin[1]

    h, w = segmented_img.shape[:2]
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)

    return segmented_img[cy, cx]
