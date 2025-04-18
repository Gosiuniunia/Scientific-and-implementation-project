import numpy as np
import cv2

def GrayworldeWB_algoritm(img):
    wb = cv2.xphoto.createGrayworldWB()
    wb_img = wb.balanceWhite(img)
    return wb_img

def crop_img(img, landmarks, indices):
    h, w, _ = img.shape
    points = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w_box, h_box = cv2.boundingRect(points)
    cropped_img = masked_img[y:y+h_box, x:x+w_box]
    return cropped_img
