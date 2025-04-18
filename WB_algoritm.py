import cv2

def simpleWB_algoritm(img):
    wb = cv2.xphoto.createSimpleWB()
    wb_img = wb.balanceWhite(img)
    return wb_img


img = cv2.imread('photo2.jpg')
wb_img = simpleWB_algoritm(img)
cv2.imshow('Oryginalny obraz', img)
cv2.imshow('Obraz po balansie bieli', wb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('photo_after_simplewb.jpg', wb_img)