import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('OIP.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Wykryj twarze na obrazie w skali szarości
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Narysuj prostokąty wokół wykrytych twarzy
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Wyświetl obraz z wykrytymi twarzami
cv2.imshow('Wykryte twarze', img)
cv2.waitKey(0)
cv2.destroyAllWindows()