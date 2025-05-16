from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import cv2
import numpy as np
import os
from sklearn.metrics import classification_report
from utils.data_operations import load_images_from_folder_armocromia, load_images_from_folder_showme

# dataset1: Korean one
TEST_IMAGES_PATH_1 = fr"C:\Users\wdomc\Documents\personal_color_analysis\ShowMeTheColor\res\test"
TRAIN_IMAGES_PATH_1 = rf"C:\Users\wdomc\Documents\personal_color_analysis\ShowMeTheColor\res\train"

# dataset 2: Italian one
IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\raw_face_pictures"
TRAIN_IMAGES_PATH_2 = rf"{IMAGES_PATH}\train"
TEST_IMAGES_PATH_2 = rf"{IMAGES_PATH}\test"

# Parameters
input_shape = (224, 224, 3)
num_classes = 4

base_model = VGG16(include_top=True, weights='imagenet', input_shape=input_shape)
x = base_model.layers[-2].output
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
            layer.trainable = False

model.summary()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

class_map_1 = {'fall': 0, 'spring': 1, 'summer': 2, 'winter': 3}
class_map_2 = {'primavera': 0, 'estate': 1, 'autunno': 2, 'inverno': 3}

# X, y = load_images_from_folder_showme(TRAIN_IMAGES_PATH_1, class_map_1)
# history = model.fit(
#     X, y,
#     epochs=30,
#     # callbacks=[callbacks_list],
#     verbose=True,
#     shuffle=True)
#
# model.save(rf'C:\Users\wdomc\Documents\personal_color_analysis\model_weights\basic_vgg16.keras')

test_model = keras.models.load_model(rf"C:\Users\wdomc\Documents\personal_color_analysis\model_weights\basic_vgg16.keras")
X_test, y_test = load_images_from_folder_armocromia(TEST_IMAGES_PATH_2, class_map_2)
loss, accuracy = test_model.evaluate(X_test, y_test, verbose=1)

print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Loss: {loss:.2f}")

y_pred_probs = test_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=class_map_2.keys()))