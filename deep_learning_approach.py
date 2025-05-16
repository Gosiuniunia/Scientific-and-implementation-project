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

def load_images_from_folder(folder_path, class_map):
    X = []
    y = []
    for color_type_class_name in os.listdir(folder_path):
        color_type_path = os.path.join(folder_path, color_type_class_name)
        label = class_map[color_type_class_name]
        for img_filename in os.listdir(color_type_path):
            img_path = os.path.join(color_type_path, img_filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                X.append(img)
                y.append(label)
    return np.array(X), to_categorical(np.array(y), num_classes=4)

IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PColA"

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

class_map = {'fall': 0, 'spring': 1, 'summer': 2, 'winter': 3}

# X, y = load_images_from_folder_showme(IMAGES_PATH_1, class_map)
# history = model.fit(
#     X, y,
#     epochs=30,
#     # callbacks=[callbacks_list],
#     verbose=True,
#     shuffle=True)
#
# model.save(rf'C:\Users\wdomc\Documents\personal_color_analysis\model_weights\basic_vgg16.keras')

test_model = keras.models.load_model(rf"C:\Users\wdomc\Documents\personal_color_analysis\model_weights\basic_vgg16.keras")
X_test, y_test = load_images_from_folder(IMAGES_PATH, class_map)
loss, accuracy = test_model.evaluate(X_test, y_test, verbose=1)

print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Loss: {loss:.2f}")

y_pred_probs = test_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=class_map.keys()))