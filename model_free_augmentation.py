import albumentations as A
import cv2
import os

TARGET_SIZE = 224
BASE_DATA_PATH = rf'C:\Users\wdomc\Documents\personal_color_analysis'
IMAGES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\raw_face_pictures"
TRAIN_IMAGES_PATH = rf"{IMAGES_PATH}\train"
TEST_IMAGES_PATH = rf"{IMAGES_PATH}\test"

if os.path.isdir(BASE_DATA_PATH):
    AUGMENTED_DATA_PATH = os.path.join(BASE_DATA_PATH, 'augmented_data')
    os.makedirs(AUGMENTED_DATA_PATH, exist_ok=True)
    part_dirs = os.listdir(IMAGES_PATH)
    class_dirs = os.listdir(TRAIN_IMAGES_PATH)
    for part_dir in part_dirs:
        part_path = os.path.join(AUGMENTED_DATA_PATH, part_dir + "_augmented")
        os.makedirs(part_path, exist_ok=True)
        for class_dir in class_dirs:
            class_path = os.path.join(part_path, class_dir)
            os.makedirs(class_path, exist_ok=True)


def augment_and_save_image(image_path, augment, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    augmented = augment(image=image)
    augmented_image = augmented['image']
    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, augmented_image_bgr)


if __name__ == '__main__':
    augment = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.1, 0.1),
            rotate=(-15, 15),
            shear=(-10, 10),
            p=1.0
        ),
        A.Resize(TARGET_SIZE, TARGET_SIZE),
    ])
    image_folders = []
    for color_type_class_name in os.listdir(TRAIN_IMAGES_PATH):
        color_type_path = os.path.join(TRAIN_IMAGES_PATH, color_type_class_name)
        # print(color_type_path)
        for subclass_name in os.listdir(color_type_path):
            subclass_folder = os.path.join(color_type_path, subclass_name)
            image_folders.append(subclass_folder)
        for image_folder in image_folders:
            images_filenames = os.listdir(image_folder)
            for image_filename in images_filenames:
                img_path = os.path.join(image_folder, image_filename)
                augmented_img_path = os.path.join(AUGMENTED_DATA_PATH,  "train_augmented", f"{color_type_class_name}", f"{image_filename}")
                # print(augmented_img_path)
                augment_and_save_image(img_path, augmented_img_path)
