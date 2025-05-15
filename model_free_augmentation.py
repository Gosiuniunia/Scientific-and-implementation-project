import albumentations as A
import cv2
import os

TARGET_SIZE = 224
BASE_DATA_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PColA_cut"
AUGMENTED_DATA_PATH = os.path.join(BASE_DATA_PATH, 'augmented_data')

def augment_and_save_image(image_path, augment_operation, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    augmented = augment_operation(image=image)
    augmented_image = augmented['image']
    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, augmented_image_bgr)


if __name__ == '__main__':

    # augment = A.Compose([
    #     A.RandomRotate90(p=0.5),
    #     A.HorizontalFlip(p=0.5),
    #     A.Affine(
    #         scale=(0.9, 1.1),
    #         translate_percent=(0.1, 0.1),
    #         rotate=(-15, 15),
    #         shear=(-10, 10),
    #         p=1.0
    #     ),
    #     A.Resize(TARGET_SIZE, TARGET_SIZE),
    # ])

    augment_random_rotate = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Resize(TARGET_SIZE, TARGET_SIZE),
    ])

