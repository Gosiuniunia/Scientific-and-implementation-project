import albumentations as A
import cv2
import os

TARGET_SIZE = 224
ORIGINAL_DATA_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PColA"
AUGMENTED_DATA_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PCoIA_model_free_augmented"

def augment_and_save_image(image_path, augment_operation, output_path):
    """
    Function applies provided augment operation, defined as Albumentations Compose operation, to an image and saves it as an image.
    Albumentations reference: https://albumentations.ai/docs/
    Args:
        image_path: path of image to be augmented
        augment_operation: Albumentations operation to be applied
        output_path: path to save augmented image
    Returns:
        None
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    augmented = augment_operation(image=image)
    augmented_image = augmented['image']
    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, augmented_image_bgr)

if __name__ == '__main__':

    horizontal_flip = A.Compose([A.Resize(TARGET_SIZE, TARGET_SIZE), A.HorizontalFlip(p=0.5)])
    translation_zoom_rotation = A.Compose([A.Resize(TARGET_SIZE, TARGET_SIZE), A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.7)])
    cutout = A.Compose([A.Resize(TARGET_SIZE, TARGET_SIZE), A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25), p=1.0)])

    classes_folders = os.listdir(ORIGINAL_DATA_PATH)

    for class_folder in classes_folders:
        class_folder_path = os.path.join(ORIGINAL_DATA_PATH, class_folder)
        images_files = os.listdir(class_folder_path)
        for img_filename in images_files:
            img_path = os.path.join(class_folder_path, img_filename)
            original_image = cv2.imread(img_path)
            original_output_img_path = os.path.join(AUGMENTED_DATA_PATH, class_folder, f"{img_filename}")
            cv2.imwrite(original_output_img_path, original_image)

            # horizontal flip application
            output_img_path = os.path.join(AUGMENTED_DATA_PATH, class_folder, f"hf_{img_filename}")
            img = augment_and_save_image(img_path, horizontal_flip, output_path=output_img_path)

            # translation, rotation, zoom application
            # output_img_path = os.path.join(AUGMENTED_DATA_PATH, class_folder, f"tz_{img_filename}")
            # img = augment_and_save_image(img_path, translation_zoom_rotation, output_path=output_img_path)

            # cutout application
            output_img_path = os.path.join(AUGMENTED_DATA_PATH, class_folder, f"co_{img_filename}")
            img = augment_and_save_image(img_path, cutout, output_path=output_img_path)

            # print(output_img_path)






