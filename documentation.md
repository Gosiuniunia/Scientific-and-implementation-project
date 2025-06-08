# Enviroment documentation

This experimental environment contains implementations of machine learning (ML) and deep learning (DL) methods for seasonal color analysis. The environment incorporates various components including color feature extraction (based on facial images and landmark detection), hyperparameter tuning of ML classifiers (KNN, SVM, Decision Trees), and preparation and training of DL models (e.g., VGG16), with support for data augmentation.

Additionally, the environment provides tools for testing and comparing classifier performance using statistical tests.

## data

The folder contains two CSV files that assign images to each fold for k-fold cross-validation:

- `fold_assignments.csv` — used to ensure consistent comparisons between machine learning and deep learning approaches.
- `model_free_fold_assignments.csv` — used to ensure consistent data separation, independent of any specific modeling approach.

## main.py

Main pipeline script for facial feature extraction, data augmentation, model tuning, 
and deep learning training.

All paths and parameters are loaded from the [config1] section of the config.ini file.

## src/utils

This module provides helper functions for facial image processing, including white balancing and color extraction.

### src/utils/white_balancing

This submodule contains functions related to correcting and adjusting image colors via white balancing.


### src/utils/color_utils 

This submodule provides helper functions for color extraction and processing from facial regions. It includes image cropping based on facial landmarks, color segmentation using K-Means, and color space conversion to LAB and HSV formats.
    

FUNCTIONS

    apply_kmeans(img, k=5)
        Applies K-Means clustering to segment colors in the image.

        Args:
            img (np.ndarray): Input image in BGR format.
            k (int, optional): Number of clusters. Defaults to 4.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Cluster centers (BGR colors) as np.ndarray of shape (k, 3).
                - Segmented image with colors replaced by their cluster center.

    crop_img(img, landmarks, indices)
        Crops a region of the image based on facial landmarks.

        Args:
            img (np.ndarray): Input image in BGR format.
            landmarks (List[NormalizedLandmark]): List of facial landmarks.
            indices (List[int]): Indices of the landmarks to define the region.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]:
                - Cropped region of the image (np.ndarray).
                - Origin (x, y) of the cropped region relative to the original image.

    get_color_between_points(p1, p2, crop_origin, segmented_img)
        Gets the color from the image at the midpoint between two points: p1 and p2.

        Args:
            p1 (Tuple[float, float]): First point (x, y).
            p2 (Tuple[float, float]): Second point (x, y).
            crop_origin (Tuple[int, int]): Origin (x, y) of the crop in original image.
            segmented_img (np.ndarray): Segmented image (from KMeans).

        Returns:
            np.ndarray: BGR color at the midpoint between p1 and p2.

    get_hsv_lab_colour(bgr_array)
        Converts a list of BGR colors to average LAB and HSV colour representations.

        Args:
            bgr_array (List[np.ndarray] or np.ndarray): List or array of BGR colors.

        Returns:
            np.ndarray: Concatenated LAB and HSV average values (length 6).

    white_balance(img)
        Performs white-balancing of the image
          Ref: Afifi, Mahmoud, et al. "When color constancy goes wrong: Correcting improperly white-balanced images."
          Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

        Args:
              img (np.ndarray): Input image in BGR format.

        Returns:
              wb_img (np.ndarray): White-balanced image in RGB format.

## src/face_features_extraction

This script extracts color features from facial regions (iris, skin, eyebrows) using MediaPipe Face Landmarker.
It processes images organized into subdirectories (each representing a class label), computes HSV and LAB color
values, and saves the results in a CSV file.

Usage:
- Place labeled image folders in a root directory (e.g., `dataset_PColA`).
- Set the correct model path (`.task` file).
- Run the script to generate a dataset CSV.

FUNCTIONS

    extract_dataset_to_csv(root_dir)
        Extracts color features (in HSV and LAB color spaces) from images located in subdirectories of the root folder.
        Each subdirectory is treated as a separate class label. The final CSV file will be saved to output_csv_path.

        Args:
            root_dir (str): Root directory containing labeled subdirectories of images.
            model_path (str): Path to the face landmarker model.
            output_csv_path (str): Full path (including filename) where CSV will be saved.


        Saves:
            - CSV file at output_csv_path containing extracted features and labels.

    extract_hair_colour(img, face_landmarks)
        Extracts eyebrow (hair) color using facial landmarks.

        Args:
            img (np.ndarray): The original image in BGR format.
            face_landmarks (list): List of facial landmarks.

        Returns:
            np.ndarray: A combined LAB and HSV color vector representing eyebrow color.

    extract_iris_colour(img, face_landmarks)
        Extracts the iris color from the image using facial landmarks.

        Args:
            img (np.ndarray): The original image in BGR format.
            face_landmarks (list): List of facial landmarks.

        Returns:
            np.ndarray: A combined LAB and HSV color vector representing the iris color.

    extract_lab_hsv_values_from_photo(image_path, FaceLandmarker, options)
        Loads an image, detects facial landmarks, and extracts iris, skin, and eyebrow colors.

        Args:
            image_path (str): Path to the input image.
            FaceLandmarker: MediaPipe FaceLandmarker class.
            options: Configuration options for the landmark model.

        Returns:
            list: A flattened list of LAB and HSV color features from iris, skin, and eyebrow.

    extract_skin_colour(img, face_landmarks)
        Extracts skin color by sampling predefined facial landmarks.

        Args:
            img (np.ndarray): The original image in BGR format.
            face_landmarks (list): List of facial landmarks.

        Returns:
            np.ndarray: A combined LAB and HSV color vector representing the skin tone.

    get_face_landmarks(FaceLandmarker, options, img_rgb)
        Detects facial landmarks from an RGB image.

        Args:
            FaceLandmarker: MediaPipe FaceLandmarker class.
            options: Configuration options for the landmark model.
            img_rgb (np.ndarray): The input image in RGB format.

        Returns:
            list: A list of detected facial landmarks.

    init_face_landmark(model_path)
        Initializes the MediaPipe FaceLandmarker model for facial landmark detection.

        Args:
            model_path (str): Path to the '.task' model file.

        Returns:
            tuple: A tuple containing the FaceLandmarker class and its configuration options (FaceLandmarkerOptions).

## src/tuning

Hyperparameter tuning for KNN, SVM, and Decision Tree classifiers
using repeated stratified k-fold cross-validation.

FUNCTIONS

    select_features(df, feature_type)
        Selects specific feature columns from the dataframe based on the feature type.

        Args:
            df (pandas.DataFrame): The input dataframe containing all features and the label.
            feature_type (str): The type of features to select.
                - "all": select all features (all columns except label).
                - "HSV": select only columns ending with '_H', '_S', '_V'.
                - "Lab": select only columns ending with '_L', '_a', '_b'.

        Returns:
            numpy.ndarray: Numpy array of selected features.

    tune_dt_params(feature_type, X, y, max_depths, rskf)
        Performs hyperparameter tuning for the Decision Tree classifier
        using repeated stratified K-fold cross-validation.

        Args:
            max_depths (list[int]): Maximum tree depths to evaluate.
            rskf (RepeatedStratifiedKFold): Cross-validation splitting strategy.

        Saves:
            - 'dt_accuracies.npy': Accuracy scores for each depth and fold.
            - 'dt_precisions.npy': Precision scores for each depth and fold.
            - 'dt_recalls.npy': Recall scores for each depth and fold.
            - 'dt_f1s.npy': F1 scores for each depth and fold.

    tune_knn_params(feature_type, X, y, metrics, weights, rskf)
        Performs hyperparameter tuning for the K-Nearest Neighbors (KNN) classifier
        using repeated stratified K-fold cross-validation.

        Args:
            metrics (list[str]): Distance metrics to evaluate, e.g., ['euclidean', 'manhattan'].
            weights (list[str]): Weight functions to evaluate, e.g., ['uniform', 'distance'].
            rskf (RepeatedStratifiedKFold): Cross-validation splitting strategy.

        Saves:
            - 'knn_accuracies.npy': Accuracy scores for each parameter combination and fold.
            - 'knn_precisions.npy': Precision scores for each parameter combination and fold.
            - 'knn_recalls.npy': Recall scores for each parameter combination and fold.
            - 'knn_f1s.npy': F1 scores for each parameter combination and fold.

    tune_svm_params(feature_type, X, y, kernels, Cs, gammas, rskf)
        Performs hyperparameter tuning for the Support Vector Machine (SVM) classifier
        using repeated stratified K-fold cross-validation.

        Args:
            kernels (list[str]): SVM kernel types to test, e.g., ['linear', 'rbf'].
            Cs (list[float]): Values for the regularization parameter C.
            gammas (list[Union[float, str]]): Gamma values, or 'scale'/'auto'.
            rskf (RepeatedStratifiedKFold): Cross-validation splitting strategy.

        Saves:
            - 'svm_accuracies.npy': Accuracy scores for each parameter combination and fold.
            - 'svm_precisions.npy': Precision scores for each parameter combination and fold.
            - 'svm_recalls.npy': Recall scores for each parameter combination and fold.
            - 'svm_f1s.npy': F1 scores for each parameter combination and fold.

## src/fold_assignments

Saves k fold assignments into csv in order to compare DL and ML approach.

FUNCTIONS
    k_fold_assignment(csv_file)
        This function performs a stratified 5-fold split of the input dataset and saves the fold assignment for each file into a CSV file.

        Parameters:
        csv_file (str): Path to the input CSV file.
        csv_output (str): Path to the output CSV fie.

        Output:
        - Saves fold assigments into csv_output file.

## model_free_augmentation

Performing model free image data augmentation, using Albumentations library.
Augmentation operations are cutoff and horizontal flip.

FUNCTIONS

    augment_and_save_image(image_path, augment_operation, output_path)
        Function applies provided augment operation, defined as Albumentations Compose operation, to an image and saves it as an image.
        Albumentations reference: https://albumentations.ai/docs/
        Args:
            image_path: path of image to be augmented
            augment_operation: Albumentations operation to be applied
            output_path: path to save augmented image
        Returns:
            None

    run_augmentation(original_data_path, augmented_data_path, target_size)
    Function runs model free image data augmentation using Albumentations library
    on images specified in original_data_path and saves them in augmented_data_path directory.
    Images are divided into folders describing class withing each directory.
    Result image size is specified by target_size.
    Args:
        original_data_path: Path to orginal dataset folder.
        augmented_data_path: Path to save augmented images.
        target_size: Desired image size (height and width) after resizing.
    Returns:
        None
    '''

## scr/deep_learning_approach

This script:

- performs a transfer learning of CNN VGG16 model, using provided image data, so it can perform seasonal beauty type classification
- VGG16 model has only an output layer changed so it's adjusted for 4 classes classification problem
- collects model training, evaluation and classification statistics like accuracy, precision, recall and F1 score
- Kfolds validation is used in the learning process - there is k models trained, each time with different train and test data, indicated by the file fold.assignments.csv


FUNCTIONS

     adjust_folds_assignment_file(assignment_file_input_path, assignment_file_output_path, prefixes_list)
        Function modifies prepares the folds assignment for the augmented data.
        Args:
            assignment_file_path: path to a .csv file with folds assignment information
            prefixes_list: list of prefixes which are used in augmented images files, for example "co" for cut-out augmented images files

        Returns:
            None

    prepare_vgg16_model()
        Function sets up a VGG16 model with Imagenet pre-trained weights for 4 classes classification task.
        Model reference: https://keras.io/api/applications/vgg/

        Returns:
            model: VGG16 model

    run_deep_learning(images_path, model_free_images_path, folds_assignment_path, model_free_folds_assignment_path, offset_val=0, current_approach='basic_shuffle_with_seed')
        Function runs training of the VGG16 model for PCoA task.
        Args:
            images_path: path to original images
            model_free_images_path: path to images with model free image augmentaation applied
            folds_assignment_path: path to the csv file containing fold assigment information
            model_free_folds_assignment_path: path to the csv file containing fold assignment information when using model free image augmentation
            offset_val: integer value needed when experiment is run twice to change the numeration of folds.
            current_approach: current approach to train model.
            Value should be set to "basic_shuffle_with_seed" when training without data augmentation applied is intended.
            Value should be set to "model_free_shuffle_with_seed" when training with data augmentation applied is intended.
            Should be set to 0 or 5. By default set to 0.

        Returns:
            None

    split_data_test_train(assignment_file_path, fold, offset=0)
        Splits images details (file path, label), provided as pandas Dataframe, into train and test sets details pandas Dataframes,
        taking into account the fold number k
        Args:
            k: number of current fold

        Returns:
            train_df: train data details pandas Dataframe, including label and file path
            test_df:  test data details pandas Dataframe, including label and file path



## src/testing

Statistical tests and tables for metrics.  

FUNCTIONS

    compare_models(scores, model_names, table_style='grid', alpha=0.05, alternative='two-sided')
        Compares sets of related samples, performs statistical tests (Shapiro-Wilk for normality,
        followed by paired t-test for normal data or Wilcoxon signed-rank test for non-normal data),
        generates and prints a table with p-value of a paired test with the information about which of them
        was performed ("t" being t-test and "w" being the Wilcoxon).

        Args:
            scores (np.array[float]): Array of samples scores for testing.
            model_names (list[str]): List of the compared model's names.
            table_style (str, optional): The formatting style for the table (e.g., "latex", "grid"). Defaults to "grid".
            alpha (float, optional): The significance level for the statistical test for normality. Defaults to 0.05.
            alternative (str, optional): The alternative hypothesis for the comparison tests. Can be 'two-sided', 'less', or 'greater'. Defaults to "two-sided".

    print_scores(classifier_name, feature_types=['all', 'HSV', 'Lab'], round=None, table_style='grid', T=False)
        Generates and prints tables of scores (mean and standard deviation) for different metrics
        based on the results stored in .npy files for the chosen classifier of Feature-based Machine Learning.

        Args:
            classifier_name (str): The name of a classifier (e.g., "DT", "KNN").
            feature_types (list[str]): The list of feature types. Defaults to ["all", "HSV", "Lab"].
            round (int, optional): The number of decimals for possible measures rounding
            table_style (str, optional): The formatting style for the table (e.g., "latex", "grid"). Defaults to "grid"
            T (bool, optional): Argument, which controls whether the table should be transposed. Defaults to False.

    print_scores_deep(round=None, table_style='grid', return_scores=False)
        Generates and prints table of scores (mean and standard deviation) for different metrics
        based on the results stored in .npy files for the Deep learning approach.

        Args:
            round (int, optional): The number of decimals for possible measures rounding
            table_style (str, optional): The formatting style for the table (e.g., "latex", "grid"). Defaults to "grid"
            return_scores (bool, optional): Whether the scores should be returned. Defaults to False.
        Returns:
            acc_scores, pre_scores, rec_scores, f1_scores (list[float], optional): Metrics values lists.

    load_tuned_models_data()
        Loads all of the model's metrics values for chosen hyperparamethers and deep learning scores.

        Returns:
            data (dict[dict]): Dictonary of all scores for chosen models.

    load_and_generate_all_scores(summary_scores_path)
        Loads and generates scores for all models for hyperparamethers tuning and deep learning comparison.

        Args:
            summary_scores_path (str): path to the txt file, where the summary of all scores will be saved.

    perform_statistical_testing(testing_path)
        Performs all statistical testing needed for conducted experiment.

        Args:
            testing_path (str): path to the txt file, where result of the statistical testing will be saved.

    visualize_scores(plot_path)
        Creates the bar plot for the tuned model parameters in state-of-art method 
        and deep learing methon without augmentation.

        Args:
            plot_path(str):  path to the png file, where bar plot of scores will be saved.
