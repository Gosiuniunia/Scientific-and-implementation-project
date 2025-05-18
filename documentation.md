# utils

This module provides helper functions for facial image processing, including white balancing and color extraction.

## utils/white_balancing

This submodule contains functions related to correcting and adjusting image colors via white balancing.


## utils/color_utils 

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

# face_features_extraction

This script extracts color features from facial regions (iris, skin, eyebrows) using MediaPipe Face Landmarker.
It processes images organized into subdirectories (each representing a class label), computes HSV and LAB color
values, and saves the results in a CSV file.

Usage:
- Place labeled image folders in a root directory (e.g., 'dataset_PColA').
- Set the correct model path (`.task` file).
- Run the script to generate a dataset CSV.

FUNCTIONS

    extract_dataset_to_csv(root_dir)
        Extracts color features (in HSV and LAB color spaces) from images located in subdirectories of the root folder.
        Each subdirectory is treated as a separate class label. The final CSV file will be named after the root folder.

        Args:
            root_dir (str): Root directory containing labeled subdirectories of images.

        Saves:
            - 'root_dir.csv': A DataFrame where each row contains extracted features and a label.

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

# tuning

Hyperparameter tuning for KNN, SVM, and Decision Tree classifiers
using repeated stratified k-fold cross-validation.

FUNCTIONS

    run_tuning(file_name)

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

# testing

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