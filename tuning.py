"""
Hyperparameter tuning for KNN, SVM, and Decision Tree classifiers 
using repeated stratified k-fold cross-validation.

"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def select_features(df, feature_type):
    """
    Selects specific feature columns from the dataframe based on the feature type.

    Args:
        df (pandas.DataFrame): The input dataframe containing all features and the label.
        feature_type (str): The type of features to select. 
            - "all": select all features (all columns except label).
            - "HSV": select only columns ending with '_H', '_S', '_V'.
            - "Lab": select only columns ending with '_L', '_a', '_b'.

    Returns:
        numpy.ndarray: Numpy array of selected features.
    """
    if feature_type == "all":
        return df.to_numpy()
    elif feature_type == "HSV":
        cols = [col for col in df.columns if col.endswith(("_H", "_S", "_V"))]
        return df[cols].to_numpy()
    elif feature_type == "Lab":
        cols = [col for col in df.columns if col.endswith(("_L", "_a", "_b"))]
        return df[cols].to_numpy()

def tune_knn_params(feature_type, X, y, metrics, weights, rskf):
    """
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
    """
    param_grid = list(itertools.product(metrics, weights))
    n_param_combinations = len(param_grid)
    n_folds = rskf.get_n_splits()

    accuracies = np.zeros((n_param_combinations, n_folds))
    precisions = np.zeros_like(accuracies)
    recalls = np.zeros_like(accuracies)
    f1s = np.zeros_like(accuracies)

    scaler = StandardScaler()

    for param_idx, (metric, weight) in enumerate(param_grid):
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
            clf = KNeighborsClassifier(metric=metric, weights=weight)
            clf.fit(X_train, y[train])
            y_pred = clf.predict(X_test)

            accuracies[param_idx, fold_idx] = accuracy_score(y[test], y_pred)
            precisions[param_idx, fold_idx] = precision_score(y[test], y_pred, average='macro', zero_division=0)
            recalls[param_idx, fold_idx] = recall_score(y[test], y_pred, average='macro', zero_division=0)
            f1s[param_idx, fold_idx] = f1_score(y[test], y_pred, average='macro', zero_division=0)
    
    # Zapis wyników
    np.save(f"knn_{feature_type}_accuracies.npy", accuracies)
    np.save(f"knn_{feature_type}_precisions.npy", precisions)
    np.save(f"knn_{feature_type}_recalls.npy", recalls)
    np.save(f"knn_{feature_type}_f1s.npy", f1s)

def tune_svm_params(feature_type, X, y, kernels, Cs, gammas, rskf):
    """
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
    """

    param_grid = list(itertools.product(kernels, Cs, gammas))
    n_param_combinations = len(param_grid)
    n_folds = rskf.get_n_splits()
    
    accuracies = np.zeros((n_param_combinations, n_folds))
    precisions = np.zeros_like(accuracies)
    recalls = np.zeros_like(accuracies)
    f1s = np.zeros_like(accuracies)

    scaler = StandardScaler()

    for param_idx, (kernel, C, gamma) in enumerate(param_grid):
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
            clf = SVC(C=C, kernel=kernel, gamma=gamma)
            clf.fit(X_train, y[train])
            y_pred = clf.predict(X_test)

            accuracies[param_idx, fold_idx] = accuracy_score(y[test], y_pred)
            precisions[param_idx, fold_idx] = precision_score(y[test], y_pred, average='macro', zero_division=0)
            recalls[param_idx, fold_idx] = recall_score(y[test], y_pred, average='macro', zero_division=0)
            f1s[param_idx, fold_idx] = f1_score(y[test], y_pred, average='macro', zero_division=0)

    np.save(f"svm_{feature_type}_accuracies.npy", accuracies)
    np.save(f"svm_{feature_type}_precisions.npy", precisions)
    np.save(f"svm_{feature_type}_recalls.npy", recalls)
    np.save(f"svm_{feature_type}_f1s.npy", f1s)


def tune_dt_params(feature_type, X, y, max_depths, rskf):
    """
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
    """
    n_param = len(max_depths)
    n_folds = rskf.get_n_splits()

    accuracies = np.zeros((n_param, n_folds))
    precisions = np.zeros_like(accuracies)
    recalls = np.zeros_like(accuracies)
    f1s = np.zeros_like(accuracies)

    for param_idx, max_depth in enumerate(max_depths):
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = DecisionTreeClassifier(max_depth=max_depth)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            accuracies[param_idx, fold_idx] = accuracy_score(y[test], y_pred)
            precisions[param_idx, fold_idx] = precision_score(y[test], y_pred, average='macro', zero_division=0)
            recalls[param_idx, fold_idx] = recall_score(y[test], y_pred, average='macro', zero_division=0)
            f1s[param_idx, fold_idx] = f1_score(y[test], y_pred, average='macro', zero_division=0)

    np.save(f"dt_{feature_type}_accuracies.npy", accuracies)
    np.save(f"dt_{feature_type}_precisions.npy", precisions)
    np.save(f"dt_{feature_type}_recalls.npy", recalls)
    np.save(f"dt_{feature_type}_f1s.npy", f1s)

def tune_rf_params(feature_type, X, y, n_estimators, max_depths, rskf):
    """
    Performs hyperparameter tuning for the Random Forest Classifier
    using repeated stratified K-fold cross-validation.

    Args:
        n_estimators (list[str]): Number of estimators.
        max_depths (list[str]): Max tree depth.
        rskf (RepeatedStratifiedKFold): Cross-validation splitting strategy.

    Saves:
        - 'rf_accuracies.npy': Accuracy scores for each parameter combination and fold.
        - 'rf_precisions.npy': Precision scores for each parameter combination and fold.
        - 'rf_recalls.npy': Recall scores for each parameter combination and fold.
        - 'rf_f1s.npy': F1 scores for each parameter combination and fold.
    """
    param_grid = list(itertools.product(n_estimators, max_depths))
    n_param_combinations = len(param_grid)
    n_folds = rskf.get_n_splits()

    accuracies = np.zeros((n_param_combinations, n_folds))
    precisions = np.zeros_like(accuracies)
    recalls = np.zeros_like(accuracies)
    f1s = np.zeros_like(accuracies)

    for param_idx, (n_estimator, max_depth) in enumerate(param_grid):
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            X_train = X[train]
            X_test = X[test]
            clf = RandomForestClassifier(random_state=42, n_estimators=n_estimator, max_depth=max_depth)
            clf.fit(X_train, y[train])
            y_pred = clf.predict(X_test)

            accuracies[param_idx, fold_idx] = accuracy_score(y[test], y_pred)
            precisions[param_idx, fold_idx] = precision_score(y[test], y_pred, average='macro', zero_division=0)
            recalls[param_idx, fold_idx] = recall_score(y[test], y_pred, average='macro', zero_division=0)
            f1s[param_idx, fold_idx] = f1_score(y[test], y_pred, average='macro', zero_division=0)
    
    # Zapis wyników
    np.save(f"rf_{feature_type}_accuracies.npy", accuracies)
    np.save(f"rf_{feature_type}_precisions.npy", precisions)
    np.save(f"rf_{feature_type}_recalls.npy", recalls)
    np.save(f"rf_{feature_type}_f1s.npy", f1s)


def run_tuning(file_name):
    metrics = ['euclidean', 'manhattan', 'minkowski']
    weights = ['uniform', 'distance']

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Cs = [0.1, 1, 10]
    gammas = ['scale', 'auto']

    max_depths = list(range(2, 11))
    feature_types = ["all", "HSV", "Lab"]
    n_estimators = [50, 100, 200]
    max_depth = [5, 10, 20]
    rskf = RepeatedStratifiedKFold(n_repeats=2, n_splits=5, random_state=100)

    df = pd.read_csv(file_name)
    df = df.iloc[:, 2:]

    all_features = df.drop("label", axis=1)
    label = df["label"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(label)
    for feature_type in feature_types:
        X = select_features(all_features, feature_type)

        tune_rf_params(feature_type, X, y, n_estimators, max_depth, rskf)

        tune_knn_params(feature_type, X, y, metrics, weights, rskf)

        tune_svm_params(feature_type, X, y, kernels, Cs, gammas, rskf)

        tune_dt_params(feature_type, X, y, max_depths, rskf)


run_tuning("dataset_PColA.csv")