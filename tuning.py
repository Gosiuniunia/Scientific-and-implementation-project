import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=100)
df = pd.read_csv("deep_armocromia.csv")
df = df.iloc[:, 2:]

X = df.drop("label", axis=1).to_numpy()
y = df["label"].to_numpy()

def tune_knn_params(metrics, weights, rskf):
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
    for param_idx, (metric, weight) in enumerate(param_grid):
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = KNeighborsClassifier(metric=metric, weights=weight)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            accuracies[param_idx, fold_idx] = accuracy_score(y[test], y_pred)
            precisions[param_idx, fold_idx] = precision_score(y[test], y_pred, average='macro', zero_division=0)
            recalls[param_idx, fold_idx] = recall_score(y[test], y_pred, average='macro', zero_division=0)
            f1s[param_idx, fold_idx] = f1_score(y[test], y_pred, average='macro', zero_division=0)
    
    # Zapis wyników
    np.save("knn_accuracies.npy", accuracies)
    np.save("knn_precisions.npy", precisions)
    np.save("knn_recalls.npy", recalls)
    np.save("knn_f1s.npy", f1s)

def tune_svm_params(kernels, Cs, gammas, rskf):
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

    for param_idx, (kernel, C, gamma) in enumerate(param_grid):
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = SVC(C=C, kernel=kernel, gamma=gamma)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            accuracies[param_idx, fold_idx] = accuracy_score(y[test], y_pred)
            precisions[param_idx, fold_idx] = precision_score(y[test], y_pred, average='macro', zero_division=0)
            recalls[param_idx, fold_idx] = recall_score(y[test], y_pred, average='macro', zero_division=0)
            f1s[param_idx, fold_idx] = f1_score(y[test], y_pred, average='macro', zero_division=0)

    np.save("svm_accuracies.npy", accuracies)
    np.save("svm_precisions.npy", precisions)
    np.save("svm_recalls.npy", recalls)
    np.save("svm_f1s.npy", f1s)


def tune_dt_params(max_depths, rskf):
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

    np.save("dt_accuracies.npy", accuracies)
    np.save("dt_precisions.npy", precisions)
    np.save("dt_recalls.npy", recalls)
    np.save("dt_f1s.npy", f1s)



metrics = ['euclidean', 'manhattan', 'minkowski']
weights = ['uniform', 'distance']

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
Cs = [0.1, 1, 10]
gammas = ['scale', 'auto']

max_depths = list(range(2, 11))

tune_knn_params(metrics, weights, rskf)

tune_svm_params(kernels, Cs, gammas, rskf)

tune_dt_params(max_depths, rskf)