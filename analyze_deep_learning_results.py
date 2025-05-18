import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def summarize_f1(f1_stats_list):
    """
    Summarizes list of f1 scores collected during training and testing

    Args:
        f1_stats_list: list of lists of f1 scores collected during training and testing

    Returns:
        f1_scores_df: Pandas dataframe with summary of f1 scores
    """
    f1_scores_df = pd.DataFrame(f1_stats_list, columns=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'], index=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'])
    return f1_scores_df
def summarize_prediction_report(prediction_report):
    """
    Summarizes prediction reports collected during training and testing for each model
    Args:
        prediction_report: a single prediction report, provided as dict

    Returns:
        pr: Pandas dataframe with summary of prediction report

    """
    pr = pd.DataFrame(prediction_report)
    return pr

def summarize_test_metrics(test_metrics_list):
    """
    Summarizes test metrics (loss, accuracy, precission, recall) collected during training and testing for each model
    Args:
        test_metrics_list: test metrics for a single model, provided as dict

    Returns:
        tm: Pandas dataframe with summary of test metrics for given model

    """
    tm_df = pd.DataFrame(test_metrics_list, index=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], columns=['Loss', 'Accuracy', 'Precission', 'Recall'])
    return tm_df

def summarize_training_history(training_history):
    """
    Summarizes training history collected during training and testing for each model
    Args:
        training_history: a dict describing model training history

    Returns:
        th: Pandas dataframe with summary of training history for given model

    """
    th = pd.DataFrame(training_history, index=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'])
    return th

def summarize_predicted_labels(true_labels, predicted_labels, model_no, model_approach):
    """
    Generates a seaborn confusion matrix for given lists of true and predicted labels
    Args:
        true_labels: list of real labels
        predicted_labels: list of predicted by model labels
        model_no: number of currently assesed model, 0 based - numbers from 0 to 4
        model_approach: shortcut of model approach - "basic" - when no data augmentation was applied, "model_free" - when model free augmentation was applied

    Returns:
        None

    """
    class_map = {0: "fall", 1: "spring", 2: "summer", 3: "winter"}
    labels = [0, 1, 2, 3]
    display_labels = ['fall', 'spring', 'summer', 'winter']
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
    plt.xlabel('predicted labels')
    plt.ylabel('true labels')
    plt.title(f'Confusion matrix of PCoIA classification - model {model_no}, {model_approach} ')
    plt.tight_layout()
    plt.savefig(f"scores/confusion_matrices/confusion_matrix_model_{model_no}_{model_approach}.png")

def fetch_all_statistics(file_prefix):
    """
    Fetches all statistics collected during training and testing, assuming following statistics order in scores folder:
    - f1 scores
    - prediction report
    - test metrics
    - training history
    - predicted labels
    - true labels
    Args:
        file_prefix: prefix of statistics file to read: "fold" for model trained without augmentation and "model_free" for model free augmentation applied during the training

    Returns:

    """
    f1_scores = []
    prediction_reports = []
    test_metrics = []
    training_histories = []
    y_preds = []
    y_trues = []

    cnt = 0

    for stats_filename in all_scores_files:
        # picking files related to deep learning approach
        if file_prefix in stats_filename[:10]:
            stats_filename_path = os.path.join(DL_SCORES_PATH, stats_filename)
            stats = np.load(stats_filename_path, allow_pickle=True)
            if cnt % 6 == 0:
                f1_scores.append(stats)
            elif cnt % 6 == 1:
                stats = np.load(stats_filename_path, allow_pickle=True).item()
                prediction_reports.append(stats)
            elif cnt % 6 == 2:
                test_metrics.append(stats)
            elif cnt % 6 == 3:
                stats = np.load(stats_filename_path, allow_pickle=True).item()
                training_histories.append(stats)
            elif cnt % 6 == 4:
                y_preds.append(stats)
            else:
                y_trues.append(stats)
        cnt += 1

    return f1_scores, prediction_reports, test_metrics, training_histories, y_preds, y_trues

DL_SCORES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\Scientific-and-implementation-project\scores\deep_learning_scores"
SCORES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\Scientific-and-implementation-project\scores"

all_scores_files = os.listdir(DL_SCORES_PATH)

f1_scores, prediction_reports, test_metrics, training_histories, y_preds, y_trues = fetch_all_statistics(file_prefix='fold')
f1_scores_model_free, prediction_reports_model_free, test_metrics_model_free, training_histories_model_free, y_preds_model_free, y_trues_model_free = fetch_all_statistics(file_prefix='model_free')

summarized_f1_basic = summarize_f1(f1_scores)
summarized_f1_model_free = summarize_f1(f1_scores_model_free)
# np.save(f"{SCORES_PATH}/f1_scores_basic_summarized", summarized_f1_basic)
# np.save(f"{SCORES_PATH}/f1_scores_model_free_summarized", summarized_f1_model_free)

# cnt = 0
# for pr in prediction_reports:
#     np.save(f"{SCORES_PATH}/prediction_reports_basic_summarized_model_{cnt}", summarize_prediction_report(pr))
#     cnt += 1
#
# cnt = 0
# for pr in prediction_reports_model_free:
#     np.save(f"{SCORES_PATH}/prediction_reports_model_free_summarized_model_{cnt}", summarize_prediction_report(pr))
#     cnt += 1


summarized_test_metrics = summarize_test_metrics(test_metrics)
summarized_test_metrics_model_free = summarize_test_metrics(test_metrics_model_free)

# np.save(f"{SCORES_PATH}/test_metrics_basic_summarized", summarized_test_metrics)
# np.save(f"{SCORES_PATH}/test_metrics_model_free_summarized", summarized_test_metrics_model_free)

cnt = 0
for th in training_histories:
    np.save(f"{SCORES_PATH}/training_history_basic_summarized_model_{cnt}", summarize_training_history(th))
    cnt += 1

cnt = 0
for th in training_histories_model_free:
    np.save(f"{SCORES_PATH}/training_history_model_free_summarized_model_{cnt}", summarize_training_history(th))
    cnt += 1

# cnt = 0
# for yt, yp in zip(y_trues, y_preds):
#     summarize_predicted_labels(yt, yp, cnt, "basic")
#     cnt += 1
#
# cnt = 0
# for yt, yp in zip(y_trues_model_free, y_preds_model_free):
#     summarize_predicted_labels(yt, yp, cnt, "model_free")
#     cnt += 1
