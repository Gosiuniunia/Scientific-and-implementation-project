import numpy as np
import pandas as pd
import os

def summarize_f1(f1_stats_list):
    f1_scores_df = pd.DataFrame(f1_stats_list, columns=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'], index=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'])
    return f1_scores_df
def summarize_prediction_report(prediction_report):
    pr = pd.DataFrame(prediction_report)
    return pr

def summarize_test_metrics(test_metrics_list):
    tm_df = pd.DataFrame(test_metrics_list, index=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], columns=['Loss', 'Accuracy', 'Precission', 'Recall'])
    return tm_df

def summarize_training_history(training_history):
    th = pd.DataFrame(training_history, index=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'])
    return th

def summarize_predicted_labels(predicted_labels_list):
    pass

def fetch_all_statistics(file_prefix):
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
            stats_filename_path = os.path.join(SCORES_PATH, stats_filename)
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

SCORES_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\Scientific-and-implementation-project\scores"

all_scores_files = os.listdir(SCORES_PATH)

f1_scores, prediction_reports, test_metrics, training_histories, y_preds, y_trues = fetch_all_statistics(file_prefix='fold')
f1_scores_model_free, prediction_reports_model_free, test_metrics_model_free, training_histories_model_free, y_preds_model_free, y_trues_model_free = fetch_all_statistics(file_prefix='model_free')

summarized_f1_basic = summarize_f1(f1_scores)
summarized_f1_model_free = summarize_f1(f1_scores_model_free)

# for pr in prediction_reports:
#     print(summarize_prediction_report(pr))

summarized_test_metrics = summarize_test_metrics(test_metrics)
summarized_test_metrics_model_free = summarize_test_metrics(test_metrics_model_free)
# print(summarized_test_metrics)
# print(summarized_test_metrics_model_free)

for th in training_histories:
    print(summarize_training_history(th))








