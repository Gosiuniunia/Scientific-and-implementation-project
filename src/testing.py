"""
Statistical tests, tables and visualization for metrics.  
"""

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


def print_scores(classifier_name, feature_types=["all", "HSV", "Lab"],  round=None, table_style="grid", T=False):
    """
    Generates and prints tables of scores (mean and standard deviation) for different metrics
    based on the results stored in .npy files for the chosen classifier of Feature-based Machine Learning. 

    Args:
        classifier_name (str): The name of a classifier (e.g., "DT", "KNN").
        feature_types (list[str]): The list of feature types. Defaults to ["all", "HSV", "Lab"].
        round (int, optional): The number of decimals for possible measures rounding
        table_style (str, optional): The formatting style for the table (e.g., "latex", "grid"). Defaults to "grid"
        T (bool, optional): Argument, which controls whether the table should be transposed. Defaults to False.
    """
    max_depths = list(range(2, 11))

    n_neighbors = [3, 5, 7, 9]
    metrics = ['euclidean', 'manhattan', 'minkowski']
    weights = ['uniform', 'distance']

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Cs = [0.1, 1, 10]
    gammas = ['scale', 'auto']

    n_estimators = [50, 100, 200]
    max_depth = [5, 10, 20]

    # A list of the parameters permutation - models names, used as column headers.
    if classifier_name.lower() == "dt":
        model_names = max_depths
    elif classifier_name.lower() == "rf":
        model_names = [f"{x}-{y}" for x in n_estimators for y in max_depth]
    elif classifier_name.lower() == "knn":
        model_names = [f"{x}-{y[:3]}-{z[:3]}" for x in n_neighbors for y in metrics for z in weights]
    elif classifier_name.lower() == "svm":
        model_names = [f"{x[:3]}-{y}-{z[0]}" for x in kernels for y in Cs for z in gammas]
    
    metrics = ["Accuracy", "Precision", "Recall", "F1 score"]

    for feature_type in feature_types:
        acc_scores = np.load(f"scores/{classifier_name.lower()}_{feature_type}_accuracies.npy")
        pre_scores = np.load(f"scores/{classifier_name.lower()}_{feature_type}_precisions.npy")
        rec_scores = np.load(f"scores/{classifier_name.lower()}_{feature_type}_recalls.npy")
        f1_scores = np.load(f"scores/{classifier_name.lower()}_{feature_type}_f1s.npy")

        scr = {"Accuracy": acc_scores, "Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores}
        mean_scores = []
        std_scores = []
        for s in scr.keys():
            if round != None:
                mean_scores.append(np.round(np.mean(scr[s], axis=1), round))
                std_scores.append(np.round(np.std(scr[s], axis=1), round))
            else:
                mean_scores.append(np.mean(scr[s], axis=1))
                std_scores.append(np.std(scr[s], axis=1))

        mean_scores = np.array(mean_scores)
        std_scores = np.array(std_scores)
        scores = np.char.add(np.char.add(mean_scores.astype(str), u' \u00B1 '), std_scores.astype(str))
        if T == True:
            scores_T = scores.T
            table = tabulate(scores_T, 
                            tablefmt=table_style, 
                            headers=metrics, 
                            showindex=model_names
            )
        elif T == False:
            table = tabulate(scores, 
                            tablefmt=table_style, 
                            headers=model_names, 
                            showindex=metrics
            )

        if table_style == "grid":
            print(f"\n", f"Scores for {classifier_name} classifiers with {feature_type} features")
            print(table)
        else:
            table_latex = "\\begin{table}[H]\n\centering\n"+ table + f"\n\\vspace{{10pt}}\n\caption{{Scores for {classifier_name} classifiers with {feature_type} features}}\n\label{{tab:{classifier_name.lower()}_{feature_type.lower()}}}\n\end{{table}}\n"
            return table_latex


def print_scores_deep(round=None, table_style="grid", return_scores=False):
    """
    Generates and prints table of scores (mean and standard deviation) for different metrics
    based on the results stored in .npy files for the Deep learning approach. 

    Args:
        round (int, optional): The number of decimals for possible measures rounding
        table_style (str, optional): The formatting style for the table (e.g., "latex", "grid"). Defaults to "grid"
        return_scores (bool, optional): Whether the scores should be returned. Defaults to False. 

    Returns:
        acc_scores, pre_scores, rec_scores, f1_scores (list[float], optional): Metrics values lists.
    """

    model_names = ["No", "Yes"]
    model_files = ['basic', 'model_free']
    metrics = ["Accuracy", "Precision", "Recall", "F1 score"]

    acc_scores = [[], []]
    pre_scores = [[], []]
    rec_scores = [[], []]
    f1_scores = [[], []]

    for i in range(len(model_names)):
        for fold in range(10):
            data = np.load(f"scores/deep_learning_scores/{model_files[i]}_shuffle_with_seed_fold{fold}_prediction_report.npy", allow_pickle=True).item()
            acc_scores[i].append(data['accuracy'])
            pre_scores[i].append(data['macro avg']['precision'])
            rec_scores[i].append(data['macro avg']['recall'])
            f1_scores[i].append(data['macro avg']['f1-score'])

    scr = {"Accuracy": acc_scores, "Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores}
    mean_scores = []
    std_scores = []
    for s in scr.keys():
        if round != None:
            mean_scores.append(np.round(np.mean(scr[s], axis=1), round))
            std_scores.append(np.round(np.std(scr[s], axis=1), round))
        else:
            mean_scores.append(np.mean(scr[s], axis=1))
            std_scores.append(np.std(scr[s], axis=1))

    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    scores = np.char.add(np.char.add(mean_scores.astype(str), u' \u00B1 '), std_scores.astype(str))
    table = tabulate(scores.T, 
                    tablefmt=table_style, 
                    headers=metrics, 
                    showindex=model_names
    )

    if return_scores == True:
        return acc_scores, pre_scores, rec_scores, f1_scores

    if table_style == "grid":
        print(f"\n", "Scores for Deep Learning approach")
        print(table)
    else:
        table_latex = "\\begin{table}[H]\n\centering\n"+ table + f"\n\\vspace{{10pt}}\n\caption{{Scores for Deep Learning approach}}\n\label{{tab:deep}}\n\end{{table}}\n"
        return table_latex


def compare_models(data, metric, model_names, table_style="grid", alpha=0.05, alternative="two-sided"):
    """
    Compares sets of related samples, performs statistical tests (Shapiro-Wilk for normality,
    followed by paired t-test for normal data or Wilcoxon signed-rank test for non-normal data),
    generates and prints a table with p-value of a paired test with the information about which of them 
    was performed ("t" being t-test and "w" being the Wilcoxon).

    Args:
        data (dict[dict]):Dictonary of all scores for chosen models.
        metric (str): Metric on which the tests are based.
        model_names (list[str]): List of the compared model's names.
        table_style (str, optional): The formatting style for the table (e.g., "latex", "grid"). Defaults to "grid".
        alpha (float, optional): The significance level for the statistical test for normality. Defaults to 0.05.
        alternative (str, optional): The alternative hypothesis for the comparison tests. Can be 'two-sided', 'less', or 'greater'. Defaults to "two-sided".
    """
    scores = np.array([data[key][metric] for key in data if key in model_names])
    stat_matrix = [[None for _ in range(scores.shape[0])] for _ in range(scores.shape[0])]
    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            if i >= j: #comparison with oneself and double is omitted
                stat_matrix[i][j] = "nan"
                continue
            t1, p1 = shapiro(scores[i])
            t2, p2 = shapiro(scores[j])
            if p1 > alpha and p2 > alpha:
                t, p = ttest_rel(scores[i], scores[j], alternative=alternative)
                stat_matrix[i][j] = f"t, {p:.4f}"
            else:
                t, p = wilcoxon(scores[i], scores[j], alternative=alternative)
                stat_matrix[i][j] = f"w, {p:.4f}"

    table = tabulate(stat_matrix,
                    tablefmt=table_style, 
                    headers=model_names, 
                    showindex=model_names)
    
    if table_style == "grid":
        print("\n Matrix of p-values from paired statistical tests between models")
        print(table)
    else:
        table_latex = "\\begin{table}[h!]\n\centering" + table + f"\n\\vspace{{10pt}}\n\caption{{Matrix of p-values from paired statistical tests between models for {metric.lower()}}}\n\end{{table}}\n"
        return table_latex

def load_tuned_models_data():
    """
    Loads all of the model's metrics values for chosen hyperparamethers and deep learning scores.

    Returns:
        data (dict[dict]): Dictonary of all scores for chosen models.
    """
    data = {}
    best_params_num = [19, 19, 19, 14, 14, 14, 7, 8, 8, 4, 3, 4]
    i = 0
    for clf in ["knn", "svm", "rf", "dt"]:
        for over in ["all" , "HSV", "Lab"]:
            pre_scores = np.load(f"scores/{clf.lower()}_{over}_precisions.npy")[best_params_num[i]]
            rec_scores = np.load(f"scores/{clf.lower()}_{over}_recalls.npy")[best_params_num[i]]
            f1_scores = np.load(f"scores/{clf.lower()}_{over}_f1s.npy")[best_params_num[i]]
            acc_scores = np.load(f"scores/{clf.lower()}_{over}_accuracies.npy")[best_params_num[i]]

            scr = {"Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores, "Accuracy":acc_scores}
            data[f"{clf.lower()}_{over.lower()}"] = scr
            i += 1

    a, p, r, f = print_scores_deep(return_scores=True)
    i = 0
    for aug in ["no_aug", "with_aug"]:
        scr = {"Precision":p[i], "Recall":r[i], "F1 score":f[i], "Accuracy":a[i]}
        data[aug] = scr
        i += 1
    
    return data

def load_and_generate_all_scores(summary_scores_path):
    """
    Loads and generates scores for all models for hyperparamethers tuning and deep learning comparison.

    Args:
        summary_scores_path (str): path to the txt file, where the summary of all scores will be saved.
    """
    model_names = ["KNN", "SVM", 'RF', 'DT']
    features = [["all"], ["HSV"], ["Lab"]]
    with open(summary_scores_path, "w", encoding="utf-8") as f:
        for model in model_names:
            for feature in features:
                result = print_scores(model, feature, table_style="latex", round=3, T=True)
                f.write(result)
                f.write('\n\n')
        
        result = print_scores_deep(round=3, table_style="latex")
        f.write(result)
        f.write('\n\n')


def perform_statistical_testing(testing_path):
    """
    Performs all statistical testing needed for conducted experiment.

    Args:
        testing_path (str): path to the txt file, where result of the statistical testing will be saved.

    """
    data = load_tuned_models_data()

    with open(testing_path, "w", encoding="utf-8") as f:
        for clf in ["knn", "svm", "rf", "dt"]:
            model_names = [key for key in data.keys() if clf in key]
            f.write(compare_models(data, "Accuracy", model_names, table_style="latex"))
            f.write("\n")

        model_names = [key for key in data.keys() if "lab" in key] + ["no_aug"]
        f.write(compare_models(data, "Accuracy", model_names, table_style="latex"))
        f.write("\n")


def visualize_scores(plot_path):
    """
    Creates the bar plot for the tuned model parameters in state-of-art method 
    and deep learing methon without augmentation.

    Args:
        plot_path(str):  path to the png file, where bar plot of scores will be saved.
    
    """
    data = load_tuned_models_data()
    labels = list(data.keys())
    metrics = ['Precision', 'Recall', 'F1 score', 'Accuracy']
    n_metrics = len(metrics)
    n_labels = len(labels)
    colors = ['orchid', 'mediumpurple', 'skyblue', 'teal']

    averaged_data = {}
    for key, metric_values in data.items():
        averaged_data[key] = {metric: np.mean(values) for metric, values in metric_values.items()}

    df_avg = pd.DataFrame.from_dict(averaged_data, orient='index')

    fig, ax = plt.subplots(figsize=(16, 6)) 

    width = 0.15 

    x = np.arange(n_labels) 

    for i, metric in enumerate(metrics):
        offset = width * (i - (n_metrics - 1) / 2)
        ax.bar(x + offset, df_avg[metric], width, label=metric, color=colors[i % len(colors)])

    ax.set_ylabel('Uśredniona Wartość Metryki')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right') 
    ax.legend(title='Metryka', loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    ax.set_title("Average metrics values of all folds for final models comparison")

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
