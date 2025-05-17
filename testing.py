import numpy as np
from tabulate import tabulate

from scipy.stats import shapiro, ttest_rel, wilcoxon

#to do: weronika's part


def print_scores(classifier_name, feature_types=["all", "HSV", "Lab"],  round=None, table_style="grid", T=False):
    """
    Generates and prints tables of mean and standard deviation for different metrics
    based on the results stored in .npy files for the chosen classifier. 

    Args:
        classifier_name (str): The name of a classifier (e.g., "DT", "KNN").
        feature_types (list[str]): The list of feature types. Defaults to ["all", "HSV", "Lab"].
        round (int, optional): The number of decimals for possible measures rounding
        table_style (str, optional): The formatting style for the table (e.g., "latex", "grid"). Defaults to "grid"
        T (bool, optional): Argument, which controls whether the table should be transposed. Defaults to False.

    """
    max_depths = list(range(2, 11))

    metrics = ['euclidean', 'manhattan', 'minkowski']
    weights = ['uniform', 'distance']

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Cs = [0.1, 1, 10]
    gammas = ['scale', 'auto']

    # A list of the parameters permutation - models names, used as column headers.
    if classifier_name.lower() == "dt":
        model_names = max_depths
    elif classifier_name.lower() == "knn":
        model_names = [f"{x[:3]}-{y[:3]}" for x in metrics for y in weights]
    elif classifier_name.lower() == "svm":
        model_names = [f"{x[:3]}-{y}-{z[0]}" for x in kernels for y in Cs for z in gammas]
    
    metrics = ["Accuracy", "Precision", "Recall", "F1 score"]

    for feature_type in feature_types:
        acc_scores = np.load(f"{classifier_name.lower()}_{feature_type}_accuracies.npy")
        pre_scores = np.load(f"{classifier_name.lower()}_{feature_type}_precisions.npy")
        rec_scores = np.load(f"{classifier_name.lower()}_{feature_type}_recalls.npy")
        f1_scores = np.load(f"{classifier_name.lower()}_{feature_type}_f1s.npy")

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

        if T == True:
            mean_scores_T = np.array(mean_scores).T
            std_scores_T = np.array(std_scores).T
            table_mean = tabulate(mean_scores_T, 
                            tablefmt=table_style, 
                            headers=metrics, 
                            showindex=model_names
            )

            table_std = tabulate(std_scores_T, 
                            tablefmt=table_style, 
                            headers=metrics, 
                            showindex=model_names
            )
        elif T == False:
            table_mean = tabulate(mean_scores, 
                            tablefmt=table_style, 
                            headers=model_names, 
                            showindex=metrics
            )

            table_std = tabulate(std_scores, 
                            tablefmt=table_style, 
                            headers=model_names, 
                            showindex=metrics
            )

        if table_style == "grid":
            print(f"\n", f"Mean for {classifier_name} classifiers with {feature_type} features")
            print(table_mean)

            print(f"\n", f"STD for {classifier_name} classifiers with {feature_type} features")
            print(table_std)
        else:
            table_mean_latex = table_mean[:-13] + f"\caption{{Mean for {classifier_name} classifier with {feature_type} features}}\n" + table_mean[-13:]
            table_std_latex = table_std[:-13] + f"\caption{{Mean for {classifier_name} classifier with {feature_type} features}}\n" + table_std[-13:]
            print(table_mean_latex, "\n")
            print(table_std_latex, "\n")


#Example usage:
print_scores("DT", table_style="latex")
print_scores("KNN", round=3)
print_scores("SVM")
print_scores("SVM", T=True, feature_types=['all', 'Lab'])


def compare_models(scores, model_names, table_style="grid", alpha=0.05, alternative="two-sided"):
    """
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
    """
    stat_matrix = [[None for _ in range(scores.shape[0])] for _ in range(scores.shape[0])]
    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            if i == j: #comparison with oneself is omitted
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
        table_latex = table[:-13] + f"\caption{{Matrix of p-values from paired statistical tests between models}}\n" + table[-13:]
        print(table_latex)


# Example usage with the provided parameters:
alpha = 0.05
alternative_hypothesis = "greater"
scores = np.load("dt_all_precisions.npy")
max_depths = list(range(2, 11))

comparison_table = compare_models(scores, max_depths, alpha=alpha, alternative=alternative_hypothesis, table_style="latex")
