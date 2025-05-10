import numpy as np
from tabulate import tabulate

from scipy.stats import shapiro, ttest_rel, wilcoxon


def print_dt_scores(param_list, table_style):
    """
    Generates and prints tables of mean and standard deviation for different metrics
    based on the results stored in .npy files for a Decision Tree classifier .

    Args:
        param_list (list[str]): A list of parameter values corresponding to the columns of the tables.
        table_style (str): The formatting style for the table (e.g., "latex", "grid").
    """

    acc_scores = np.load("dt_accuracies.npy")
    pre_scores = np.load("dt_precisions.npy")
    rec_scores = np.load("dt_recalls.npy")
    f1_scores = np.load("dt_f1s.npy")

    scr = {"Accuracy": acc_scores, "Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores}
    mean_scores = []
    std_scores = []
    for s in scr.keys():
        mean_scores.append(np.mean(scr[s], axis=1))
        std_scores.append(np.std(scr[s], axis=1))

    print(f"\n", f"Mean for dt")
    table = tabulate(mean_scores, 
                    tablefmt=table_style, 
                    headers=param_list, 
                    showindex=["Accuracy", "Precision", "Recall", "F1 score"]
    )
    print(table)

    print(f"\n", f"STD for dt")
    table = tabulate(std_scores, 
                    tablefmt=table_style, 
                    headers=param_list, 
                    showindex=["Accuracy", "Precision", "Recall", "F1 score"]
    )
    print(table)

def print_knn_scores(first_param_list, second_param_list, table_style):
    """
    Generates and prints tables of mean and standard deviation for different metrics
    based on the results stored in .npy files for a K-Nearest Neighbors (KNN) classifier.

    Args:
        first_param_list (list): A list of the first parameter's values, used as row indices.
        second_param_list (list): A list of the second parameter's values, used as column headers.
        table_style (str): The formatting style for the table (e.g., "latex", "grid").

    """
    acc_scores = np.load("knn_accuracies.npy")
    pre_scores = np.load("knn_precisions.npy")
    rec_scores = np.load("knn_recalls.npy")
    f1_scores = np.load("knn_f1s.npy")

    scr = {"Accuracy": acc_scores, "Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores}
    for s in scr.keys():
        means = np.mean(scr[s], axis=1)
        stds = np.std(scr[s], axis=1)
        mean_scores = []
        std_scores = []
        print(f"\n", f"Mean {s} for knn")
        for idx in range(len(first_param_list)):
            mean_scores.append(means[idx*len(second_param_list):(idx+1)*len(second_param_list)])
            std_scores.append(stds[idx*len(second_param_list):(idx+1)*len(second_param_list)])

        table = tabulate(mean_scores, 
                        tablefmt=table_style, 
                        headers=second_param_list, 
                        showindex=first_param_list
        )
        print(table)

        print(f"\n", f"STD {s} for knn")
        table = tabulate(std_scores, 
                        tablefmt=table_style, 
                        headers=second_param_list, 
                        showindex=first_param_list
        )
        print(table)

def print_svm_scores(first_param_list, second_param_list, third_param_list, table_style):
    """
    Generates and prints tables of mean and standard deviation for different metrics
    based on the results stored in .npy files for the Support Vector Machine (SVM) classifier 
    and three parameters. 

    Args:
        first_param_list (list): A list of the first parameter's values, used as column headers.
        second_param_list (list): A list of the second parameter's values, used in the row index.
        third_param_list (list): A list of the third parameter's values, used in the row index (combined with the second).
        table_style (str): The formatting style for the table (e.g., "latex", "grid").

    """
    acc_scores = np.load("svm_accuracies.npy")
    pre_scores = np.load("svm_precisions.npy")
    rec_scores = np.load("svm_recalls.npy")
    f1_scores = np.load("svm_f1s.npy")

    scr = {"Accuracy": acc_scores, "Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores}
    indexes = [f"{x}-{y}" for x in second_param_list for y in third_param_list]
    for s in scr.keys():
        means = np.mean(scr[s], axis=1)
        stds = np.std(scr[s], axis=1)
        mean_scores = []
        std_scores = []
        for idx_1 in range(len(second_param_list)*len(third_param_list)):
            mean_scores.append(means[idx_1*len(first_param_list):(idx_1+1)*len(first_param_list)])   
            std_scores.append(stds[idx_1*len(first_param_list):(idx_1+1)*len(first_param_list)])       

        print(f"\n", f"Mean {s} dla svm")
        table = tabulate(mean_scores, 
                        tablefmt=table_style, 
                        headers=first_param_list, 
                        showindex=indexes
        )
        print(table)

        print(f"\n", f"STD {s} for svm")
        table = tabulate(std_scores, 
                        tablefmt=table_style, 
                        headers=first_param_list, 
                        showindex=indexes
        )
        print(table)

#Example usage with the provided parameters:
metrics = ['euclidean', 'manhattan', 'minkowski']
weights = ['uniform', 'distance']

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
Cs = [0.1, 1, 10]
gammas = ['scale', 'auto']

max_depths = list(range(2, 11))

print_knn_scores(metrics, weights, "grid")

print_svm_scores(gammas, kernels, Cs, "grid")

print_dt_scores(max_depths, "grid")



def compare_models(scores, alpha=0.05, alternative="two-sided"):
    """
    Compares sets of related samples, performs statistical tests (Shapiro-Wilk for normality,
    followed by paired t-test for normal data or Wilcoxon signed-rank test for non-normal data),
    and generates a matrix indicating statistical significance based on the provided alpha level.

    Args:
        scores (np.array[float]): Array of samples scores for testing
        alpha (float, optional): The significance level for the statistical tests. Defaults to 0.05.
        alternative (str, optional): The alternative hypothesis for the tests.
                                     Can be 'two-sided', 'less', or 'greater'. Defaults to "two-sided".

    Returns:
        str: A tabular representation of the statistical significance matrix.
             Each cell (i, j) indicates whether the p-value from the comparison
             of the i-th and j-th set of scores is greater than the alpha level
             (True) or not (False), suggesting no statistically significant difference
             at the given alpha.
    """
    stat_matrix = np.zeros((scores.shape[0], scores.shape[0]), dtype=bool)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            print(f"Comparing sample {i+1} and sample {j+1}:")
            print(scores[i])
            print(scores[j])
            t1, p1 = shapiro(scores[i])
            t2, p2 = shapiro(scores[j])
            print(f"  Shapiro-Wilk p-values: p1 = {p1}, p2 = {p2}")
            if p1 > alpha and p2 > alpha:
                t, p = ttest_rel(scores[i], scores[j], alternative=alternative)
                print(f"  Paired t-test p-value: p = {p}")
            else:
                t, p = wilcoxon(scores[i], scores[j], alternative=alternative)
                print(f"  Wilcoxon signed-rank test p-value: p = {p}")
            stat_matrix[i, j] = 1 if p <= alpha else 0

    table = tabulate(stat_matrix)
    print("\nSignificance Matrix (1 = significant difference, 0 = no significant difference):")
    print(table)
    return table

# Example usage with the provided parameters:
alpha = 0.05
alternative_hypothesis = "greater"
scores = np.load("dt_precisions.npy")

comparison_table = compare_models(scores, alpha, alternative_hypothesis)
