"""
Saves k fold assignments into csv in order to compare DL and ML approach.
"""



import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

def k_fold_assignment(csv_file, csv_output):
    """
    This function performs a stratified 5-fold split of the input dataset 
    and saves the fold assignment for each file into a CSV file.

    Parameters:
    csv_file (str): Path to the input CSV file.
    csv_output (str): Path to the output CSV fie.

    Output:
    - Saves fold assigments into csv_output file.
    """

    df_full = pd.read_csv(csv_file)

    filename_column = df_full.columns[1]
    label_column = df_full.columns[-1]

    X = df_full.iloc[:, 2:-1]
    y = df_full[label_column]

    rskf = RepeatedStratifiedKFold(n_repeats=1, n_splits=5, random_state=100)

    results = []

    for fold_num, (train_index, test_index) in enumerate(rskf.split(X, y)):
        test_filenames = df_full.iloc[test_index][filename_column]
        test_labels = df_full.iloc[test_index][label_column]

        for filename, label in zip(test_filenames, test_labels):
            results.append({
                "filename": filename,
                "label": label,
                "kfold": fold_num
            })

    df_folds = pd.DataFrame(results)
    df_folds.to_csv(csv_output, index=False)