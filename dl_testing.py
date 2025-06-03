import numpy as np
import pandas as pd

fall_precissions = []
for i in range(10):
    print(f"Results for fold {i}")
    print("Prediction report")
    scores = np.load(f"scores/deep_learning_scores/model_free_shuffle_with_seed_fold{i}_prediction_report.npy", allow_pickle = True).item()
    print(scores)
    fall_precissions.append(scores['winter']['support'])

    print("----------------------------------------------------------------------------------------")

print(sum(fall_precissions)/len(fall_precissions))
print(np.std(fall_precissions))