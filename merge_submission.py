import numpy as np
import pandas as pd

submissions = [
    "submission_fold1.csv",
    "submission_fold2.csv",
    "submission_fold3.csv",
    "submission_fold4.csv",
    "submission_fold5.csv",
    "submission_fold6.csv",
    "submission_fold7.csv",
    "submission_fold8.csv",
    "submission_fold9.csv",
    "submission_fold10.csv",
    "submission_fold1_best.csv",
    "submission_fold2_best.csv",
    "submission_fold3_best.csv",
    "submission_fold4_best.csv",
    "submission_fold5_best.csv",
]

dfs = [pd.read_csv(sub) for sub in submissions]

merge_sub = dfs[0].copy()
merge_sub.result = np.mean([df.result for df in dfs], axis=0)
merge_sub.to_csv("merge_submission.csv", index=False)
