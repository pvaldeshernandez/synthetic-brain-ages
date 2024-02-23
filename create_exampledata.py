import os
import pandas as pd

folder = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage"
data_file = os.path.join(folder, "data", "Tn_linear.csv")
data_dm = pd.read_csv(data_file)
variables_to_keep = [
    "ID",
    "actual_session",
    "modality",
    "UID",
    "age",
    "Sex",
    "Race",
    "scanner",
    "t1s",
    "domain_Holdout_01",
    "domain_KFold_01",
    "domain_KFold_02",
    "domain_KFold_03",
]
data_dm = data_dm[variables_to_keep]
data_dm[["Race", "Sex", "age"]] = (
    data_dm[["Race", "Sex", "age"]].sample(frac=1).reset_index(drop=True)
)
data_dm.to_csv(os.path.join(folder, "example_data", "data", "Tn_linear.csv"), index=False)
