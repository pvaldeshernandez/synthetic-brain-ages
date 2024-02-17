import pickle
import pandas as pd
from myClassesFunctions import cronbach_from_df, myplots

project_folder = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/"
results_folder = project_folder + "results/"

fig_pred_train = results_folder + "BrainAge_Predictions_Train.png"
fig_pred = results_folder + "BrainAge_Predictions.png"
fig_corrpred = results_folder + "Corrected_BrainAge_Predictions.png"
fig_corrpred_in1 = results_folder + "Corrected_BrainAge_Predictions_in1.png"
fig_corrpred_out1 = results_folder + "Corrected_BrainAge_Predictions_out1.png"
fig_corrpred_in2 = results_folder + "Corrected_BrainAge_Predictions_in2.png"
fig_corrpred_out2 = results_folder + "Corrected_BrainAge_Predictions_out2.png"
fig_corrpad = results_folder + "Corrected_PADs.png"
fig_corrpad_in1 = results_folder + "Corrected_PADs_in1.png"
fig_corrpad_in2 = results_folder + "Corrected_PADs_in2.png"

variable_file = project_folder + "/variables/variables_after_test.pkl"

df_train = pd.read_csv(results_folder + "train_predictions.csv")
df = pd.read_csv(results_folder + "test_predictions.csv")

# Create a boolean vector for rows containing the set of values in column modality
mask = df["modality"].isin(["T1wGRE-SR"])
df = df[~mask]
mask1 = df["modality"].isin(["MPRAGE", "MPRAGE-SR", "T2w-SR", "T1w-SR", "T2wFLAIR-SR"])
mask2 = df["modality"].isin(
    [
        "MPRAGE",
        "MPRAGE-SR",
        "T2w-SR",
        "T1w-SR",
        "T2wFLAIR-SR",
        "T1wFLAIR-SR",
        "IR-SR",
    ]
)

# Use the boolean vector to separate the DataFrame into two DataFrames
df_in1 = df[mask1]
df_out1 = df[~mask1]
df_in2 = df[mask2]
df_out2 = df[~mask2]

corrected_cronbach_alpha_test_in1 = cronbach_from_df(
    df_in1.loc[:, ["ID", "modality", "corrected_PAD"]]
)
print(corrected_cronbach_alpha_test_in1)
corrected_cronbach_alpha_test_in2 = cronbach_from_df(
    df_in2.loc[:, ["ID", "modality", "corrected_PAD"]]
)
print(corrected_cronbach_alpha_test_in2)
corrected_cronbach_alpha_test = cronbach_from_df(df.loc[:, ["ID", "modality", "corrected_PAD"]])
print(corrected_cronbach_alpha_test)

hue_order = [
    "MPRAGE",
    "MPRAGE-SR",
    "T1w-SR",
    "T2w-SR",
    "T1wFLAIR-SR",
    "T2wFLAIR-SR",
    "IR-SR",
    "T2wGRE-SR",
]

myplots(
    df_train,
    [fig_pred_train],
    y=["brainage"],
    hue="modality",
    hue_order=hue_order,
    labels=["Chronological age", "Predicted brain age"],
    bysize=True,
)

myplots(df, [fig_pred], y=["brainage"], hue="modality", hue_order=hue_order)
myplots(df, [fig_corrpred, fig_corrpad], hue="modality", hue_order=hue_order)
myplots(
    df_in1,
    [fig_corrpred_in1, fig_corrpad_in1],
    hue="modality",
    hue_order=hue_order,
)
myplots(
    df_out1,
    [fig_corrpred_out1],
    y=["corrected_brainage"],
    hue="modality",
    hue_order=hue_order,
)
myplots(df_in2, [fig_corrpred_in2, fig_corrpad_in2], hue="modality", hue_order=hue_order)
myplots(
    df_out2,
    [fig_corrpred_out2],
    y=["corrected_brainage"],
    hue="modality",
    hue_order=hue_order,
)

# plot predicted slice in train data to check effect of the median operation on the linear bias
fig_pred_train_slices = results_folder + "BrainAge_Predictions_Train_slices.png"
fig_pred_test_slices = results_folder + "BrainAge_Predictions_Test_slices.png"
with open(variable_file, "rb") as f:
    data = pickle.load(f)
(
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    final_train_df,
    test_df,
    _,
) = data
myplots(
    final_train_df,
    [fig_pred_train_slices],
    y=["brainage_slices"],
    hue="modality",
    hue_order=hue_order,
    labels=["Chronological age", "Predicted brain age per slices"],
)
mask1 = test_df["modality"].isin(["MPRAGE", "MPRAGE-SR", "T2w-SR", "T1w-SR", "T2wFLAIR-SR"])
# Use the boolean vector to separate the DataFrame into two DataFrames
test_df_in1 = test_df[mask1]
myplots(
    test_df_in1,
    [fig_pred_test_slices],
    y=["brainage_slices"],
    hue="modality",
    hue_order=hue_order,
    labels=["Chronological age", "Predicted brain age per slices"],
)
