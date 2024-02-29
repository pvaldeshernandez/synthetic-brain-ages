import os
import pandas as pd
from myClassesFunctions import cronbach_from_df, myplots
from utils import load_state

# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
results_folder_dbn = "[ROOT]/results_dbn"
progress_folder = "[ROOT]/progress"

fig_pred_dbn = os.path.join(results_folder_dbn, "BrainAge_Predictions_OriginalDBN.png")
fig_pred = os.path.join(results_folder, "BrainAge_Predictions.png")
fig_corrpred = os.path.join(results_folder, "Corrected_BrainAge_Predictions.png")
fig_corrpad = os.path.join(results_folder, "Corrected_PADs.png")

df = pd.read_csv(os.path.join(results_folder, "test_predictions.csv"))
df_dbn = pd.read_csv(os.path.join(results_folder_dbn, "test_predictions_dbn.csv"))

corrected_cronbach_alpha_test = cronbach_from_df(df.loc[:, ["ID", "modality", "corrected_PAD"]])
print(corrected_cronbach_alpha_test)

df_file = os.path.join(progress_folder, "data.pkl")
data = load_state(df_file)
modalities = data["modalities"]

myplots(df_dbn, [fig_pred_dbn], y=["brainage"], hue="modality", hue_order=modalities)
myplots(df, [fig_pred], y=["brainage"], hue="modality", hue_order=modalities)
myplots(df, [fig_corrpred, fig_corrpad], hue="modality", hue_order=modalities)
