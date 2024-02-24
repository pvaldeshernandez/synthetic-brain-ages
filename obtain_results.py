# %%
# Import packages
import datetime
import os
import pickle

import pandas as pd
from keras.losses import mean_absolute_error
from keras.models import load_model, Model

from myClassesFunctions import (
    CustomDataSequenceTwoInputsAndAge,
    cronbach_from_df,
    group_by_ID,
)
from utils import load_state

# % Use AIC
aic = ""  # "_aic_corr"  # "_aic" ""

# %%
# Directories and files
# Directories and files
# Define the folder containing the JPEG files
data_dir = "/blue/cruzalmeida/chavilaffitte/DBA_Shands_slices"
# Define the folder containing the models
data_dir_models = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/data"
# Define the folder containing the results and progress files
results_folder = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results"
progress_folder = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/progress"
# Define the folder containing variables that will be generated during the training
variables_folder = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/variables"

train_prediction_csv = os.path.join(results_folder, "train_predictions.csv")
prediction_csv = os.path.join(results_folder, "test_predictions" + aic + ".csv")
pd_test_csv = os.path.join(results_folder, "pd_test" + aic + ".csv")

# %%
# Read all saved data needed
variable_file = os.path.join(variables_folder, "cv_grid_setup.pkl")
with open(variable_file, "rb") as f:
    data = pickle.load(f)

(
    Number_of_KFolds_to_run,
    batch_factors,
    do_weights,
    model_names,
    losses,
    learning_rates,
    moderates,
    formulas,
    layers_to_train,
    ow,
    date_string,
) = data

df_file = os.path.join(progress_folder, "data.pkl")
data = load_state(df_file)
modalities = data["modalities"]
scanners = data["scanners"]
test_df = data["test_df"]
final_train_df = data["final_train_df"]
gen = data["gen"]

variable_file = os.path.join(variables_folder, "variables_after_cv.pkl")
with open(variable_file, "rb") as f:
    data = pickle.load(f)

(
    mae_valid_array,
    mean_mae_array,
    min_index,
    grid,
    grid_selection,
    grid_with_names,
    grid_with_names_selection,
    corrected_mae_valid_array,
    useaic,
    corrected_mean_mae_valid_array,
    mean_aic_array,
    _,
    formula_selection_aic,
    formula_selection_aic_corr,
    formula_selection,
    _,
) = data

print(grid_with_names_selection)
print(formula_selection)

variable_file = os.path.join(variables_folder, "variables_final_model.pkl")
with open(variable_file, "rb") as f:
    data = pickle.load(f)

(
    final_flow_args,
    step_size_train_final,
    step_size_valid_final,
    step_size_linear_final,
    final_results,
    final_actual_epochs,
    final_history,
    final_times,
    _,
) = data

_, filename = os.path.split(grid_selection[2])
final_corrected_model = load_model(os.path.join(results_folder, "new" + aic + "_" + filename))
# final_model = load_model(os.path.join(results_folder, "trained_" + filename))
final_model = Model(
    inputs=final_corrected_model.input, outputs=final_corrected_model.layers[-4].output
)

# %%
# Predict in the final training data to evaluate the actual regression towards the mean effect
final_flow_args["batch_size"] = 80 * batch_factors[-1]
final_train_sequence = CustomDataSequenceTwoInputsAndAge(
    gen=gen,
    dataframe=final_train_df,
    x_col=["File_name", "modality", "scanner", "age"],
    y_col="age",
    flow_args=final_flow_args,
    modalities=modalities,
    scanners=scanners,
    use_sample_weights=False,
)
step_size_train_final = -(
    -final_train_sequence.dataflow1.n // final_train_sequence.dataflow1.batch_size
)

train_brainage_slices = final_model.predict(
    final_train_sequence, verbose=1, steps=step_size_train_final
)

final_train_df = final_train_df.copy()
final_train_df.loc[:, "brainage_slices"] = train_brainage_slices

# Predict in the held-out test data
final_flow_args["batch_size"] = 80 * batch_factors[-1]
test_sequence = CustomDataSequenceTwoInputsAndAge(
    gen=gen,
    dataframe=test_df,
    x_col=["File_name", "modality", "scanner", "age"],
    y_col="age",
    flow_args=final_flow_args,
    modalities=modalities,
    scanners=scanners,
    use_sample_weights=False,
)
step_size_test = -(-test_sequence.dataflow1.n // test_sequence.dataflow1.batch_size)

brainage_slices = final_model.predict(test_sequence, verbose=1, steps=step_size_test)
corrected_brainage_slices = final_corrected_model.predict(
    test_sequence, verbose=1, steps=step_size_test
)

test_df = test_df.copy()
test_df.loc[:, "brainage_slices"] = brainage_slices
test_df.loc[:, "corrected_brainage_slices"] = corrected_brainage_slices

# Calculate agregated dataframe
final_train_grouped_agg = group_by_ID(final_train_df)
test_grouped_agg = group_by_ID(test_df)

# Calculate the mean absolute error between the predicted and true age values
mae_final_train = mean_absolute_error(
    final_train_grouped_agg["age"], final_train_grouped_agg["brainage"]
)
mae_test = mean_absolute_error(test_grouped_agg["age"], test_grouped_agg["brainage"])
corrected_mae_test = mean_absolute_error(
    test_grouped_agg["age"], test_grouped_agg["corrected_brainage"]
)

# write the predictions on the test set
test_grouped_agg.to_csv(prediction_csv, index=True)
test_df.to_csv(pd_test_csv, index=True)
final_train_grouped_agg.to_csv(train_prediction_csv, index=True)

# %%
# Save all local variables
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
variable_file = os.path.join(variables_folder, "variables_after_test.pkl")
with open(variable_file, "wb") as f:
    pickle.dump(
        [
            final_flow_args,
            step_size_test,
            step_size_train_final,
            mae_final_train,
            mae_test,
            corrected_mae_test,
            final_train_grouped_agg,
            test_grouped_agg,
            final_train_df,
            test_df,
            date_string,
        ],
        f,
    )

# Calculate the Cronbach's alpha
corrected_cronbach_alpha_test = cronbach_from_df(
    test_grouped_agg.loc[:, ["ID", "modality", "corrected_PAD"]]
)

# Calculate and plot median brain age
grouped = test_grouped_agg.groupby(by="ID")
median_test_grouped_agg = grouped.agg(
    {"corrected_brainage": "median", "brainage": "median", "age": "first", "ID": "first"}
)
median_mae_test = mean_absolute_error(
    median_test_grouped_agg["age"], median_test_grouped_agg["corrected_brainage"]
)

# Compare MPRAGEs
mpr_test_grouped_agg = test_grouped_agg[(test_grouped_agg["modality"] == "MPRAGE")]
mpr_mae_test = mean_absolute_error(
    mpr_test_grouped_agg["age"], mpr_test_grouped_agg["corrected_brainage"]
)
mprsr_test_grouped_agg = test_grouped_agg[(test_grouped_agg["modality"] == "MPRAGE-SR")]
mprsr_mae_test = mean_absolute_error(
    mprsr_test_grouped_agg["age"], mprsr_test_grouped_agg["corrected_brainage"]
)
mpr_test_merged_df = pd.merge(
    mpr_test_grouped_agg[["ID", "corrected_brainage"]],
    mprsr_test_grouped_agg[["ID", "corrected_brainage"]],
    on="ID",
    suffixes=("", "SR"),
)
mpr_merged_mae_test = mean_absolute_error(
    mpr_test_merged_df["corrected_brainage"], mpr_test_merged_df["corrected_brainageSR"]
)

# Save all local variables
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
variable_file = os.path.join(variables_folder, "variables_at_end.pkl")
with open(variable_file, "wb") as f:
    pickle.dump(
        [
            corrected_cronbach_alpha_test,
            grouped,
            median_test_grouped_agg,
            median_mae_test,
            mpr_test_grouped_agg,
            mpr_mae_test,
            mprsr_test_grouped_agg,
            mprsr_mae_test,
            mpr_test_merged_df,
            mpr_merged_mae_test,
            date_string,
        ],
        f,
    )
