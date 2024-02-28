# %%
# Import packages
import cProfile
import datetime
import gc
import os
import pickle
import time
from itertools import product

import numpy as np
import tensorflow as tf
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model

from myClassesFunctions import (
    CustomDataSequenceTwoInputsAndAge,
    create_model,
    fit_with_transfer_learning,
    group_by_ID,
)
from prepare_train import prepare_train
from utils import calculate_aic_corr, load_state, save_state, calculate_aic

# %%
# Configure enviroment
# Set level of warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# %%
# Define the folder containing the JPEG files
data_dir = "path/to/jpegs/"
# Define the folder containing the models
data_dir_models = "[ROOT]/data"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
progress_folder = "[ROOT]/progress"
# Define the folder containing variables that will be generated during the training
variables_folder = "[ROOT]/variables"

# %%
# Set general variables
Number_of_KFolds_to_run = 3  # number of folds 3
batch_factors = np.linspace(1, 3, 3).astype(int).tolist()  # 3
do_weights = [["id"], ["age", "sex", "id"], False]
model_names = ["DBN_VGG16.h5" "DBN_InceptionResnetv2.h5"]  #
losses = [mean_squared_error]
learning_rates = (7 * np.logspace(-6, -4, 2)).tolist()  # 3
moderates = [False]  # , False]
formulas = [
    "brainage ~ age",
    # "brainage ~ age ^ 2",
    "brainage ~ age * modality",
    # "brainage ~ age ^ 2 * modality",
    "brainage ~ age * scanner",
    # "brainage ~ age ^ 2 * scanner",
    # "brainage ~ age * modality * scanner",
    # "brainage ~ age ^ 2 * modality * scanner",
]
correction_method = "clear"  # 'cov' is an option but do not use it unless to you want to get
# spuriously low MAEs
do_wls = [False, True]
layers_to_train = 3
ow = False
useaic = True  # deprecated

# modifying formulas for OLS and/or WLS
formulas = [[f, dw] for f in formulas for dw in do_wls]

now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
variable_file = os.path.join(variables_folder, "cv_grid_setup.pkl")
with open(variable_file, "wb") as f:
    pickle.dump(
        [
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
        ],
        f,
    )

# %%
df_file = os.path.join(progress_folder, "data.pkl")
data = load_state(df_file)
data_df = data["data_df"]
data_dm = data["data_dm"]
modalities = data["modalities"]
scanners = data["scanners"]
train_df_list = data["train_df_list"]
linear_df_list = data["linear_df_list"]
valid_df_list = data["valid_df_list"]
Number_of_KFolds = data["Number_of_KFolds"]
final_train_df = data["final_train_df"]
final_linear_df = data["final_linear_df"]
final_eval_df = data["final_eval_df"]
test_df = data["test_df"]
gen = data["gen"]

# %%
# Prepare for training
(
    NFolds,
    grid_with_names,
    dbn_models,
    grid,
    grid_with_folds_with_names,
    shape,
) = prepare_train(
    data_dir_models,
    batch_factors,
    do_weights,
    model_names,
    losses,
    learning_rates,
    moderates,
    Number_of_KFolds,
    Number_of_KFolds_to_run,
)

# %%
# Create strategy object
strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.experimental.CentralStorageStrategy()

# %%
# Define arrays of measures
state_file = os.path.join(progress_folder, "state.pkl")
state = load_state(state_file)
do_crossvalidate = True
if (state == {}) or ow:
    mae_valid_array = np.zeros(shape)
    corrected_mae_valid_array = np.zeros(shape, dtype="object")
    inf_list = [float("inf")] * len(formulas)
    corrected_mae_valid_array.fill(inf_list)
    aic_array = np.zeros(shape, dtype="object")
    aic_array.fill(inf_list)
    aic_corr_array = np.zeros(shape, dtype="object")
    aic_corr_array.fill(inf_list)
    index = 0
    actual_epochs_list = []
    # ---
    fold0 = 0
    batch_factor0 = batch_factors[0]
    do_weight0 = do_weights[0]
    dbn_model0 = dbn_models[0]
    loss0 = losses[0]
    learning_rate0 = learning_rates[0]
    moderate0 = moderates[0]
else:
    mae_valid_array = state["mae_valid_array"]
    corrected_mae_valid_array = state["corrected_mae_valid_array"]
    aic_array = state["aic_array"]
    aic_corr_array = state["aic_corr_array"]
    index = state["index"]
    actual_epochs_list = state["actual_epochs_list"]
    # ---
    if index + 1 <= len(grid_with_folds_with_names):
        grid_state = dict(grid_with_folds_with_names[index])
        fold0 = grid_state["fold"]
        batch_factor0 = grid_state["batch_factor"]
        do_weight0 = grid_state["do_weight"]
        dbn_model0 = grid_state["dbn_model"]
        loss0 = grid_state["loss"]
        learning_rate0 = grid_state["learning_rate"]
        moderate0 = grid_state["moderate"]
    else:
        do_crossvalidate = False

# Cross validation
if do_crossvalidate:
    profiler = cProfile.Profile()
    profiler.enable()
    for batch_factor in batch_factors[batch_factors.index(batch_factor0) :]:
        # Start cros-validation loop
        flow_args = {
            "directory": data_dir,
            "has_ext": False,
            "batch_size": 80 * batch_factor,
            "seed": 42,
            "shuffle": False,
            "class_mode": "other",
        }

        for fold in range(fold0, NFolds):
            flow_args["shuffle"] = False
            valid_sequence = CustomDataSequenceTwoInputsAndAge(
                gen=gen,
                dataframe=valid_df_list[fold],
                x_col=["File_name", "modality", "scanner", "age"],
                y_col="age",
                flow_args=flow_args,
                modalities=modalities,
                scanners=scanners,
                use_sample_weights=False,
            )
            step_size_valid = -(-valid_sequence.dataflow1.n // valid_sequence.dataflow1.batch_size)

            linear_sequence = CustomDataSequenceTwoInputsAndAge(
                gen=gen,
                dataframe=linear_df_list[fold],
                x_col=["File_name", "modality", "scanner", "age"],
                y_col="age",
                flow_args=flow_args,
                modalities=modalities,
                scanners=scanners,
                use_sample_weights=False,
            )
            step_size_linear = -(
                -linear_sequence.dataflow1.n // linear_sequence.dataflow1.batch_size
            )

            # Set shuffle to True
            flow_args["shuffle"] = True
            for do_weight in do_weights[do_weights.index(do_weight0) :]:
                train_sequence = CustomDataSequenceTwoInputsAndAge(
                    gen=gen,
                    dataframe=train_df_list[fold],
                    x_col=["File_name", "modality", "scanner", "age"],
                    y_col="age",
                    flow_args=flow_args,
                    modalities=modalities,
                    scanners=scanners,
                    use_sample_weights=do_weight,
                )
                step_size_train = -(
                    -train_sequence.dataflow1.n // train_sequence.dataflow1.batch_size
                )

                for dbn_model in dbn_models[dbn_models.index(dbn_model0) :]:
                    dbnmodel = load_model(dbn_model)
                    for loss, learning_rate, moderate in product(
                        losses[losses.index(loss0) :],
                        learning_rates[learning_rates.index(learning_rate0) :],
                        moderates[moderates.index(moderate0) :],
                    ):
                        with strategy.scope():
                            start_time = time.time()

                            model = create_model(
                                dbnmodel,
                                layers_to_train,
                                len(modalities),
                                len(scanners),
                                moderate,
                            )
                            # Fit and correct the model
                            (
                                corrected_models,
                                results,
                                actual_epochs,
                                _,
                                _,
                            ) = fit_with_transfer_learning(
                                model,
                                layers_to_train,
                                loss,
                                learning_rate,
                                train_sequence,
                                step_size_train,
                                linear_sequence,
                                step_size_linear,
                                valid_sequence,
                                step_size_valid,
                                linear_df_list[fold],
                                formulas,
                                workers=16,
                                # epochs=[1, 0],
                            )

                            # Append the corrected model to the list
                            actual_epochs_list.append(actual_epochs)

                            # Predict all brain ages to calculate brain age per subject in the
                            # validation data and add the predictions as a new column in valid_df
                            valid_sequence.on_epoch_end()
                            valid_brainage_slices = model.predict(
                                valid_sequence,
                                verbose=1,
                                steps=step_size_valid,
                            )
                            valid_df = valid_df_list[fold].copy()
                            valid_df.loc[:, "brainage_slices"] = valid_brainage_slices

                            valid_grouped_agg = group_by_ID(valid_df)

                            # Calculate the mean absolute error between the predicted and true age
                            # values
                            mae_valid_array.flat[index] = mean_absolute_error(
                                valid_grouped_agg["age"], valid_grouped_agg["brainage"]
                            )

                            mae_per_formulas = []
                            aic_per_formulas = []
                            aic_corr_per_formulas = []
                            for f, _ in enumerate(formulas):
                                # Predict all corrected brain ages to calculate brain age per
                                # subject in the validation data and add the predictions as a new
                                # column in valid_df
                                valid_sequence.on_epoch_end()
                                # corrected_models[f].run_eagerly = True
                                valid_corrected_brainage_slices = corrected_models[f].predict(
                                    valid_sequence,
                                    verbose=1,
                                    steps=step_size_valid,
                                )
                                valid_df.loc[:, "corrected_brainage_slices"] = (
                                    valid_corrected_brainage_slices
                                )

                                # Calculate agregated dataframe
                                valid_grouped_agg = group_by_ID(valid_df)

                                # Calculate the mean absolute error between the predicted and true
                                # ages
                                mae = mean_absolute_error(
                                    valid_grouped_agg["age"],
                                    valid_grouped_agg["corrected_brainage"],
                                )
                                mae_per_formulas.append(mae)

                                # Calculate the aic
                                aic = calculate_aic(results[f], valid_grouped_agg, "brainage")
                                aic_per_formulas.append(aic)
                                aic_corr = calculate_aic_corr(results[f], valid_grouped_agg)
                                aic_corr_per_formulas.append(aic)

                            corrected_mae_valid_array.itemset(index, mae_per_formulas)
                            aic_array.itemset(index, aic_per_formulas)
                            aic_corr_array.itemset(index, aic_corr_per_formulas)

                            # Dump profiler results after the index
                            profiler.dump_stats(f"{progress_folder}/iteration_{index}.profile")

                            # increase the index of the arrays
                            print(
                                "{}/{} --> {}".format(
                                    index + 1,
                                    len(grid_with_folds_with_names),
                                    grid_with_folds_with_names[index],
                                )
                            )

                            index += 1

                            # Save state to resume if interrupted
                            state = {
                                "mae_valid_array": mae_valid_array,
                                "corrected_mae_valid_array": corrected_mae_valid_array,
                                "aic_array": aic_array,
                                "aic_corr_array": aic_corr_array,
                                "index": index,
                                "actual_epochs_list": actual_epochs_list,
                            }
                            save_state(state_file, state)

                            end_time = time.time()
                            print(
                                f"\nRuntime of the whole training: {end_time - start_time} seconds"
                            )

                            del (
                                state,
                                model,
                                corrected_models,
                                actual_epochs,
                                valid_df,
                                valid_brainage_slices,
                                valid_corrected_brainage_slices,
                                valid_grouped_agg,
                                mae,
                                mae_per_formulas,
                            )
                            gc.collect()
                    del dbnmodel
                    gc.collect()
                del train_sequence, step_size_train
                gc.collect()
            del valid_sequence, step_size_valid, linear_sequence, step_size_linear
            gc.collect()
        gc.collect()
    profiler.disable()

# Select the configuration of hyperparameters that yield the lowest MAE
mean_mae_array = np.mean(mae_valid_array, axis=1)
min_index = np.argmin(mean_mae_array)
grid_selection = grid[min_index]
grid_with_names_selection = grid_with_names[min_index]

corrected_mean_mae_valid_array = [
    sum(corrected_mae_valid_array[:, fold].flat[min_index][j] for fold in range(0, NFolds))
    / NFolds
    for j in range(len(corrected_mae_valid_array[:, 0].flat[min_index]))
]
mean_aic_array = [
    sum(aic_array[:, fold].flat[min_index][j] for fold in range(0, NFolds)) / NFolds
    for j in range(len(aic_array[:, 0].flat[min_index]))
]
mean_aic_corr_array = [
    sum(aic_corr_array[:, fold].flat[min_index][j] for fold in range(0, NFolds)) / NFolds
    for j in range(len(aic_corr_array[:, 0].flat[min_index]))
]
formula_selection_aic = formulas[np.argmin(mean_aic_array)]
formula_selection_aic_corr = formulas[np.argmin(mean_aic_corr_array)]
formula_selection = formulas[np.argmin(corrected_mean_mae_valid_array)]
# formula_selection = formulas[0]  # the MAE is virtually identical between the selected and the
# simplest, so we chose the simplest

# %%
# Save all local variables
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
variable_file = os.path.join(variables_folder, "variables_after_cv.pkl")
with open(variable_file, "wb") as f:
    pickle.dump(
        [
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
            formulas,
            formula_selection_aic,
            formula_selection_aic_corr,
            formula_selection,
            date_string,
        ],
        f,
    )

# %%
# Train with the whole train data
final_flow_args = {
    "directory": data_dir,
    "has_ext": False,
    "batch_size": 80 * grid_selection[0],
    "seed": 42,
    "shuffle": True,
    "class_mode": "other",
}
final_train_sequence = CustomDataSequenceTwoInputsAndAge(
    gen=gen,
    dataframe=final_train_df,
    x_col=["File_name", "modality", "scanner", "age"],
    y_col="age",
    flow_args=final_flow_args,
    modalities=modalities,
    scanners=scanners,
    use_sample_weights=grid_selection[1],
)
step_size_train_final = -(
    -final_train_sequence.dataflow1.n // final_train_sequence.dataflow1.batch_size
)

final_flow_args["shuffle"] = False
final_eval_sequence = CustomDataSequenceTwoInputsAndAge(
    gen=gen,
    dataframe=final_eval_df,
    x_col=["File_name", "modality", "scanner", "age"],
    y_col="age",
    flow_args=final_flow_args,
    modalities=modalities,
    scanners=scanners,
    use_sample_weights=False,
)
step_size_valid_final = -(
    -final_eval_sequence.dataflow1.n // final_eval_sequence.dataflow1.batch_size
)

final_linear_sequence = CustomDataSequenceTwoInputsAndAge(
    gen=gen,
    dataframe=final_linear_df,
    x_col=["File_name", "modality", "scanner", "age"],
    y_col="age",
    flow_args=final_flow_args,
    modalities=modalities,
    scanners=scanners,
    use_sample_weights=False,
)
step_size_linear_final = -(
    -final_linear_sequence.dataflow1.n // final_linear_sequence.dataflow1.batch_size
)

dbnmodel = load_model(grid_selection[2])
# Fit and correct the model
with strategy.scope():
    model = create_model(
        dbnmodel, layers_to_train, len(modalities), len(scanners), [grid_selection[5]]
    )
    (
        final_corrected_models,
        final_results,
        final_actual_epochs,
        final_history,
        final_times,
    ) = fit_with_transfer_learning(
        model,
        layers_to_train,
        grid_selection[3],
        grid_selection[4],
        final_train_sequence,
        step_size_train_final,
        final_linear_sequence,
        step_size_linear_final,
        final_eval_sequence,
        step_size_valid_final,
        final_linear_df,
        [formula_selection_aic, formula_selection_aic_corr, formula_selection],
        epochs=[10, 10],
        workers=16,
    )
final_corrected_model_aic = final_corrected_models[0]
final_corrected_model_aic.compile(loss="mse", optimizer="adam")
final_corrected_model_aic_corr = final_corrected_models[1]
final_corrected_model_aic_corr.compile(loss="mse", optimizer="adam")
final_corrected_model = final_corrected_models[2]
final_corrected_model.compile(loss="mse", optimizer="adam")

_, filename = os.path.split(grid_selection[2])
final_corrected_model_aic.save(os.path.join(results_folder, "new_aic_" + filename))
final_corrected_model_aic_corr.save(os.path.join(results_folder, "new_aic_corr_" + filename))
final_corrected_model.save(os.path.join(results_folder, "new_" + filename))
model.save(os.path.join(results_folder, "trained_" + filename))

# Save all local variables
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
variable_file = os.path.join(variables_folder, "variables_final_model.pkl")
with open(variable_file, "wb") as f:
    pickle.dump(
        [
            final_flow_args,
            step_size_train_final,
            step_size_valid_final,
            step_size_linear_final,
            final_results,
            final_actual_epochs,
            final_history,
            final_times,
            date_string,
        ],
        f,
    )
