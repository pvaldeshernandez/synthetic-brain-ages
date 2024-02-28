# %%
# Import packages
import os
import pickle

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import mean_absolute_error
from keras.models import load_model, Model

from myClassesFunctions import (
    CustomDataSequenceTwoInputsAndAge,
    group_by_ID,
)

# %%
# Directories and files
# Define the folder containing the JPEG files
data_dir = "/path/to/jpegs"
# Define the folder containing the new participants' data
data_file = "/path/to/new_participants_data.csv"
# Define the folder containing the retrained model
retrained_model = "path/to/retrained_model.h5"
# Define the folder containing the results
results_folder = "path/to/results"

prediction_csv = os.path.join(results_folder, "test_predictions.csv")
pd_test_csv = os.path.join(results_folder, "pd_test.csv")

# %%
# Read all the variables
test_df = pd.read_csv(data_file)
modalities = test_df["modality"].unique()
scanners = test_df["scanner"].unique()

# Read the model
final_corrected_model = load_model(retrained_model)
final_model = Model(
    inputs=final_corrected_model.input, outputs=final_corrected_model.layers[-4].output
)

# %%
# Create generators and sequences for validation and test data
datagen_args = {
    "rescale": 1.0 / 255,
    "horizontal_flip": False,
    "vertical_flip": False,
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    # "brightness_range": [0.9, 1.1],  # adds random brightness
    # "preprocessing_function": utils.add_gaussian_noise,  # adds Gaussian noise
}
flow_args = {
    "directory": data_dir,
    "has_ext": False,
    "batch_size": 400,
    "seed": 42,
    "shuffle": False,
    "class_mode": "other",
}

test_sequence = CustomDataSequenceTwoInputsAndAge(
    gen=ImageDataGenerator(**datagen_args),
    dataframe=test_df,
    x_col=["File_name", "modality", "scanner", "age"],
    y_col="age",
    flow_args=flow_args,
    modalities=modalities,
    scanners=scanners,
    use_sample_weights=False,
)

# Predict the brainage
step_size_test = -(-test_sequence.dataflow1.n // test_sequence.dataflow1.batch_size)

brainage_slices = final_model.predict(test_sequence, verbose=1, steps=step_size_test)
corrected_brainage_slices = final_corrected_model.predict(
    test_sequence, verbose=1, steps=step_size_test
)

test_df = test_df.copy()
test_df.loc[:, "brainage_slices"] = brainage_slices
test_df.loc[:, "corrected_brainage_slices"] = corrected_brainage_slices

# Calculate agregated dataframe
test_grouped_agg = group_by_ID(test_df)

# Calculate the mean absolute error between the predicted and true age values
mae_test = mean_absolute_error(test_grouped_agg["age"], test_grouped_agg["brainage"])
corrected_mae_test = mean_absolute_error(
    test_grouped_agg["age"], test_grouped_agg["corrected_brainage"]
)

# write the predictions on the test set
test_grouped_agg.to_csv(prediction_csv, index=True)
test_df.to_csv(pd_test_csv, index=True)

# %%
# Save all local variables
variable_file = os.path.join(results_folder, "results.pkl")
with open(variable_file, "wb") as f:
    pickle.dump(
        [
            mae_test,
            corrected_mae_test,
        ],
        f,
    )
