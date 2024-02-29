# %%
# Import packages
import os

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from utils import save_state

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Directories and files
# Define the path to the csv file containing the list of jpg files generated with DeepBrainNet's
# Slicer.py
csv_file = "[ROOT]/data/slices_filenames.csv"
# Define the path to the table containing the subjects' information
data_file = "[ROOT]/data/participants_data.csv"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
progress_folder = "[ROOT]/progress"

# %%
# Read data into DataFrames
data_df = pd.read_csv(csv_file, header=None)
data_dm = pd.read_csv(data_file, dtype={"ID": str})

# Rename columns and add new columns to the data_df DataFrame
data_df.columns = ["File_name"]
data_df["UID"] = data_df["File_name"].str.split("_slice").str[0]

# Merge the data_df and data_dm DataFrames using the "name" column as the key
data_df = pd.merge(data_df, data_dm, on="UID", how="left")

# Fit the encoder to the ID categories and transform them to numeric values (deprecated)
data_df[["numeric_ID"]] = data_df[["ID"]].apply(pd.to_numeric)

# Create several lists of DataFrames
train_df_list = []
linear_df_list = []
valid_df_list = []
Number_of_KFolds = sum(data_dm.columns.str.startswith("domain_KFold_"))
for fold in range(1, Number_of_KFolds + 1):
    # Define the column name for the current fold
    kfold_column = f"domain_KFold_{fold:02d}"

    # Select the training and evaluation subjects for the current fold
    indices = (data_dm["domain_Holdout_01"] == "training") | (
        data_dm["domain_Holdout_01"] == "evaluation"
    )
    train_subjects = data_dm[indices & (data_dm[kfold_column] == "training")].ID.values
    linear_subjects = data_dm[indices & (data_dm[kfold_column] == "linear")].ID.values
    eval_subjects = data_dm[indices & (data_dm[kfold_column] == "evaluating")].ID.values

    # Create the DataFrames for the current fold
    train_df_i = data_df[data_df["ID"].isin(train_subjects)]
    linear_df_i = data_df[data_df["ID"].isin(linear_subjects)]
    valid_df_i = data_df[data_df["ID"].isin(eval_subjects)]

    # Append the DataFrames to the lists
    train_df_list.append(train_df_i)
    linear_df_list.append(linear_df_i)
    valid_df_list.append(valid_df_i)

# Create a test DataFrame by selecting a subset of the data from the data
final_train_subjects = data_dm[data_dm["domain_Holdout_01"] == "training"].ID.values
final_train_df = data_df[data_df["ID"].isin(final_train_subjects)]

# Create a final training linear DataFrame by selecting a subset of the data from the data
final_linear_subjects = data_dm[data_dm["domain_Holdout_01"] == "linear"].ID.values
final_linear_df = data_df[data_df["ID"].isin(final_linear_subjects)]

# Create a final evaluation linear DataFrame by selecting a subset of the data from the data
final_eval_subjects = data_dm[data_dm["domain_Holdout_01"] == "evaluation"].ID.values
final_eval_df = data_df[data_df["ID"].isin(final_eval_subjects)]

# Create a test DataFrame by selecting a subset of the data from the data
test_subjects = data_dm[data_dm["domain_Holdout_01"] == "testing"].ID.values
test_df = data_df[data_df["ID"].isin(test_subjects)]

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

gen = ImageDataGenerator(**datagen_args)

# Determine the unique categories in the "modality" column of the total data
modalities = pd.unique(final_train_df["modality"].values)
# Determine the unique categories in the "scanner" column of the total data
scanners = pd.unique(final_train_df["scanner"].values)

# %%
filename = os.path.join(progress_folder, "data.pkl")
state = {
    "data_df": data_df,
    "data_dm": data_dm,
    "modalities": modalities,
    "scanners": scanners,
    "train_df_list": train_df_list,
    "linear_df_list": linear_df_list,
    "valid_df_list": valid_df_list,
    "Number_of_KFolds": Number_of_KFolds,
    "final_train_df": final_train_df,
    "final_linear_df": final_linear_df,
    "final_eval_df": final_eval_df,
    "test_df": test_df,
    "gen": gen,
}
save_state(filename, state)
