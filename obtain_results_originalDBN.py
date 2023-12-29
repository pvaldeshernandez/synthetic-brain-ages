# %%
# Import packages
import os

from keras.losses import mean_absolute_error
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import pandas as pd
import numpy as np

from myClassesFunctions import group_by_ID, myplots

from utils import load_state


class CustomSequence(Sequence):
    def __init__(self, gen, dataframe, x_col, y_col, flow_args):
        # Create a DataFrameIterator instance
        self.dataflow = gen.flow_from_dataframe(
            dataframe=dataframe, x_col=x_col, y_col=y_col, **flow_args
        )

    def __len__(self):
        return len(self.dataflow)

    def __getitem__(self, idx):
        # Get the next batch of data from dataflow
        x, y = self.dataflow[idx]

        return x, y

    def on_epoch_end(self):
        # Shuffle the data in dataflow
        self.dataflow.on_epoch_end()


# %%
# Directories and files
progress_folder = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/progress"
data_dir = "/blue/cruzalmeida/chavilaffitte/DBA_Shands_slices"
data_dir_models = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/data"
results_folder = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results_dbn"
prediction_csv = os.path.join(results_folder, "test_predictions_dbn.csv")
pd_test_csv = os.path.join(results_folder, "pd_test_dbn.csv")
fig_pred = "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/results_dbn/BrainAge_Predictions_dbn.png"

# %%
# Read all saved data needed

df_file = os.path.join(progress_folder, "data.pkl")
data = load_state(df_file)
test_df = data["test_df"]

datagen_args = {
    "rescale": 1.0 / 255,
    "horizontal_flip": False,
    "vertical_flip": False,
    "featurewise_center": False,
    "featurewise_std_normalization": False,
}
gen = ImageDataGenerator(**datagen_args)

# Load original published model
dbnmodel = load_model(
    # "/orange/cruzalmeida/chavilaffitte/software/DeepBrainNet-master/Models/DBN_model.h5"
    "/orange/cruzalmeida/pvaldeshernandez/projects/shands-brainage/data/DBN_VGG16.h5"
)

# %%
# Predict in the final testing data
flow_args = {
    "directory": data_dir,
    "has_ext": False,
    "batch_size": 80 * 10,
    "seed": 42,
    "shuffle": False,
    "class_mode": "other",
}
test_sequence = CustomSequence(
    gen=gen,
    dataframe=test_df,
    x_col="File_name",
    y_col="age",
    flow_args=flow_args,
)

brainage_slices = dbnmodel.predict(
    test_sequence,
    verbose=1,
    steps=-(-test_sequence.dataflow.n // test_sequence.dataflow.batch_size),
)

test_df = test_df.copy()
test_df.loc[:, "brainage_slices"] = brainage_slices

# Calculate agregated dataframe
test_grouped_agg = group_by_ID(test_df)

mae_test = mean_absolute_error(test_grouped_agg["age"], test_grouped_agg["brainage"])
print(mae_test)

# write the predictions on the test set
test_grouped_agg.to_csv(prediction_csv, index=True)
test_df.to_csv(pd_test_csv, index=True)

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

myplots(test_grouped_agg, [fig_pred], y=["brainage"], hue="modality", hue_order=hue_order)
