# %%
# Import packages
import os

from keras.losses import mean_absolute_error
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from myClassesFunctions import group_by_ID

from utils import load_state

# %%
# Directories and files
# Define the folder containing the JPEG files
data_dir = "path/to/jpegs"
# Define the selected DeepBrainNet model, un-retrained
selected_model = "data/DBN_[architecture].h5"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results_dbn"
progress_folder = "[ROOT]/progress"
# Define the folder containing variables that will be generated after the prediction
variables_folder = "[ROOT]/variables"

prediction_csv = os.path.join(results_folder, "test_predictions_dbn.csv")
pd_test_csv = os.path.join(results_folder, "pd_test_dbn.csv")

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

# Load winner or published CNN model
dbnmodel = load_model(selected_model)


# %%
# Predict in the final testing data
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
