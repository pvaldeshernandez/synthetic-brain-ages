# Codes for retraining DeepBrainNet for synthetic MPRAGEs

## Summary
This toolbox was used to retrain, via transfer learning, [DeepBrainNet](https://github.com/vishnubashyam/DeepBrainNet) models to predict brain age from synthetic research-grade MPRAGEs predicted from clinical-grade MRIs of arbitrary modalities. A DeepBrainNet model is a Convolutional Neural Network (CNN) developed by [Bashyan et al., (2020)](https://doi.org/10.1093%2Fbrain%2Fawaa160) to predict brain age that can be based on different known architectures like InceptionResNetv2, VGG16, etc. The synthetic MPRAGEs can be predicted using [SynthSR](https://github.com/BBillot/SynthSR/tree/main/SynthSR), developed by [Iglesias et al., 2023](https://doi.org/10.1126%2Fsciadv.add3607), or any super-resolution method that has learned the map between the MRI to its corresponding research-grade MPRAGE.

## General workflow
The retraining consists of the following steps:
1. Predict Synthetic MPRAGEs from any MRI (any modality, slice orientation, voxel dimension, clinical-grade or research-grade).
2. Extract the brain, i.e., skull-strip the MRI.
3. Normalize the synthetic MPRAGEs to the FSL's 1mm-isotropic template.
4. Use [Slicer.py](https://github.com/vishnubashyam/DeepBrainNet/blob/master/Script/Slicer.py) (follow their installation requirements, e.g., modules, packages) to save the axial slices into separate image files.
5. Retrain the models.
6. Select the best model and save it.
8. Predict the brain ages, save them, and plot them against the chronological ages.

Note: steps 2 and 3 are interchangeable depending on the adopted strategy, but our [paper](https://doi.org/10.1038/s41598-023-47021-y) offers a route.

### Install dependencies
To [run the workflow](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#run-the-workflow), you will need to install all of the Python libraries that are required. 

The easiest way to install the requirements is with [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
```bash
#!/bin/bash
ml conda
conda create -p /path/to/envs/synthetic_brain_ages_env pip python=3.9 -y
conda config /path/to/envs
conda activate /path/to/envs/synthetic_brain_ages_env
pip install numpy pandas scipy scikit-learn keras matplotlib tensorflow-gpu
```

### Prepare the data
Use [SynthSR](https://github.com/BBillot/SynthSR/tree/main/SynthSR) to predict the synthetic MPRAGEs. Normalize and skull-strip the synthetic MPRAGEs, as well as any original MPRAGE also present in your sample. Perform a careful QC to remove the MRIs that are noisy or were not preprocessed correctly--see our [paper](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#citation) for suggestions.

Use [Slicer.py](https://github.com/vishnubashyam/DeepBrainNet/blob/master/Script/Slicer.py) on the synthetic and original MPRAGEs to save 80 of their axial slices into separate JPEG files in a single folder, preferably in a fast drive. Rename the JPEGs according to:

/path/to/jpg/sub-[ID]_session-01_run-[run_number]_slice-[slice_number].jpg (e.g., sub-1234_session-01_run-12_slice-00.jpg)

"ID" is the unique identifier of the subject and "slice_number" goes from 00 to 79. The variable "run_number" is a unique instance number that refers to a specific combination of modality and repetition (e.g. if the subject has 3 different modalities, with 2, 3, 1 repetitions, "number" will go from 01 to 06). Note that we always set the session as 01 as we consider it a repetition (i.e., "run_number" encodes actual sessions and repetitions indistinctly). However, nothing stops you from using the actual session number and letting "run_number" exclusively encode repetitions within sessions.

These file names (without the folder name) have to be written to a text file named 'slices_filenames.csv'. 

To prepare the folders needed to [run the workflow](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#run-the-workflow), copy the content of the folder [example_data](/example_data) to the [root folder](/)
This folder contains the following folders and files:
 - [data](/example_data/data):
   - [slices_filenames.csv](/example_data/data/slices_filenames.csv). This is an example we provide as a guide to the abovementioned file containing the names of the JPEGs.
   - [participants_data.csv](/example_data/data/participants_data.csv) This is an example of a file containing data from the participants needed for the analysis (see a detailed example below).
 - [progress](/example_data/progress) (empty)
 - [results](/example_data/results) (empty)
 - [variables](/example_data/variables) (empty)

The example we provide in [participants_data.csv](/example_data/data/participants_data.csv) is a table that has the following form ('-' means empty, NaN or undefined):

| UID                    | ID   | modality     | scanner | age | domain_Holdout_01 | domain_KFold_01 | domain_KFold_02 | domain_KFold_03 |
|------------------------|------|--------------|---------|-----|-------------------|-----------------|-----------------|-----------------|
| sub-0002_ses-01_run-02 | 0002 | MPRAGE-SR    | Avanto  | 41  | training          | training        | linear          | training        |
| sub-0002_ses-01_run-04 | 0002 | T1w-SR       | Avanto  | 41  | training          | training        | linear          | training        |
| sub-0002_ses-01_run-07 | 0002 | T2w-SR       | Avanto  | 41  | training          | training        | linear          | training        |
| sub-0003_ses-01_run-02 | 0003 | MPRAGE-SR    | Verio   | 65  | training          | training        | training        | linear          |
| sub-0003_ses-01_run-03 | 0003 | MPRAGE       | Verio   | 65  | training          | training        | training        | linear          |
| ...                    | ...  | ...          | ...     | ... | ...               | ...             | ...             | ...             |
| sub-1989_ses-01_run-07 | 1989 | T2w-SR       | Prisma  | 36  | testing           | -               | -               | -               |
| sub-1989_ses-01_run-08 | 1989 | T2wFLAIR-SR  | Prisma  | 36  | testing           | -               | -               | -               |
| sub-1990_ses-01_run-02 | 1990 | MPRAGE-SR    | Verio   | 21  | testing           | -               | -               | -               |
| sub-1990_ses-01_run-03 | 1990 | MPRAGE       | Verio   | 21  | testing           | -               | -               | -               |
| sub-1990_ses-01_run-08 | 1990 | T2wFLAIR-SR  | Verio   | 21  | testing           | -               | -               | -               |

UID is a unique identifier of the image and follows the structure sub-[ID]_session-01_run-[run_number], the modality of the synthetic MPRAGEs has the suffix '-SR' and the 'domain' columns define membership to training, bias correction, and testing sets, as described in Figure 6 of our [paper](https://doi.org/10.1038/s41598-023-47021-y).

Finally, go to https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 and download the following files:
+ DeepBrainNet_InceptionResnetv2.h5
+ DeepBrainNet_VGG16.h5

Copy these files to [data](/data/) and rename them by substituting "DeepBrainNet" with "DBN".

#### Adding more CNN architectures
Note that more models from https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 can be used as long as line 47 of [train_model_deployed.py](/train_model_deployed.py) is modified accordingly.

We warn that some models may have been saved using an old version of Keras (e.g., 2.2.4). In that case, Keras 2.2.4 must be installed to extract and save the model weights via:
```python
from keras.models import load_model
import pickle

model = load_model("path_to_model")

weights = model.get_weights()
with open('path/to/model/model_weights.pkl', 'wb') as file:
    pickle.dump(model.get_weights(), file)
```
Then, the newer version of Keras (e.g., 2.11.0) used to [run the workflow](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#run-the-workflow) must be reinstalled and used to load the weights and set them to a vanilla version of the architecture (e.g., InceptionResnetV2, etc.):
```python
from keras.models import load_model, Model
from keras.applications import InceptionResnetV2
import pickle

model = InceptionResNetV2(weights='imagenet',input_shape=(182, 218, 3))
new_model = Model(
    model.inputs,
    layers.Dense(1, activation="linear")(
        layers.Dropout(0.8)(layers.Dense(1024, activation="relu")(model.layers[-2].output))
    ),
)
new_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

with open('path/to/model/model_weights.pkl', 'rb') as file:
    weights = pickle.load(file)
new_model.set_weights(weights)

new_model.save('/data/DBN_InceptionResnetv2.h5')
```

### Run the workflow
1. Run [create_data_deployed.py](/create_data_deployed.py) after modifying the piece of code below. This will merge the data in [slices_filenames.csv](/example_data/data/slices_filenames.csv) and [participants_data.csv](/example_data/data/participants_data.csv), create lists of dataframes, image generators, and other variables.
```python
# Directories and files
# Define the path to the csv file containing the list of JPEGs files generated with DeepBrainNet's Slicer.py
csv_file = "[ROOT]/data/slices_filenames.csv"
# Define the path to the table containing the subjects' information
data_file = "[ROOT]/data/participants_data.csv"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
progress_folder = "[ROOT]/progress"
```
2. Run [train_model_deployed.py](/train_model_deployed.py) after modifying the piece of code below. This will re-train the DeepBrainNet model.
```python
# Define the folder containing the models
data_dir_models = "[ROOT]/data"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
progress_folder = "[ROOT]/progress"
# Define the folder containing variables that will be generated during the training
variables_folder = "[ROOT]/variables"
```
More lines in the following piece of code of [train_model_deployed.py](/train_model_deployed.py) could be uncommented if more bias models are desired to be accounted for:
```python
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
```
3. Run [obtain_results.py](/obtain_results_deployed.py) after modifying the piece of code below. This will predict the brain ages in the test data using the winning models, i.e., the CNN model that minimized the bias-uncorrected MAE and the bias correction model that minimized the bias-corrected MAE.
```python
# Define the folder containing the JPEG files
data_dir = "path/to/jpegs"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
progress_folder = "[ROOT]/progress"
# Define the folder containing variables that will be generated after the prediction
variables_folder = "[ROOT]/variables"
```
4. Run [obtain_results_originalDBN_deployed.py](/obtain_results_originalDBN_deployed.py) after modifying exactly like in [obtain_results_deployed.py](/obtain_results_deployed.py) except for:
```python
# Define the selected DeepBrainNet model, un-retrained
selected_model = "data/DBN_[architecture].h5"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results_dbn"
```
This will predict the brain ages in the test set using the original version of the selected CNN model (i.e., without retraining) on original MPRAGEs.

5. Run [selected_results_deployed.py](/selected_results_deployed.py) after modifying:
```python 
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
results_folder_dbn = "[ROOT]/results_dbn"
progress_folder = "[ROOT]/progress"
```
## Using our re-trained model
Use [predict_brainages_deployed.py](/predict_brainages_deployed.py) on new user-provided data after modifying the following piece of code:
```python 
# Define the folder containing the JPEG files
data_dir = "/path/to/jpegs"
# Define the path to the csv file containing the list of JPEGs files generated with DeepBrainNet's Slicer.py
csv_file = "[ROOT]/data/new_slices_filenames.csv"
# Define the folder containing the new participants' data
data_file = "/path/to/new_participants_data.csv"
# Define the folder containing the retrained model or the winner model provided by us
retrained_model = "path/to/retrained_model.h5"
# Define the folder containing the results
results_folder = "path/to/results"
```
The new  file containing the text slices' JPEGs follows the same rules as in [slices_filenames.csv](/example_data/data/slices_filenames.csv) (see [Prepare the data](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#prepare-the-data)), while the text file containing the new participants' data follows the same rules as in [participants_data.csv](/example_data/data/participants_data.csv) (see [Prepare the data](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#prepare-the-data)), except the 'domains' columns are not needed. The path to the retrained model (the one with the bias correction layer) is also needed. Use the one generated after running the previous steps or request ours via pvaldeshernandez@ufl.edu.

## Cite our paper
If you use this code in your research, please acknowledge this work by citing the paper: 

[Valdes-Hernandez, P.A., Laffitte Nodarse, C., Peraza, J.A. et al. Toward MR protocol-agnostic, unbiased brain age predicted from clinical-grade MRIs. Scientific Reports 13, 19570 (2023)](https://doi.org/10.1038/s41598-023-47021-y)
