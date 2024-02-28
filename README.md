# Codes for retraining DeepBrainNet for synthetic MPRAGEs

## Summary
This toolbox was used to retrain, via transfer learning, [DeepBrainNet](https://github.com/vishnubashyam/DeepBrainNet) models to predict brain age from synthetic research-grade MPRAGEs predicted from clinical-grade MRIs of arbitrary modalities. A DeepBrainNet model is a CNN developed by [Bashyan et al., (2020)](https://doi.org/10.1093%2Fbrain%2Fawaa160) to predict brain age that can be based on different known architectures like InceptionResNetv2, VGG16, etc. The synthetic MPRAGEs can be predicted using [SynthSR](https://github.com/BBillot/SynthSR/tree/main/SynthSR), developed by [Iglesias et al., 2023](https://doi.org/10.1126%2Fsciadv.add3607) or any super-resolution that has learned the map between any MRI to its corresponding research-grade MPRAGE.

## General workflow
The retraining consists of the following steps:
1. Predict Synthetic MPRAGEs from any MRI (any modality, slice orientation, voxel dimension, clinical-grade or research-grade).
2. Extract the brain, i.e., skull-strip the MRI.
3. Normalize the synthetic MPRAGEs to the FSL's 1mm-isotropic template.
4. Use [Slicer.py](https://github.com/vishnubashyam/DeepBrainNet/blob/master/Script/Slicer.py) (follow their installation requirements, e.g., modules, packages) to save the axial slices into separate image files.
5. Retrain the models.
6. Select the best model.
7. Generate the results.

(steps 2 and 3 are interchangeable depending on what strategy is adopted)

### Install dependencies
To [run the workflow](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#run-the-workflow), you will need to install all of the Python libraries that are required. 

The easiest way to install the requirements is with Conda.
```bash
#!/bin/bash
ml conda
conda create -p /path/to/clinicalDeepBrainNet_env pip python=3.9 -y
conda config --append envs_dirs/path/to/clinicalDeepBrainNet_env
source activate /path/to/clinicalDeepBrainNet_env
pip install numpy pandas scipy scikit-learn keras matplotlib tensorflow-gpu
```

### Prepare the data
Use [SynthSR](https://github.com/BBillot/SynthSR/tree/main/SynthSR) to predict the synthetic MPRAGEs. Normalize and skull-strip the synthetic MPRAGEs and any original MPRAGE also present in your sample. Perform a careful QC to remove the MRIs that are noisy or were not preprocessed correctly--see our [paper](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#citation) for suggestions.

Use [Slicer.py](https://github.com/vishnubashyam/DeepBrainNet/blob/master/Script/Slicer.py) on the synthetic and original MPRAGEs to save their axial slices into separate JPEG files in a single folder in a fast drive. Each filename must have the following format:

/path/to/jpg/Subject[ID]_run[number]_T1_BrainAlig-[slice].jpg

"ID" is the ID of the subject, and "number" is a unique instance number that refers to a specific modality and repetition (e.g. if the subject has 3 different modalities, with 2, 3, 1 repetitions, "number" will go from 01 to 06.

These filenames names will be listed in [slicesdir.csv](/example_data/data/slicesdir.csv), as explained below. "Slice" is the slice number that goes from 0 to 79. as explained in our [paper](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#citation).

The folder [example_data](/example_data) contains the following folders and files:
 - [data](/example_data/data):
   - [slicesdir.csv](/example_data/data/slicesdir.csv)
   - [Tn_linear.csv](/example_data/data/Tn_linear.csv)
 - [progress](/example_data/progress) (empty)
 - [results](/example_data/results) (empty)
 - [variables](/example_data/variables) (empty)

Copy these folders to the [root folder](/).

Go to https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 and download the following files:
+ DeepBrainNet_InceptionResnetv2.h5
+ DeepBrainNet_VGG16.h5

Copy these files to [data](/data/slicesdir.csv) and rename them by substituting "DeepBrainNet" with "DBN".

* [slicesdir.csv](/data/slicesdir.csv) will contain a list of the names of the JPEG files
* [Tn_linear.csv](//data/Tn_linear.csv) will be the following table:

| ID   | modality  | UID                  | age | Sex    | Race  | scanner | t1s                                                                                                      | domain_Holdout_01 | domain_KFold_01 | domain_KFold_02 | domain_KFold_03 |
| :--- | :---      | :---                 | :---| :---   | :---  | :---    | :---                                                                                                     | :---              | :---            | :---            | :---            |
| 0002 | MPRAGE-SR | sub-0002_ses-01_run-02 | 41 | female | white | Avanto  | /orange/cruzalmeida/pvaldeshernandez/Data/Shands_brainage/torun/Subject0002run02_T1_BrainAligned.nii | training          | training        | linear          | training        |
| 0002 | T1w-SR    | sub-0002_ses-01_run-04 | 41 | female | white | Avanto  | /orange/cruzalmeida/pvaldeshernandez/Data/Shands_brainage/torun/Subject0002run04_T1_BrainAligned.nii | training          | training        | linear          | training        |
| 0002 | T2w-SR    | sub-0002_ses-01_run-07 | 41 | female | white | Avanto  | /orange/cruzalmeida/pvaldeshernandez/Data/Shands_brainage/torun/Subject0002run07_T1_BrainAligned.nii | training          | training        | linear          | training        |
| 0003 | MPRAGE-SR | sub-0003_ses-01_run-02 | 65 | female | white | Verio   | /orange/cruzalmeida/pvaldeshernandez/Data/Shands_brainage/torun/Subject0003run02_T1_BrainAligned.nii | training          | training        | training        | linear          |
| 0003 | MPRAGE    | sub-0003_ses-01_run-03 | 65 | female | white | Verio   | /orange/cruzalmeida/pvaldeshernandez/Data/Shands_brainage/torun/Subject0003run03_T1_BrainAligned.nii | training          | training        | training        | linear          |
| 0004 | T1w-SR    | sub-0004_ses-01_run-04 | 25 | male   | white | Verio   | /orange/cruzalmeida/pvaldeshernandez/Data/Shands_brainage/torun/Subject0004run04_T1_BrainAligned.nii | training          | linear          | training        | training        |
| 0004 | T2w-SR    | sub-0004_ses-01_run-07 | 25 | male   | white | Verio   | /orange/cruzalmeida/pvaldeshernandez/Data/Shands_brainage/torun/Subject0004run07_T1_BrainAligned.nii | training          | linear          | training        | training        |
| 0004 | T1w-SR    | sub-0004_ses-01_run-13 | 25 | male   | white | Verio   | /orange/cruzalmeida/pvaldeshernandez/Data/Shands_brainage/torun/Subject0004run13_T1_BrainAligned.nii | training          | linear          | training        | training        |

Note that, in column "t1s", the nifti file name of the first row contains the string "run02". As explained above, this unique string encodes modality and repetition. At the same time, UID, which is in BIDs format, does the same (they are all ses_01). Unfortunately, this is redundant: for historical reasons, we kept the convention required by the codes in [DeepBrainNet](https://github.com/vishnubashyam/DeepBrainNet). You can feel free to use the same BIDs convention for the file names in t1s.

The modality of the synthetic MPRAGEs has the suffix '-SR'.
The 'domains' columns define membership to training, bias, and testing sets, as described in Figure 6 of our [paper](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#citation).

#### MRI modalities
The tools are implemented to deal with real MPRAGEs and the synthetic MPRAGEs predicted from the following modalities:
* T2w (T2-weighted) 
* T1w (T2-weigthed, non-MPRAGEs)
* T2wFLAIR (T2-weighted FLAIR)
* T1wFLAIR-SR (T1-weighted FLAIR)
* IR (Inversion Recovery)

#### Adding more architectures
Note that more models from https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 can be used as long as line 43 of [train_model.py](/train_model.py) is modified accordingly.
Also, some models may have been saved using an old version of Keras (e.g., 2.2.4). In that case, Keras 2.2.4 must be installed to extract and save the model weights via:
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
* Run [create_data.py](/create_data.py) after modifying:
```python
# Directories and files (change as needed)
# Define the folder containing the nifti files. This is only used to remove the path from the file
# in line 40, to merge the data_df and data_dm DataFrames in line 44.
nii_dir = "/path/to/niftis"
# Define the path to the csv file containing the list of jpg files generated with DeepBrainNet
# Slicer.py
csv_file = "[ROOT]/data/slicesdir.csv"
# Define the path to the table containing the subjects' information
data_file = "[ROOT]/data/Tn_linear.csv"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
progress_folder = "[ROOT]/progress"
```
* Run [train_model.py](/train_model.py) after modifying:
```python
# Define the folder containing the models
data_dir_models = "[ROOT]/data"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
progress_folder = "[ROOT]/progress"
# Define the folder containing variables that will be generated during the training
variables_folder = "[ROOT]/variables"
```

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

* Run [obtain_results.py](/obtain_results.py) after modifying:
```python
# Define the folder containing the JPEG files
data_dir = "path/to/jpegs/DBA_Shands_slices"
# Define the folder containing the models
data_dir_models = "[ROOT]/data"
# Define the folder containing the results and progress files
results_folder = "[ROOT]/results"
progress_folder = "[ROOT]/progress"
# Define the folder containing variables that will be generated after the prediction
variables_folder = "[ROOT]/variables"
```

* Run [obtain_results_originalDBN.py](/obtain_results_originalDBN.py) after modifying exactly like in [obtain_results.py](/obtain_results.py) except for:
```python
results_folder = "[ROOT]/results_dbn"
```
   Note: [obtain_results.py](/obtain_results.py) generates the results for the best re-trained model for all MRIs, while [obtain_results_originalDBN.py](/obtain_results_originalDBN.py) generates the results for the original MPRAGEs using the original [BeepBrainNetModel](https://github.com/vishnubashyam/DeepBrainNet/blob/master/Models/DBN_model.h5)

* Run [selected_results.py](/obtain_results.py) after modifying:
```python 
project_folder = "[ROOT]"
```
* Run [modality_comparisons.py](/modality_comparisons.py) after modifying:
```python 
# Define the folders containing the results
results_folder = "[ROOT]/results"
results_folder_dbn = "[ROOT]/results_dbn"
``` 
## Using our re-trained model
Use [predict_brainages.py](/predict_brainages.py) on new user-provided data. The participants' data need to be provided similar to that in file [Tn_linear.csv](//data/Tn_linear.csv) (see [Prepare the data](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#prepare-the-data)).

The path to the retrained model (the one with the bias correction layer) is also needed. Use yours or request ours via pvaldeshernandez@ufl.edu.

## Cite our paper
If you use this code in your research, please acknowledge this work by citing the
paper: https://doi.org/10.1038/s41598-023-47021-y.
