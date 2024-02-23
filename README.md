# Codes for retraining DeepBrainNet for synthetic MPRAGEs

## Summary
This toolbox was used to retrain, via transfer learning, [DeepBrainNet](https://github.com/vishnubashyam/DeepBrainNet) models to predict brain age from synthetic research-grade MPRAGEs predicted from clinical-grade MRIs of arbitrary modalities. A DeepBrainNet model is a CNN developed by [Bashyan et al., (2020)](https://doi.org/10.1093%2Fbrain%2Fawaa160) to predict brain age that can be based on different known architectures like InceptionResNetv2, VGG16, etc. The synthetic MPRAGEs can be predicted using [SynthSR](https://github.com/BBillot/SynthSR/tree/main/SynthSR), developed by [Iglesias et al., 2023](https://doi.org/10.1126%2Fsciadv.add3607) or any super-resolution that has learned the map between any MRI to its corresponding research-grade MPRAGE.

## Workflow
The retraining consists of the following steps:
1. Predict Synthetic MPRAGEs from any MRI (any modality, slice orientation, voxel dimension, clinical-grade or research-grade).
2. Extract the brain, i.e., skull-strip the MRI.
3. Normalize the synthetic MPRAGEs to the FSL's 1mm-isotropic template.
4. Use [Slicer.py](https://github.com/vishnubashyam/DeepBrainNet/blob/master/Script/Slicer.py) (follow their installation requirements, e.g., modules, packages) to save the axial slices into separate image files.

(steps 2 and 3 can be interchangeable depending on what strategy is adopted)

### Install dependencies
To [run the workflow](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#run-the-workflow), you will need to install all of the Python libraries that are required. 

The easiest way to install the requirements is with Conda.
```
#!/bin/bash
ml conda
conda create -p /path/to/clinicalDeepBrainNet_env pip python=3.9 -y
conda config --append envs_dirs/path/to/clinicalDeepBrainNet_env
source activate /path/to/clinicalDeepBrainNet_env
pip install numpy pandas scipy scikit-learn keras matplotlib tensorflow-gpu
```

### Preparing the data
Use [SynthSR](https://github.com/BBillot/SynthSR/tree/main/SynthSR) to predict the synthetic MPRAGEs. Use [Slicer.py](https://github.com/vishnubashyam/DeepBrainNet/blob/master/Script/Slicer.py) on the synthetic and original MPRAGEs to save their axial slices into separate JPEG files in a single folder in a fast drive. Each filename must have the following format:

/path/to/jpg/Subject[ID]_run[number]_T1_BrainAlig-[slice].jpg

"ID" is the ID of the subject, "number" is a unique instance number that accounts for the session, repetitions, and modality (e.g. if the subject has 3 different modalities, and one is repeated, in session 1 and two modalities in a session 2, "number" takes values from 1 to 6. These filenames names will be listed in [slicesdir.csv](/example_data/data/slicesdir.csv), as explained below. "Slice" is the slice number that goes from 0 to 79. as explained in our paper []

The folder [example_data](/example_data) contains the following folders and files:
 - [data](/example_data/data):
   - [slicesdir.csv](/example_data/data/slicesdir.csv)
   - [Tn_linear.csv](/example_data/data/Tn_linear.csv)
 - [progress](/example_data/progress) (empty)
 - [results](/example_data/results) (empty)
 - [variables](/example_data/variables) (empty)

Copy these folders to the [root](/). 

Go to https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 and download the following files:
+ DeepBrainNet_InceptionResnetv2.h5
+ DeepBrainNet_VGG16.h5

Copy these files to [data](/data/slicesdir.csv) and rename them by substituting "DeepBrainNet" with "DBN".

#### Adding more architectures
Note that more models from https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 can be used as long as line 43 of [train_model.py](/train_model.py) is modified accordingly.
Also, some models may have been saved using an old version of Keras (e.g., 2.2.4). In that case, Keras 2.2.4 must be installed to extract and save the model weights via:
```
from keras.models import load_model
import pickle

model = load_model("path_to_model")

weights = model.get_weights()
with open('path/to/model/model_weights.pkl', 'wb') as file:
    pickle.dump(model.get_weights(), file)
```
Then, the newer version of Keras (e.g., 2.11.0) used to [run the workflow](https://github.com/pvaldeshernandez/Multimodal_DeepBrainNet_Clinical_BrainAge_Training/blob/main/README.md#run-the-workflow) must be reinstalled and used to load the weights and set them to a vanilla version of the architecture (e.g., InceptionResnetV2, etc.):
```
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

## Citation
If you use this code in your research, please acknowledge this work by citing the
paper: https://doi.org/10.1038/s41598-023-47021-y.
