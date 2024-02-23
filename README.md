# Codes for retraining DeepBrainNet for synthetic MPRAGEs

## Summary
This toolbox was used to retrain, via transfer learning, [DeepBrainNet](https://github.com/vishnubashyam/DeepBrainNet) models to predict brain age from synthetic research-grade MPRAGEs predicted from clinical-grade MRIs of arbitrary modalities. A DeepBrainNet model is a CNN developed by [Bashyan et al., (2020)](https://doi.org/10.1093%2Fbrain%2Fawaa160) to predict brain age that can be based on different known architectures like InceptionResNetv2, VGG16, etc. The synthetic MPRAGEs can be predicted using [SynthSR](https://github.com/BBillot/SynthSR/tree/main/SynthSR), developed by [Iglesias et al., 2023](https://doi.org/10.1126%2Fsciadv.add3607) or any super-resolution that has learned the map between any MRI to its corresponding research-grade MPRAGE.

## Workflow
The retraining consists of the following steps:
1. Predict Synthetic MPRAGEs from any MRI (any modality, slice orientation, voxel dimension, clinical-grade or research-grade).
2. Extract the brain, i.e., skull-strip the MRI.
3. Normalize the synthetic MPRAGEs to the FSL's 1mm-isotropic template.
4. Use [Slicer.py](https://github.com/vishnubashyam/DeepBrainNet/blob/master/Script/Slicer.py) (follow their instalation requirements, e.g., modules, packages) to save the axial slices into separate image files.

(steps 2 and 3 can be interchangeable depending on what strategy is adopted)

### Install dependencies

### Preparing the data
Folder example_data" contains the following folders and files:
 - data:
   - [slicesdir.csv](/example_data/data/slicesdir.csv)
   - [Tn_linear.csv](/example_data/data/Tn_linear.csv)
 - progress (empty)
 - results (empty)
 - variables (empty)

Copy these folders to the root. 

Go to https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 and download the following files:
+ DeepBrainNet_InceptionResnetv2.h5
+ DeepBrainNet_VGG16.h5

Copy these files to [data](/data/slicesdir.csv) and rename them by substituting "DeepBrainNet" with "DBN".

Note that more models from https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 can be used as long as line 43 of [train_model.py](/train_model.py) is modified accordingly.

Also, some models may have been saved using an old version of Keras. In that case, the older version must be used to get the weights, via:
```
from keras.models import load_model
model = load_model("path_to_model")
weights = model.get_weights()
```

then, in the latest Keras version,
```
model.set_weights(weights)
```

### Run the workflow

## Citation

If you use this code in your research, please acknowledge this work by citing the
paper: https://doi.org/10.1038/s41598-023-47021-y.


