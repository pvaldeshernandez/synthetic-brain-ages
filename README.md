# Codes for retraining DeepBrainNet for synthetic MPRAGEs

## Summary

## Workflow

## How to use

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

Copy these files to [data](/data/slicesdir.csv) and rename them by substituting "DeepBrainNet" with "DBN"
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


