# Codes for retraining DeepBrainNet for synthetic MPRAGEs

## Preparing the data
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
Note that mode models from https://upenn.app.box.com/v/DeepBrainNet/folder/120404890511 could be downloaded and line 43 of [train_model.py](/train_model.py)

To create the necessary variables first run
