
# R-CST
COMPACT SELECTIVE TRANSFORMER BASED ON INFORMATION ENTROPY FOR FACIAL EXPRESSION RECOGNITION IN THE WILD
## Requirements
- Python=3.8
- PyTorch=1.10
- torchvision=0.11.0
- cudatoolkit=11.3
- matplotlib=3.5.3

## Training & Evaluate
We evaluate QOT on RAF-DB, AffectNet and SFEW. We take RAF-DB as an example to introduce our method.
### Traning
- Step 1: download RAF-DB datasets from official website, and put it into ./datasets
- Step 2: download pre-trained ResNet-50 from [Google Drive]
- Step 3: run main_Upload.py to train R-CST model.
- Step 4: or directly download the pre-generated parameters from [Google Drive], and run q-vit_RAFDB_Upload.py to training QOT module.
### Evaluate
- Step 1: download RAF-DB datasets from official website, and put it into ./datasets
- Step 2: download the checkpoint from [Google Drive], and put it into ./checkpoint_cnn
- Step 3: edit the evaluate_path with path in Step 2 and run main_Upload.py to evaluate R-CST model.
- Step 4: download the pre-generated orthogonal feature from Google Drive
- Step 5: run the evaluate code in q-vit_RAFDB_Upload.py to evaluate R-CST module.

## Pre-trained Model


