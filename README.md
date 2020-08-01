# Attention Deeplabv3+: Multi-level Context Attention Mechanism for Skin Lesion Segmentation
Implementation code fo paper ID xx,  ISIC Skin Image Analysis Workshop, CVPR 2020

## Updates
- March 22, 2020: Code release for Skin Lesion Segmentatio on three public datasets.

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras 
- tensorflow backend


## Run Demo
For training deep model and evaluating on each data set follow the bellow steps:</br>
1- Download the ISIC 2018 train dataset from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `Train_Skin_Lesion_Segmentation.py` for training the model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. </br>
4- For performance calculation and producing segmentation result, run `Evaluate_Skin.py`. It will represent performance measures and will saves related results in `output` folder.</br>

**Notice:**
For training and evaluating on ISIC 2017 and ph2 follow the bellow steps: :</br>
**ISIC 2017**- Download the ISIC 2017 train dataset from [this](https://challenge.kitware.com/#phase/5841916ccad3a51cc66c8db0) link and extract both training dataset and ground truth folders inside the `dataset_isic18\7`. </br> then Run ` 	Prepare_ISIC2017.py` for data preperation and dividing data to train,validation and test sets. </br>
**ph2**- Download the ph2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract it then Run ` 	Prepare_ph2.py` for data preperation and dividing data to train,validation and test sets. </br>
Follow step 3 and 4 for model traing and performance estimation. For ph2 dataset you need to first train the model with ISIC 2018 data set and then fine-tune the trained model using ph2 dataset.



## Quick Overview
![Diagram of the proposed method](https://github.com/ISIC-CVPR20/attentiondeeplab/blob/master/images/aggregation2.png)
### Segmentation visualization
![ISIC 2018](https://github.com/ISIC-CVPR20/attentiondeeplab/blob/master/images/result.png)
