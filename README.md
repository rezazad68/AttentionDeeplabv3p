# [Attention Deeplabv3+: Multi-level Context Attention Mechanism for Skin Lesion Segmentation](https://www.bioimagecomputing.com/program/selected-contributions/)

Implementation of Attention Deeplabv3+, an extended version of Deeplabv3+ for skin le-sion segmentation by employing the idea of attention mechanism in two stages.We first capture the relationship between the channels of a set of feature mapsby assigning a weight for each channel (i.e., channels attention). Channel atten-tion allows the network to emphasize more on the informative and meaningful channels by a context gating mechanism. It also exploit the second level atten-tion strategy to integrate different layers of the atrous convolution. It helps thenetwork to focus on the more relevant field of view to the target. If this code helps with your research please consider citing the following papers:
</br>
> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [M. Asadi](https://scholar.google.com/citations?hl=en&user=8UqpIK8AAAAJ&view_op=list_works&sortby=pubdate), [Mahmood Fathy](https://scholar.google.com/citations?hl=en&user=CUHdgPcAAAAJ&view_op=list_works&sortby=pubdate) and [Sergio Escalera](https://scholar.google.com/citations?hl=en&user=oI6AIkMAAAAJ&view_op=list_works&sortby=pubdate) "Attention Deeplabv3+: Multi-level Context Attention Mechanism for Skin Lesion Segmentation ",ECCV, 2020, download [link](https://www.bioimagecomputing.com/program/selected-contributions/).

## Updates
- Augest 1, 2020: Complete implemenation for SKin Lesion Segmentation task on three different data set has been released.
- Augest 1, 2020: Paper Accepted in the [ECCV workshop](https://sites.google.com/view/iccv19-vrmi/home?authuser=0]) 2019 (Oral presentation).

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras 
- tensorflow backend

## Run Demo
For training deep model and evaluating on each data set follow the bellow steps:</br>
1- Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `Train_Skin_Lesion_Segmentation.py` for training the model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. </br>
4- For performance calculation and producing segmentation result, run `Evaluate_Skin.py`. It will represent performance measures and will saves related results in `output` folder.</br>

**Notice:**
For training and evaluating on ISIC 2017 and ph2 follow the bellow steps: :</br>
**ISIC 2017**- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18\7`. </br> then Run ` 	Prepare_ISIC2017.py` for data preperation and dividing data to train,validation and test sets. </br>
**ph2**- Download the ph2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract it then Run ` 	Prepare_ph2.py` for data preperation and dividing data to train,validation and test sets. </br>
Follow step 3 and 4 for model traing and performance estimation. For ph2 dataset you need to first train the model with ISIC 2018 data set and then fine-tune the trained model using ph2 dataset.



## Quick Overview
![Diagram of the proposed Attention mechanism](https://github.com/rezazad68/AttentionDeeplabv3p/blob/master/images/aggregation2.png)
### Segmentation visualization
![ISIC 2018](https://github.com/rezazad68/AttentionDeeplabv3p/blob/master/images/result.png)




### Model weights
You can download the learned weights for each dataset in the following table. 

Dataset |Learned weights
------------ | -------------
[ISIC 2018](http://www.isi.uu.nl/Research/Databases/DRIVE/) |[Deeplabv3pa]()
[ISIC 2017](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) |[Deeplabv3pa]()
[Ph2](https://www.kaggle.com/kmader/finding-lungs-in-ct-data/data) | [Deeplabv3pa]()


