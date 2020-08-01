# [Attention Deeplabv3+: Multi-level Context Attention Mechanism for Skin Lesion Segmentation](https://www.bioimagecomputing.com/program/selected-contributions/)

Implementation of Attention Deeplabv3+, an extended version of Deeplabv3+ for skin lesion segmentation by employing the idea of attention mechanism in two stages. In this method, the relationship between the channels of a set of feature maps by assigning a weight for each channel (i.e., channels attention) is captured. In which channel atten-tion allows the network to emphasize more on the informative and meaningful channels by a context gating mechanism. It also exploit the second level attention strategy to integrate different layers of the atrous convolution. It helps thenetwork to focus on the more relevant field of view to the target. If this code helps with your research please consider citing the following papers:
</br>
> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [M. Asadi](https://scholar.google.com/citations?hl=en&user=8UqpIK8AAAAJ&view_op=list_works&sortby=pubdate), [Mahmood Fathy](https://scholar.google.com/citations?hl=en&user=CUHdgPcAAAAJ&view_op=list_works&sortby=pubdate) and [Sergio Escalera](https://scholar.google.com/citations?hl=en&user=oI6AIkMAAAAJ&view_op=list_works&sortby=pubdate) "Attention Deeplabv3+: Multi-level Context Attention Mechanism for Skin Lesion Segmentation ",ECCV, 2020, download [link](https://www.bioimagecomputing.com/program/selected-contributions/).

## Updates
- Augest 1, 2020: Complete implemenation for SKin Lesion Segmentation task on three different data set has been released.
- Augest 1, 2020: Paper Accepted in the [ECCV workshop](https://www.bioimagecomputing.com/program/selected-contributions/) 2020 (Oral presentation).

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
### Diagram of the proposed Attention mechanism
![Diagram of the proposed Attention mechanism](https://github.com/rezazad68/AttentionDeeplabv3p/blob/master/images/aggregation2.png)


#### Performance Evalution on the Skin Lesion Segmentation ISIC 2018

Methods | Year |F1-scores | Sensivity| Specificaty| Accuracy | PC | JS 
------------ | -------------|----|-----------------|----|---- |---- |---- 
Ronneberger and etc. all [U-net](https://arxiv.org/abs/1505.04597)	     	    |2015   | 0.647	|0.708	  |0.964	  |0.890  |0.779 |0.549
Alom  et. all [Recurrent Residual U-net](https://arxiv.org/abs/1802.06955)	|2018	  | 0.679 |0.792 |0.928 |0.880	  |0.741	  |0.581
Oktay  et. all [Attention U-net](https://arxiv.org/abs/1804.03999)	|2018	  | 0.665	|0.717	  |0.967	  |0.897	  |0.787 | 0.566 
Alom  et. all [R2U-Net](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf)	        |2018	  | 0.691	|0.726	  |0.971	  |0.904	  |0.822 | 0.592
Azad et. all [BCDU-Net](https://github.com/rezazad68/LSTM-U-net/edit/master/README.md)	  |2019 	| 0.847	|0.783	  |0.980	  |0.936	  |0.922| 0.936
Azad et. all [MCGU-Net](https://128.84.21.199/pdf/2003.05056.pdf)	  |2020	| 0.895	|0.848	  |0.986	  |0.955	  |0.947| 0.955
Azad et. all [Attention Deeplabv3p](https://www.bioimagecomputing.com/program/selected-contributions/)	  |2020	| **0.912**	|**0.885**	  |**0.988**	  |**0.964**	  |..| **0.964**



### Segmentation visualization
![ISIC 2018](https://github.com/rezazad68/AttentionDeeplabv3p/blob/master/images/result.png)




### Model weights
You can download the learned weights for each dataset in the following table. 

Dataset |Learned weights
------------ | -------------
[ISIC 2018](http://www.isi.uu.nl/Research/Databases/DRIVE/) |[Deeplabv3pa](https://drive.google.com/file/d/10S9ewav837izWaraOlUB8OOQoWY9szzU/view?usp=sharing)
[ISIC 2017](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) |[Deeplabv3pa]()
[Ph2](https://www.kaggle.com/kmader/finding-lungs-in-ct-data/data) | [Deeplabv3pa](https://drive.google.com/file/d/1Ni9PldLL9bMYlyjcRxgDitr-MR6o-RY4/view?usp=sharing)


