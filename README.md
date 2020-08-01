# Attention Deeplabv3+: Multi-level Context Attention Mechanism for Skin Lesion Segmentation (http://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Azad_Bi-Directional_ConvLSTM_U-Net_with_Densley_Connected_Convolutions_ICCVW_2019_paper.pdf)


Deep auto-encoder-decoder network for medical image segmentation with state of the art results on skin lesion segmentation, lung segmentation, and retinal blood vessel segmentation. This method applies bidirectional convolutional LSTM layers in U-net structure to non-linearly encode both semantic and high-resolution information with non-linearly technique. Furthermore, it applies densely connected convolution layers to include collective knowledge in representation and boost convergence rate with batch normalization layers. If this code helps with your research please consider citing the following papers:
</br>
> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [M. Asadi](https://scholar.google.com/citations?hl=en&user=8UqpIK8AAAAJ&view_op=list_works&sortby=pubdate), [Mahmood Fathy](https://scholar.google.com/citations?hl=en&user=CUHdgPcAAAAJ&view_op=list_works&sortby=pubdate) and [Sergio Escalera](https://scholar.google.com/citations?hl=en&user=oI6AIkMAAAAJ&view_op=list_works&sortby=pubdate) "Attention Deeplabv3+: Multi-level Context Attention Mechanism for Skin Lesion Segmentation ",ECCV, 2020, download [link](https://arxiv.org/pdf/1909.00166.pdf).

## Updates
- Augest 1, 2020: First release (Complete implemenation for [SKin Lesion Segmentation on ISIC 218](https://challenge.isic-archive.com/data), [SKin Lesion Segmentation on ISIC 217](https://challenge.isic-archive.com/data) and [PH2](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar)
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
![Diagram of the proposed method](https://github.com/rezazad68/AttentionDeeplabv3p/edit/master/images/aggregation2.png)
### Segmentation visualization
![ISIC 2018](https://github.com/rezazad68/AttentionDeeplabv3p/edit/master/images/result.png)




### Model weights
You can download the learned weights for each dataset in the following table. 

Dataset |Learned weights
------------ | -------------
[ISIC 2018](http://www.isi.uu.nl/Research/Databases/DRIVE/) |[BCDU_net_D3]()
[ISIC 2017](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) |[BCDU_net_D3]()
[Ph2](https://www.kaggle.com/kmader/finding-lungs-in-ct-data/data) | [BCDU_net_D3]()


