# WCD for Remote Sensing Object Recognition

This repository includes introductions and implementation of ***Weakly Correlated Distillation for Remote Sensing Object Recognition*** in PyTorch.


# Datasets

We conduct experiments using three remote sensing datasets: **[NWPU VHR-10 (Cheng, Zhou, and Han 2016)](https://gcheng-nwpu.github.io/#Datasets), [DOTA (Xia et al. 2018)](https://captain-whu.github.io/DOTA/)** and **[HRRSD (Zhang et al. 2019)](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset)**.

Remote sensing objects are cut out from object detection ground truth.

Ten common object categories among three datasets are reserved for experiments, i.e., ***baseball-diamond, basketball-court, bridge, ground-track-field, harbor, airplane, ship, vehicle, storage-tank and tennis-court*** .

Specifically, folder index and categories are as follows:

>01 baseball-diamond  
02 basketball-court  
03 bridge  
04 ground-track-field  
05 harbor  
06 airplane  
07 ship  
08 vehicle  
09 storage-tank  
10 tennis-court  

You can download post-processed datasets from this link:  **[datasets](https://pan.baidu.com/s/1ykhDbLpFuBGn2QCEIS67Eg?pwd=ju93)** 
 

## File Structure
File structure is divided into two categories: teacher models and student models. The directories with the "S" prefix are student models, and with the "T" prefix are teacher models. Teacher models of three network structures (VGG, ResNet, DenseNet) are all used, but for student models, you can choose only one of them to train and test.

```
WCD 
.
├── README.md
├── S-DenseNet
│   ├── class_indices.json
│   ├── Densenet
│   │   └── model.py
│   ├── model.py
│   ├── my_utils
│   │   └── spearman.py
│   ├── Resnet
│   │   └── model.py
│   ├── train.py
│   └── VGG
│       └── model.py
├── S-ResNet
│   ├── class_indices.json
│   ├── Densenet
│   │   └── model.py
│   ├── model.py
│   ├── my_utils
│   │   └── spearman.py
│   ├── Resnet
│   │   └── model.py
│   ├── train.py
│   └── VGG
│       └── model.py
├── S-VGG
│   ├── class_indices.json
│   ├── Densenet
│   │   └── model.py
│   ├── model.py
│   ├── my_utils
│   │   └── spearman.py
│   ├── Resnet
│   │   └── model.py
│   ├── train.py
│   └── VGG
│       └── model.py
├── T-DenseNet
│   ├── class_indices.json
│   ├── model.py
│   ├── my_dataset.py
│   ├── train.py
│   └── utils.py
├── T-ResNet
│   ├── class_indices.json
│   ├── model.py
│   └── train.py
└── T-VGG
    ├── class_indices.json
    ├── model.py
    └── train.py

```

# Models

All model parameters can be obtained from this link: [**WCD-pth**](https://pan.baidu.com/s/17PHjoQ1c2AUmVzwr_G2hnw?pwd=ii0c).

# Requirements

>PyTorch >= 1.3.1  
>TorchVision >= 0.4.2  
>
>>Recommended  
>>tqdm >= 4.61  
>>matplotlib >= 1.5.1


# Train and Eval

## Train
For detailed argparse params, go to the S-DenseNet directory and run
> python train.py --func=train

If you don't intend to customize params and paths, run  
> python train.py

## Eval
For detailed argparse params, go to the S-DenseNet directory and run
> python train.py --func=test
