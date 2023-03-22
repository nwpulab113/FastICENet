## FastICENet
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

This project aims at providing a Real-time and Accurate Semantic Segmentation Model for Aerial Remote Sensing River Ice Image


---

### Table of Contents:
- <a href='#Requirements'>Requirements</a>
- <a href='#Models'>Models</a>
- <a href='#Dataset-Setting'>Dataset Setting</a>
- <a href='#Usage'>Usage</a>
- <a href='#Contact'>Contact</a>

### Requirements

 [**PyTorch**](https://pytorch.org/) and [**Torchvision**](https://pytorch.org/) needs to be installed before running the scripts,  PyTorch v1.1 or later is supported. 

```bash
pip3 install -r requirements.txt
```



#### Losses

 The project supports these loss functions: 

> 1. Weighted Cross Entropy
> 2. Weighted Cross Entropy with Label Smooth
> 3. Focal Loss
> 4. Ohem Cross Entropy
> 5. [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax)
> 6. [SegLoss-List](https://github.com/JunMa11/SegLoss)
> 7. ...

#### Optimizers

 The project supports these optimizers: 

> 1. SGD
> 2. Adam 
> 3. AdamW 
> 4. [RAdam](https://github.com/LiyuanLucasLiu/RAdam)
> 5. RAdam + Lookahead
> 6. ...

#### Activations

> 1. ReLu
> 2. PReLU
> 3. ReLU6
> 4. Swish
> 5. [Mish](https://github.com/digantamisra98/Mish) : A Self Regularized Non-Monotonic Neural Activation Function
> 6. ...

#### Learning Rate Scheduler

The project supports these LR_Schedulers: 

> 1. Poly decay
> 2. Warmup Poly  
> 3. ...

#### Normalization methods

> 1. [In-Place Activated BatchNorm](https://github.com/mapillary/inplace_abn)
> 2. [Switchable Normalization](https://github.com/switchablenorms/Switchable-Normalization)
> 3. [Weight Standardization](https://github.com/joe-siyuan-qiao/WeightStandardization)
> 4. ...

#### Enhancing Semantic Feature Learning Method

> 1. [Attention Family](https://github.com/implus/PytorchInsight)
> 2. [NAS Family](https://github.com/D-X-Y/NAS-Projects)
> 3. ...

#### Some useful Tools

> 1. [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)
> 2. [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 
> 3. [Netron](https://github.com/lutzroeder/Netron) : Visualizer for neural network models, On line URL: [Netron](https://lutzroeder.github.io/netron/)
> 4. [Falshtorch](https://github.com/MisaOgura/flashtorch): Visualization toolkit for neural networks in PyTorch !
> 5. [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks)
> 6. ...



### Usage



##### Training
- For river

1. training on **train** set

```
python train.py  --help
```

2. training on **train+val** set

```
python train.py --help
```

##### Testing
- For river

```
python test.py --help
```

##### Predicting
- For river

```
python predict.py --help
```

##### Evaluating
- For river

```
cd tools
python trainID2labelID.py 
```



