# ResidualAttentionNetwork-paddle

## 1.Introduction
This project is based on the paddlepaddle_V2.1 framework to reproduce ResidualAttentionNetwork and the [official code](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/) of pytoch.

## 2.Result

The model is trained on the train set of Cifar10, and tested on the test set of Cifar10.


 model  | top1 error  
 ---- | ----- 
 paper  | 4.99%
 pytorch version(official)  |  4.6%
 paddle version(ours)  | 4.31%

## 3.Requirements

 * Hardware：GPU（Tesla V100-32G is recommended）
 * Framework:  PaddlePaddle >= 2.1.2


## 3.Quick Start

### Step1: Clone

``` 
git clone https://github.com/tbymiracle/Paddle-RAN.git
cd Paddle-RAN/RAN
``` 

### Step2: Training

Make sure the varible  `is_train = True`
```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
### Step3: Evaluating

Make sure the varible  `is_train = False`
```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  

## Align

We use the [`repord_log`](https://github.com/WenmuZhou/reprod_log) tool to align.
 * Network structure transfer
 * Weight transfer(paddle version link): 
 * Verify the network code
 * forward align : [RAN/step1-forward/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step1-forward)
 * loss function align : [RAN/step2-loss/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step2-loss)
 * backward align : [RAN/step3-backward/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step3-backward)



## Paper referenced
[Residual Attention Network for Image Classification (CVPR-2017 Spotlight)](https://arxiv.org/pdf/1704.06904v1.pdf)
By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang.
