# ResidualAttentionNetwork-paddle

## 1.Introduction
This project is based on the paddlepaddle_V2.1 framework to reproduce ResidualAttentionNetwork and the [official code](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/) of pytoch.

## 2.Accuracy

The model is trained on the train set of Cifar10, and tested on the test set of Cifar10.
The top1 error of Attention-92 model given in the paper is 4.6%, and the top1 error obtained in this project is %. 

## 3.Quick Start

### Step1:clone

### Step2:training

make sure the varible  *is_train = True*
```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
### Step3:evaluating

make sure the varible  *is_train = False*
```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  

## Align
 * Network structure transfer
 * Weight transfer(paddle version link): 
 * Verify the network code
 * forward align : [RAN/step1-forward/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step1-forward)
 * loss function align : [RAN/step2-loss/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step2-loss)
 * backward align : [RAN/step3-backward/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step3-backward)



## Paper referenced
[Residual Attention Network for Image Classification (CVPR-2017 Spotlight)](https://arxiv.org/pdf/1704.06904v1.pdf)
By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang
