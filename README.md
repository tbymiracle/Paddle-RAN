# ResidualAttentionNetwork-paddle

## 1.Introduction
A paddle code about Residual Attention Network.  

This project is based on the paddlepaddle_V2.1 framework to reproduce ResidualAttentionNetwork.

and the official code from 

https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/

## 2.Accuracy

The model is trained on the train set of Cifar10, and tested on the test set of Cifar10.
The top1 error of Attention-92 model given in the paper is 4.6%, and the top1 error obtained in this project is %. 

## 3.Quick Start

### Step1:clone

### Step2:training

make sure the varible  _is_train = True _
```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
### Step3:evaluating

make sure the varible   _is_train = False _
```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  

## 
Align（对齐）
 * 网络结构代码转换
 * 权重转换 转换完成的模型链接: https://pan.baidu.com/s/1I19luoHYwiSAxm-vLqlGMQ  密码: 1cs3
 * 模型组网正确性验证
 * 前向对齐 ./step1-forward/
 * 损失函数对齐 ./step2-loss/
 * 反向对齐 ./step3-backward/



## Paper referenced
Residual Attention Network for Image Classification (CVPR-2017 Spotlight) (https://arxiv.org/pdf/1704.06904v1.pdf)
By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang


model_92_sgd.pkl is the trained model file, accuracy of 0.954
