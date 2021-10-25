# ResidualAttentionNetwork-Paddle

## 1.Introduction
This project is based on the paddlepaddle_V2.1 framework to reproduce ResidualAttentionNetwork and the [official code](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/) of pytorch.

## 2.Result

The model is trained on the train set of CIFAR10, and tested on the test set of CIFAR10. The dataset can be downloaded at [here](http://www.cs.toronto.edu/~kriz/cifar.html).



 Version | Accuracy | Top1 error 
 ---- | ----- | ----- 
 paper | 95.01% | 4.99%
 pytorch version(official) | 95.4% |  4.6%
 paddle version(ours)| 95.69%  | 4.31%



<img src="https://github.com/tbymiracle/Paddle-RAN/blob/master/325621635038522_.pic_hd.jpg" width="600"/>

The given model file of the pytorch version (accuracy of 95.4%)

Link: https://pan.baidu.com/s/1GfHo25O-WUWz2X8uzjWA2A  Password: 8ukd

The given model file of the paddle version (accuracy of 95.4%)

Link: https://pan.baidu.com/s/1SRz0QP2gYSeimUwF-dLh5A  Password: nmcd

The model file we trained (accuracy of 95.69%)

Link: https://pan.baidu.com/s/1hxNrwfauaKZo53e-CQ_j8w  Password: 9rd3


Put these two model files in Paddle-RAN/RAN/.

## 3.Requirements

 * Hardware：GPU（Tesla V100-32G is recommended）
 * Framework:  PaddlePaddle >= 2.1.2


## 4.Quick Start

### Step1: Clone

``` 
git clone https://github.com/tbymiracle/Paddle-RAN.git
cd Paddle-RAN/RAN
``` 

### Step2: Training

```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
### Step3: Evaluating

```  
CUDA_VISIBLE_DEVICES=0 python test.py
```  

## 5.Align

We use the [`repord_log`](https://github.com/WenmuZhou/reprod_log) tool to align.

```  
python check_log_diff.py # check diff of all the steps (step1 - step4).
```  

 * Network structure transfer.
 * Weight transfer(paddle version link): https://pan.baidu.com/s/1l4uZJxOJjWjCDbK_-2bnNg  Password: 4hle
 * Verify the network.
 * Forward align : [RAN/step1-forward/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step1-forward)
 * Loss function align : [RAN/step2-loss/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step2-loss)
 * Backward align : [RAN/step3-backward/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step3-backward)
 * train align : [RAN/step4-acc/](https://github.com/tbymiracle/Paddle-RAN/tree/master/RAN/step4-acc)



## Paper referenced
[Residual Attention Network for Image Classification (CVPR-2017 Spotlight)](https://arxiv.org/pdf/1704.06904v1.pdf)
By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang.
