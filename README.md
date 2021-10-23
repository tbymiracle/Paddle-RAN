# ResidualAttentionNetwork-paddle
A paddle code about Residual Attention Network.  

This code is based on the pytorch code from 

https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/

# paper referenced
Residual Attention Network for Image Classification (CVPR-2017 Spotlight)
By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang


# how to train?
first, download the data from http://www.cs.toronto.edu/~kriz/cifar.html
make sure the varible 
# 
is_train = True
```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
# how to test?
make sure the varible 
#
is_train = False
```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  

# result
1. cifar-10: Acc-95.4(Top-1 err 4.6) with ResidualAttentionModel_92_32input_update(higher than paper top-1 err 4.99)

# model file： 
model_92_sgd.pkl is the trained model file, accuracy of 0.954
