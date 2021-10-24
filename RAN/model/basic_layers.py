# import torch
# import torch.nn as nn
# from torch.nn import init
# import functools
# from torch.autograd import Variable
import paddle.nn as nn
import paddle
class ResidualBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # print(input_channels)
        # print(output_channels)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2D(input_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2D(in_channels=input_channels, out_channels=int(output_channels/4), kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2D(int(output_channels/4))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(int(output_channels/4), int(output_channels/4), kernel_size=3, stride=stride, padding = 1, bias_attr = False)
        self.bn3 = nn.BatchNorm2D(int(output_channels/4))
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2D(int(output_channels/4), output_channels, kernel_size=1, stride=1, bias_attr = False)
        self.conv4 = nn.Conv2D(input_channels, output_channels , 1, stride, bias_attr = False)
        
    def forward(self, x):
        # print(x)
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        # print(out)
        return out

import numpy as np


def gen_fake_data():
    fake_data = np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.5
    fake_label = np.arange(1).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)


# if __name__ == "__main__":
#     # gen_fake_data()
#     fake_data = np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.5
#     x = paddle.to_tensor(fake_data)
#     model = ResidualBlock(32,128)
#     output = model(x)