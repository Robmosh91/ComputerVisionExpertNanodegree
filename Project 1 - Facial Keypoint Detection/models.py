## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, input_size=(1, 96, 96)):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # 1 input image channel (grayscale)
        # 10 output channels/feature maps
        # 3x3 square convolution kernel
        # output size = (W-F+2*padding)/S + 1
        self.filters_conv1 = 32
        self.conv1 = nn.Conv2d(in_channels=input_size[0], 
                               out_channels=self.filters_conv1, 
                               kernel_size=2, 
                               stride=2, 
                               padding=0)
        self.out_shape_conv1 = (self.filters_conv1, 
                                int((input_size[1]-2+2*0)/2+1), 
                                int((input_size[2]-2+2*0)/2+1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.out_shape_pool1 = (self.filters_conv1, 
                                int((self.out_shape_conv1[1]-2+2*0)/1+1), 
                                int((self.out_shape_conv1[2]-2+2*0)/1+1))
        self.conv1_bn = nn.BatchNorm2d(self.filters_conv1)
        # self.dropout1 = nn.Dropout(p=0.05)
        # Second Conv block
        self.filters_conv2 = 64
        self.conv2 = nn.Conv2d(in_channels=self.filters_conv1, out_channels=self.filters_conv2,
                              kernel_size=3, stride=1, padding=0)
        self.out_shape_conv2 = (self.filters_conv2, 
                                int((self.out_shape_pool1[1]-3+2*0)/1+1), 
                                int((self.out_shape_pool1[2]-3+2*0)/1+1))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.out_shape_pool2 = (self.filters_conv2, 
                                int((self.out_shape_conv2[1]-3+2*0)/2+1),
                                int((self.out_shape_conv2[2]-3+2*0)/2+1))
        self.conv2_bn = nn.BatchNorm2d(self.filters_conv2)
        # self.dropout2 = nn.Dropout(p=0.1)
        # Third Conv block
        self.filters_conv3 = 128
        self.conv3 = nn.Conv2d(in_channels=self.filters_conv2, out_channels=self.filters_conv3,
                              kernel_size=3, stride=1, padding=0)
        self.out_shape_conv3 = (self.filters_conv3, 
                                int((self.out_shape_pool2[1]-3+2*0)/1+1), 
                                int((self.out_shape_pool2[2]-3+2*0)/1+1))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_shape_pool3 = (self.filters_conv3, 
                                int((self.out_shape_conv3[1]-2+2*0)/2+1), 
                                int((self.out_shape_conv3[2]-2+2*0)/2+1))
        self.conv3_bn = nn.BatchNorm2d(self.filters_conv3)
        # self.dropout3 = nn.Dropout(p=0.3)
        # Fourth Conv block
        self.filters_conv4 = 256
        self.conv4 = nn.Conv2d(in_channels=self.filters_conv3, out_channels=self.filters_conv4,
                              kernel_size=3, stride=2, padding=0)
        self.out_shape_conv4 = (self.filters_conv4, 
                                int((self.out_shape_pool3[1]-3+2*0)/2+1), 
                                int((self.out_shape_pool3[2]-3+2*0)/2+1))
        # self.pool4_kernel = self.out_shape_conv4[1]
        self.pool4_kernel = 3
        self.pool4_stride = 2
        self.pool4 = nn.MaxPool2d(kernel_size=self.pool4_kernel, stride=self.pool4_stride)
        # self.pool4 = nn.AvgPool2d(kernel_size=self.pool4_kernel, stride=self.pool4_stride)
        self.out_shape_pool4 = (self.filters_conv4, 
                                int((self.out_shape_conv4[1]-self.pool4_kernel+2*0)/self.pool4_stride+1), 
                                int((self.out_shape_conv4[2]-self.pool4_kernel+2*0)/self.pool4_stride+1))
        self.conv4_bn = nn.BatchNorm2d(self.filters_conv4)
        # self.dropout4 = nn.Dropout(p=0.5)
        # Flatten output shape
        self.flat_shape = int(self.out_shape_pool4[0]*self.out_shape_pool4[1]*self.out_shape_pool4[2])
        # Linear block
        self.linear1_out = 1024
        self.linear1 = nn.Linear(in_features=self.flat_shape, out_features=self.linear1_out)
        self.linear1_dropout = nn.Dropout(p=0.5)
        self.linear2_out = 512
        self.linear2 = nn.Linear(in_features=self.linear1_out, out_features=self.linear2_out)
        self.linear2_dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(in_features=self.linear2_out, out_features=136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        # Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        # Block 3
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.conv3_bn(x)
        # Block 4
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.conv4_bn(x)
        # Linear Block
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear1_dropout(x)
        x = F.relu(self.linear2(x))
        x = self.linear2_dropout(x)
        x = self.output(x)
        x = x.view(x.size(0), 68, 2)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
