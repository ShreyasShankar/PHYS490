#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    '''
    Neural network class inherited from nn.Module.
    Architecture:
        Convolution (10 filters)
        ReLU
        MaxPool
        BatchNorm
        Convolution (20 filters)
        MaxPool
        ReLU
        BatchNorm
        Flatten
        Dropout
        Fully-connected
        ReLU
        Dropout
        Fully-connected
        Softmax
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            # First convolution layer with 10 filters, kernel 3, stride 1, padded
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            # ReLU
            nn.ReLU(inplace=True),
            # MaxPool 12*12*10 to 6*6*10
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Batch Normalization
            nn.BatchNorm2d(num_features=10),
            # Second convolution layer with 20 filters, kernel 5, stride 1, no pad
            nn.Conv2d(10, 20, kernel_size=5),
            # MaxPool 4*4*20 to 2*2*20
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ReLU
            nn.ReLU(inplace=True),
            # Batch Normalization
            nn.BatchNorm2d(num_features=20))

        self.fc = nn.Sequential(
            # Fully-connected linear layer
            nn.Linear(2*2*20, 100),
            # ReLU
            nn.ReLU(inplace=True),
            # Dropout 50%
            nn.Dropout(inplace=True),
            # Final fully-connected linear layer
            nn.Linear(100, 5))
