#!/usr/bin/env python3

import torch
import torch.nn as nn


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
        ''' Initialize network sequentially '''
        # Inherit
        super(Net, self).__init__()

        # Build convolution layers with activations, pooling, and dropout
        self.cnn = nn.Sequential(
            # First convolution layer with 10 filters, kernel 5, stride 1, padded
            nn.Conv2d(1, 10, kernel_size=5, padding=1),
            # ReLU
            nn.ReLU(inplace=False),
            # MaxPool 12*12*10 to 6*6*10
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Batch Normalization
            nn.BatchNorm2d(num_features=10),
            # Second convolution layer with 20 filters, kernel 3, stride 1, no pad
            nn.Conv2d(10, 20, kernel_size=3),
            # MaxPool 4*4*20 to 2*2*20
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ReLU
            nn.ReLU(inplace=False),
            # Batch Normalization
            nn.BatchNorm2d(num_features=20),
            # Flatten
            nn.Flatten(),
            # Dropout 15%
            nn.Dropout(0.15, inplace=False))

        # Build fully-connected layers with activations and dropout
        self.fc = nn.Sequential(
            # Fully-connected linear layer
            nn.Linear(2*2*20, 128),
            # ReLU
            nn.ReLU(inplace=False),
            # Dropout 50%
            nn.Dropout(inplace=False),
            # Final fully-connected linear layer
            nn.Linear(128, 5),
            # Softmax
            nn.Softmax(dim=1))

    def forward(self, x):
        ''' Compute feedforward '''
        output = self.cnn(x)
        output = self.fc(output)
        return output

    def backprop(self, data, loss, optimizer, device):
        ''' Train network with backpropagation '''
        # Set model to training mode
        self.train()
        # Get inputs and send them to the appropriate device (cpu or cuda)
        inputs = data.x_train
        targets = data.y_train
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Compute model output from given inputs
        outputs = self(inputs)
        # Compute loss
        obj_val = loss(self.forward(inputs), targets)
        # Initialize gradients to 0
        optimizer.zero_grad()
        # Propagate loss
        obj_val.backward()
        # Update optimizer
        optimizer.step()
        # Return loss for tracking and visualizing
        return obj_val.item()

    def test(self, data, loss, device):
        ''' Test network on cross-validation data '''
        # Set model to evaluation mode
        self.eval()
        # Deactivate autograd engine
        with torch.no_grad():
            # Get inputs and send them to the appropriate device (cpu or cuda)
            inputs = data.x_test
            targets = data.y_test
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Compute model output from given inputs
            outputs = self(inputs)
            # Compute loss
            cross_val = loss(self.forward(inputs), targets)
        # Return loss for tracking and visualizing
        return cross_val.item()
