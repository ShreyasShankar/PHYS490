#!/usr/bin/env python3

import torch
import torch.nn as nn


class Net(nn.Module):
    '''
    Neural network class inherited from nn.Module.
    Architecture:
        Convolution (5 filters, kernel 5, stride 1, 0 pad)
        BatchNorm
        ReLU
        Convolution (10 filters, kernel 3, stride 1, 0 pad)
        BatchNorm
        MaxPool (kernel 2, stride 2)
        ReLU
        Flatten
        Fully-connected (64 neurons)
        ReLU
        BatchNorm
        Dropout (40%)
        Fully-connected (output, 5 neurons)
        Softmax
    '''

    def __init__(self):
        ''' Build convolutional neural network sequentially '''
        # Inherit
        super(Net, self).__init__()

        # Build convolution layers with activations, batchnorm, pooling
        self.cnn = nn.Sequential(
            # First convolution layer with 5 filters, kernel 3, stride 1, 0 pad
            nn.Conv2d(1, 5, kernel_size=5),  # (14, 14, 1) -> (10, 10, 5)
            # Batch Normalization
            nn.BatchNorm2d(num_features=5),
            # ReLU
            nn.ReLU(),
            # Second convolution layer with 10 filters, kernel 3, stride 1, 0 pad
            nn.Conv2d(5, 10, kernel_size=3),  # (10, 10, 5) -> (8, 8, 10)
            # Batch Normalization
            nn.BatchNorm2d(num_features=10),
            # MaxPool 8*8*10 to 4*4*10
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ReLU
            nn.ReLU(),
            # Flatten
            nn.Flatten())

        # Build fully-connected layers with activations, batchnorm, dropout
        self.fc = nn.Sequential(
            # Fully-connected linear layer
            nn.Linear(4*4*10, 64),
            # ReLU
            nn.ReLU(),
            # Batch Normalization
            nn.BatchNorm1d(64),
            # Dropout 40%
            nn.Dropout(0.4, inplace=True),
            # Final fully-connected linear layer
            nn.Linear(64, 5),
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
