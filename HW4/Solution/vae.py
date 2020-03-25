#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint


class VAE(nn.Module):
    ''' Neural network class inherited from nn.Module '''
    def __init__(self):
        ''' Build VAE sequentially '''
        # Inherit
        super(VAE, self).__init__()

        # Standalone layers
        lin_size = 300
        self.fc1 = nn.Linear(4*4*10, lin_size)
        self.fc21 = nn.Linear(lin_size, 20)
        self.fc22 = nn.Linear(lin_size, 20)
        # self.fc3 = nn.Linear(20, 100)
        # self.fc4 = nn.Linear(100, 14*14)

        # Base encoder network
        self.encoder = nn.Sequential(
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
        )

        # Build decoder network
        self.decoder = nn.Sequential(
            # Fully-connected linear layer
            nn.Linear(20, 100),
            # nn.Linear(4*4*10, 64),
            # ReLU
            nn.ReLU(),
            # # Batch Normalization
            # nn.BatchNorm1d(64),
            # Final fully-connected linear layer
            nn.Linear(100, 14*14),
            # nn.Linear(64, 5),
            # # Softmax
            # nn.Softmax(dim=1)
            # Sigmoid
            nn.Sigmoid()
        )


    def encode(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def sampling(self, mu, logvar):
        ''' Sampling through reparameterization '''
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        ''' Compute feedforward '''
        mu, logvar = self.encode(x)
        z = sampling(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, reconstruction, mu, logvar):
        ''' Reconstruction + KL divergence losses summed over all elements and batch '''
        BCE = F.binary_cross_entropy(reconstruction, x.view(-1, 196), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD)/x.size(0)

    def init_data(self, data, device):
        '''
        Extract training & testing data (without labels) and send them to appropriate device (cpu or gpu)
        '''
        self.train = data.x_train
        self.test - data.x_test
        # targets = data.y_train  #################
        self.train = self.train.to(device)
        self.test = self.test.to(device)
        # targets = targets.to(device)  #################

    def batcher(self, inputs, batch_size):
        ''' Create batch from input data and batch_size '''
        batch_start = randint(0, len(inputs) - batch_size)
        return inputs[batch_start: batch_start+batch_size]

    def backprop(self, batch_size, optimizer):
        ''' Train network with backpropagation '''
        # Set model to training mode
        self.train()
        # Initialize gradients to 0
        optimizer.zero_grad()
        # Batching
        inputs = self.batcher(self.train, batch_size)
        # targets = self.targets[batch_start: batch_start+batch_size]  #################
        # Compute model output from inputs
        reconstruction, mu, logvar = self(inputs)
        # Compute loss
        obj_val = self.loss(inputs, reconstruction, mu, logvar)
        # Propagate loss
        obj_val.backward()
        # Update optimizer
        optimizer.step()
        # Return loss for tracking and visualizing
        return obj_val.item()

    def test(self, test_size):
        ''' Test network on cross-validation data '''
        # Set model to evaluation mode
        self.eval()
        # Deactivate autograd engine
        with torch.no_grad():
            # # Get inputs and send them to the appropriate device (cpu or cuda)
            # inputs = data.x_test
            # targets = data.y_test
            # inputs = inputs.to(device)
            # targets = targets.to(device)

            # Batching
            inputs = self.batcher(self.test, test_size)
            # Compute model output from given inputs
            reconstruction, mu, logvar = self(inputs)
            # Compute loss
            cross_val = self.loss(inputs, reconstruction, mu, logvar)
        # Return loss for tracking and visualizing
        return cross_val.item()
