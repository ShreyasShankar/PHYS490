#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt, cm as cm
import os.path as op
from random import randint
import math


class VAE(nn.Module):
    ''' Neural network class inherited from nn.Module '''

    def __init__(self):
        ''' Build VAE sequentially '''
        # Inherit
        super(VAE, self).__init__()

        # Standalone layers
        lin_size = 200
        self.fc1 = nn.Linear(7*7*10, lin_size)
        self.fc21 = nn.Linear(lin_size, 10)
        self.fc22 = nn.Linear(lin_size, 10)

        # Base encoder network
        self.encoder = nn.Sequential(
            # First convolution layer with 5 filters, kernel 3, stride 1, 1 padding
            # (14, 14, 1) -> (14, 14, 5)
            nn.Conv2d(1, 5, kernel_size=3, padding=1, stride=1),
            # Batch Normalization
            nn.BatchNorm2d(num_features=5),
            # ReLU
            nn.ReLU(),
            # Second convolution layer with 10 filters, kernel 3, stride 1, 1 padding
            # (14, 14, 5) -> (14, 14, 10)
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1),
            # Batch Normalization
            nn.BatchNorm2d(num_features=10),
            # MaxPool 14*14*10 to 7*7*10
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ReLU
            nn.ReLU(),
            # Flatten
            nn.Flatten()
        )

        # Decoder network
        self.decoder = nn.Sequential(
            # Fully-connected linear layer
            nn.Linear(10, lin_size),
            # ReLU
            nn.ReLU(),
            # Batch Normalization
            nn.BatchNorm1d(lin_size),
            # Fully-connected linear layer
            nn.Linear(lin_size, lin_size*2),
            # ReLU
            nn.ReLU(),
            # Final fully-connected linear layer
            nn.Linear(lin_size*2, 14*14),
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
        z = self.sampling(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, reconstruction, mu, logvar):
        ''' Reconstruction + KL divergence losses summed over all elements and batch '''
        BCE = F.binary_cross_entropy(
            reconstruction, x.view(-1, 14*14), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD)/x.size(0)

    def init_data(self, data, device):
        '''
        Extract training & testing data (without labels) and send them to appropriate device (cpu or gpu)
        '''
        self.train_set = data.x_train
        self.test_set = data.x_test
        self.train_set = self.train_set.to(device)
        self.test_set = self.test_set.to(device)

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
        inputs = self.batcher(self.train_set, batch_size)
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
            # Batching
            inputs = self.batcher(self.test_set, test_size)
            # Compute model output from given inputs
            reconstruction, mu, logvar = self(inputs)
            # Compute loss
            cross_val = self.loss(inputs, reconstruction, mu, logvar)
        # Return loss for tracking and visualizing
        return cross_val.item()

    def is_square(self, n):
        ''' Check if n is a perfect square using Newton's method '''
        x = n // 2
        y = set([x])
        while x * x != n:
            x = (x + (n // x)) // 2
            if x in y:
                return False
            y.add(x)
        return True

    def subplotter(self, n):
        ''' Based on input number's properties, determine subplot grid shape '''
        if self.is_square(n):
            return int(math.sqrt(n)), int(math.sqrt(n))
        else:
            if n >= 12:
                return 4, math.ceil(n/4)
            else:
                return 2, math.ceil(n/2)

    def final_reconstruction(self, n_images, path, device):
        ''' Reconstruct n_images images '''
        nrow, ncol = self.subplotter(n_images)  # Get subplot grid shape
        with torch.no_grad():
            for i in range(0, n_images):
                # Generate n z-tensors from a uniform distribution
                z_test = torch.randn(1, 10)
                z_test = z_test.to(device)
                # Reconstruct
                reconstruction = self.decode(z_test)
                image = reconstruction.view(-1, 14, 14).cpu().numpy()
                plt.subplot(nrow, ncol, i+1)
                plt.imshow(image[0, :, :], cmap=cm.gray)
                plt.axis('off')
            if path:
                plt.savefig(
                    op.join(path, '{}_nums.png'.format(n_images)), dpi=800)
            plt.show()
