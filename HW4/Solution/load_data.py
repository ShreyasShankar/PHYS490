#!/usr/bin/env python3

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Data():
    '''
    Data loader class
    '''

    def __init__(self, data_file, test_size):
        '''
        Load MNIST data from provided csv file and separate into training and
        testing datasets, given the test set's size (training set is always the
        complement of the test set)
        '''

        raw_data = pd.read_csv(data_file, sep=' ', header=None)
        raw_data = np.array(raw_data)
        y = raw_data[:, -1].astype(np.long)
        x = raw_data[:, :-1].astype(np.float)
        y = y/2  # Scale y to range 0-4
        x = x/255  # Scale x values 0-1 range

        # Split x and y into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, shuffle=True)

        # Reshape x_train and x_test into 4D input shape expected by convolution layer
        train0, dim1 = x_train.shape
        dim1 = int(np.sqrt(dim1))
        test0, _ = x_test.shape
        x_train = np.reshape(x_train, (train0, 1, dim1, dim1))
        x_test = np.reshape(x_test, (test0, 1, dim1, dim1))

        # Cast all data to correct torch tensor dtypes
        self.x_train = torch.tensor(x_train, dtype=torch.float)
        self.x_test = torch.tensor(x_test, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
