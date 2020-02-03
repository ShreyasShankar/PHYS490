#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt


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
        y = raw_data[:, -1].astype(np.int)
        x = raw_data[:, :-1]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, shuffle=True)

        # fig = plt.figure()
        # for i in range(6):
        #     plt.subplot(2, 3, i+1)
        #     plt.tight_layout()
        #     plt.imshow(self.x_train[i].reshape((14, 14)),
        #                cmap='gray')
        #     plt.title('Ground Truth {}'.format(self.y_train[i]))
        #     plt.xticks([])
        #     plt.yticks([])
        # plt.show()


# hj = Data('../Problem/even_mnist.csv', 3000)
