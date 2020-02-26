#!/usr/bin/env python3

from ising_math import *

import argparse
import torch
import os
import json
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt


def get_args():
    ''' Parse arguments from CLI '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, metavar='./files/param.json',
                        default='./files/param.json',
                        help='Path to .json file containing hyperparameters')
    parser.add_argument('-d', '--data', type=str, metavar='../Problem/in.txt',
                        default='../Problem/in.txt',
                        help='Path to .txt file containing the dataset')
    parser.add_argument('-r', '--results', default=False,
                        metavar='./results',
                        help='Directory where loss plot should be saved, if desired. Results will be saved to file only if path is provided')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Boolean flag for high verbosity output')

    return parser.parse_args()


def run():
    args = get_args()

    with open(args.params) as p:
        params = json.load(p)

    data = np.loadtxt(args.data, dtype=str)
    J = np.random.choice([-1, 1], size=4)  # Random initialization of J
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    train_loss = []

    chain_dict = Counter(data)
    chain_P = np.array(list(chain_dict.values())) / 1000
    unique_chains = np.array(list(chain_dict.keys()))

    epochs = int(params['num_epochs'])
    print('Training over {} epochs:'.format(epochs))
    with tqdm(total=epochs, dynamic_ncols=True) as pbar:
        for epoch in range(1, epochs+1):
            # Training
            Z = get_Z(unique_chains, J)
            predicted_P = torch.from_numpy(np.log(get_P(unique_chains, J)))
            true_P = torch.from_numpy(chain_P)

            # loss
            loss = loss_fn(predicted_P, true_P)
            train_loss.append(loss)

            # Grad Update
            gradient = grad(unique_chains, J, chain_P)
            J += lr*np.array(gradient)

            # Update progress bar
            pbar.update()

            # High verbosity report
            if args.verbose:
                if not ((epoch) % params['display_epochs']):
                    print('\nEpoch [{}/{}]'.format(epoch, epochs)
                          + '\tLoss: {:.4f}'.format(loss))

    # Final loss report
    print('Final training loss: {:.4f}'.format(train_loss[-1]))

    rounded_J = np.round(J, 0)

    # Plot loss and save results if desired
    # plt.plot(range(1, epochs+1), obj_vals, label='Training loss', color='blue')
    # plt.plot(range(1, epochs+1), cross_vals, label='Test loss', color='green')
    # plt.legend()
    # if args.results:
    #     if not os.path.exists(args.results):
    #         os.mkdir(args.results)
    #     plt.savefig(os.path.join(args.results, 'loss_report.png'), dpi=600)
    # plt.show()


if __name__ == '__main__':
    run()
