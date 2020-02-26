#!/usr/bin/env python3

from ising_math import *

import argparse
import torch
import os
import json
import torch.nn as nn
import numpy as np
from tqdm import tqdm
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
    loss = nn.KLDivLoss(reduction='batchmean')
    train_loss = []


if __name__ == '__main__':
    run()
