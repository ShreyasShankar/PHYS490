#!/usr/bin/env python3

import argparse
import torch


def get_args():
    ''' Parse arguments from CLI '''
    parser = argparse.ArgumentParser(
        description='Multi-label CNN classifier for the set of 14x14 even numbers from MNIST')
    parser.add_argument('-p', '--params', type=str, metavar='./files/param.json',
                        default='./files/param.json',
                        help='Path to .json file containing hyperparameters')
    parser.add_argument('-d', '--data', type=str, metavar='../Problem/even_mnist.csv',
                        default='../Problem/even_mnist.csv',
                        help='Path to .csv file containing the 14x14 dataset')
    parser.add_argument('-r', '--results', default=False,
                        metavar='./results',
                        help='Directory where loss plot should be saved, if desired. Results will be saved to file only if path is provided')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Boolean flag for high verbosity output')
    parser.add_argument('-c', '--cuda', type=bool, default=False,
                        help='Boolean flag indicating whether or not to use CUDA GPU')

    return parser.parse_args()


def run():
    get_args()


if __name__ == '__main__':
    run()
