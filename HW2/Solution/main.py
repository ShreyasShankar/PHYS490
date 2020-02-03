#!/usr/bin/env python3

import torch
import json
import argparse
import os
from tqdm import tqdm
import torch.optim as optim
from nn import Net
from load_data import Data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


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
                        help='Directory where results should be saved, if desired')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Boolean flag for high verbosity output')
    parser.add_argument('-c', '--cuda', type=bool, default=False,
                        help='Boolean flag indicating whether or not to use GPU')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open(args.params) as p:
        params = json.load(p)

    # CUDA usage
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Initialize network and get data from .csv file
    model = Net().to(device)
    data = Data(data_file=args.data, test_size=int(params['test_size']))

    # Define SGD optimizer and multi-class cross entropy loss function
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    loss = torch.nn.CrossEntropyLoss()

    # Training and evaluation cycle
    # Initialize lists
    obj_vals = []
    cross_vals = []
    epochs = int(params['num_epochs'])
    print('Training over {} epochs:'.format(epochs))
    with tqdm(total=epochs, dynamic_ncols=True) as pbar:
        for epoch in range(1, epochs+1):
            # Training
            train_val = model.backprop(data, loss, optimizer, device)
            obj_vals.append(train_val)

            # Cross-validation evaluation
            test_val = model.test(data, loss, device)
            cross_vals.append(test_val)

            # Update progress bar
            pbar.update()

            # High verbosity report
            if args.verbose:
                if not ((epoch) % params['display_epochs']):
                    print('\nEpoch [{}/{}]'.format(epoch, epochs)
                          + '\tTraining Loss: {:.4f}'.format(train_val)
                          + '\tTest Loss: {:.4f}'.format(test_val))

    # Final loss report
    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    # Plot loss and save results if desired
    plt.plot(range(1, epochs+1), obj_vals, label='Training loss', color='blue')
    plt.plot(range(1, epochs+1), cross_vals, label='Test loss', color='green')
    plt.legend()
    if args.results:
        if not os.path.exists(args.results):
            os.mkdir(args.results)
        plt.savefig(os.path.join(args.results, 'loss_report.png'), dpi=600)
    plt.show()
