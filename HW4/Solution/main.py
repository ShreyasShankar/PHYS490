#!/usr/bin/env python3

from load_data import Data
from vae import VAE

import torch
import json
import argparse
import os
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt


def get_args():
    ''' Parse arguments from CLI '''
    parser = argparse.ArgumentParser(
        description='Multi-label Variational Auto-encoder for the set of 14x14 even numbers from MNIST')
    parser.add_argument('-p', '--params', type=str, default='./files/param.json',
                        help='Path to .json file containing hyperparameters')
    parser.add_argument('-d', '--data', type=str, default='../Problem/even_mnist.csv',
                        help='Path to .csv file containing the 14x14 dataset')
    parser.add_argument('-r', '--results', default='./results',
                        help='Directory where loss plot & output images should be saved')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Boolean flag for high verbosity output')
    parser.add_argument('-c', '--cuda', type=bool, default=False,
                        help='Boolean flag indicating whether or not to use CUDA GPU')
    parser.add_argument('-n', '--num_images', type=int, default=100,
                        help='Number of "handwritten" images to generate upon completion')

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.params) as p:
        params = json.load(p)
    test_size = int(params['test_size'])

    # CUDA usage
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Initialize network and get data from .csv file
    model = VAE().to(device)
    data = Data(data_file=args.data, test_size=test_size)
    model.init_data(data, device)

    # Define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    ## Training and evaluation cycle
    # Initialize lists
    obj_vals = []
    cross_vals = []
    epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])

    print('Training over {} epochs:'.format(epochs))
    with tqdm(total=epochs, dynamic_ncols=True) as pbar:
        for epoch in range(1, epochs+1):
            # Training
            train_val = model.backprop(batch_size, optimizer)
            obj_vals.append(train_val)

            # Cross-validation evaluation
            test_val = model.test(min(test_size, batch_size))
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

    # Plot loss and save results
    plt.plot(range(1, epochs+1), obj_vals, label='Training loss', color='blue')
    plt.plot(range(1, epochs+1), cross_vals, label='Test loss', color='green')
    plt.title('BCE + KLD Loss over %i Training Epochs' % epochs)
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    if args.results:
        if not os.path.exists(args.results):
            os.mkdir(args.results)
        plt.savefig(os.path.join(args.results, 'loss_report.png'), dpi=600)
    plt.show()

    model.final_reconstruction(args.num_images, args.results, device)


if __name__ == '__main__':
    main()
