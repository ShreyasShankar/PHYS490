import numpy as np
import json
import argparse
import os


def get_args():
    '''
    Parse arguments from CLI
    '''
    parser = argparse.ArgumentParser(
        description='Compute regression coefficients analytically and with stochastic gradient descent')
    parser.add_argument('in_file', type=str,
                        help='path to .in file containing dataset')
    parser.add_argument('json_file', type=str,
                        help='path to .json file containing gradient descent hyperparameters')

    return parser.parse_args()


def loadData(args):
    '''
    Load dataset from arguments given
    '''
    data = np.loadtxt(args.in_file)

    # Pad data array with ones in the first column
    ones = np.ones((data.shape[0], 1), dtype=data.dtype)
    data = np.hstack((ones, data))

    # Separate data into targets and features
    targets = data[:, -1]
    features = data[:, 0:-1]

    return features, targets


def analyticSolution(features, targets):
    '''
    Compute the analytic solution to multiple linear regression
    '''
    gram_mat = np.dot(features.T, features)  # Compute the Gram matrix
    moment_mat = np.dot(features.T, targets)  # Compute the moment matrix
    w_analytic = np.around(np.dot(np.linalg.inv(
        gram_mat), moment_mat), 4)

    return w_analytic


def sgdSolution(json_file, features, targets):
    '''
    Compute the weights of the linear regression problem using stochastic gradient descent
    '''
    # Load json file and extract hyperparameters
    with open(json_file, 'r') as f:
        params = json.load(f)
    alpha = params['learning rate']
    epochs = params['num iter']

    # Initialize new weight vector to random values
    w_sgd = np.random.uniform(size=(features.shape[1],))
    while epochs:
        # Stochastic selection of feature-target index
        idx = np.random.randint(0, len(targets))
        x = features[idx, :]
        y = targets[idx]

        # Compute predicted target, gradient, and update weights
        predicted_y = np.dot(w_sgd.T, x)
        gradient = (predicted_y - y) * x
        w_sgd -= alpha * gradient

        epochs -= 1

    return np.around(w_sgd, 4)


def makeOutFile(args, w_analytic, w_sgd):
    '''
    Create a .out file in the same directory as the .in file, storing the analytic coefficients followed by the sgd coefficients
    '''
    out_file = os.path.splitext(args.in_file)[0] + '.out'
    with open(out_file, 'w+') as f:
        for i in list(w_analytic):
            f.write('{:.4f}\n'.format(i))
        for i in list(w_sgd):
            f.write('\n{:.4f}'.format(i))


def run():
    args = get_args()
    features, targets = loadData(args)
    w_analytic = analyticSolution(features, targets)
    w_sgd = sgdSolution(args.json_file, features, targets)
    makeOutFile(args, w_analytic, w_sgd)


if __name__ == '__main__':
    run()
