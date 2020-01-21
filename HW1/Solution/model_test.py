import numpy as np


def verifyModel(x, y, weights, output=False):
    '''
    Verify model using sklearn
    '''
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(x, y)

    w = np.around(np.array([reg.intercept_] + list(reg.coef_)), 4)

    if output:
        print(reg.score(x, y))
        print(w)
    print(all(w == weights))


# Load data from .in file
filename = '../Problem/data/2.in'
data = np.loadtxt(filename)

# Pad data array with ones in the first row
ones = np.ones((data.shape[0], 1), dtype=data.dtype)
data = np.hstack((ones, data))

# Separate data into targets and features
targets = data[:, -1]
features = data[:, 0:-1]

# Analytic model
gram_mat = np.matmul(features.T, features)
moment_mat = np.matmul(features.T, targets)
weights = np.around(np.matmul(np.linalg.inv(
    gram_mat), moment_mat, dtype=float), 4)

verifyModel(features[:, 1:], targets, weights, True)

# Stochastic gradient descent
