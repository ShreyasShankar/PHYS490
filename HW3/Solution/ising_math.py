#!/usr/bin/env python3

''' Mathematical functions necessary for the physics of the problem '''


import numpy as np


def get_S(char):
    ''' Map '+' = +1, '-' = -1 '''
    S = 1 if char is '+' else -1
    return int(S)


def get_H(chain, J):
    '''
    Returns the hamiltonian H of a single chain, provided a list of coupler strengths J.
    Note: len(chain) must match len(J)
    '''
    x = []
    for i in range(len(chain)):
        val = get_S(chain[i])*get_S(chain[(i+1) % 4]) * J[i]
        x.append(val)
    H = - np.sum(x)
    return int(H)


def get_Z(chains, J):
    '''
    Returns the partition function Z for the full dataset, for the given coupler strengths J, assuming thermodynamic beta=1 (omitted from calculation).
    '''
    x = []
    for chain in chains:
        H = get_H(chain, J)
        x.append(np.exp(- H))
    Z = np.sum(x)
    return Z


def get_P(chains, J):
    '''
    Returns the normalized Boltzmann probability distribution of the dataset of chains
    '''
    Z = get_Z(chains, J)
    P = []
    for chain in chains:
        x = np.exp(- get_H(chain, J))
        P.append(x/Z)
    return P


def get_E(chains, P):
    '''
    Returns the expectation values for neighbouring pairs of chains for the given probability distribution P
    '''
    E = []
    for S1 in range(4):  # Because chain length is 4
        S2 = (S1 + 1) % 4
        x = []
        for i in range(2**4):
            val = get_S(chains[i][S1])*get_S(chains[i][S2]) * P[i]
            x.append(val)
        E.append(np.sum(x))
    return E


def grad(chains, J, true_P):
    ''' Compute gradient for training purposes '''
    pred_P = get_P(chains, J)
    true_E = np.array(get_E(chains, true_P))
    pred_E = np.array(get_E(chains, pred_P))
    grad = np.subtract(true_E, pred_E)
    return grad
