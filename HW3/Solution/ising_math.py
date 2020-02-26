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
    assert len(chain) == len(J)

    x = []
    for i, char in enumerate(chain):
        val = get_S(char)*get_S(chain[(i+1) % len(chain)])
        val *= J[i]
        x.append(val)
    H = - np.sum(x)

    return H


def get_Z(chains, J):
    '''
    Returns the partition function Z for the full dataset, for the given coupler strengths J, assuming thermodynamic beta=1 (omitted from calculation).
    '''
    x = []
    for chain in chains:
        H = get_H(chain, J)
        x = np.exp(-H)
    Z = np.sum(x)

    return Z


def get_P(chains, J):
    '''
    Returns the normalized Boltzmann probability distribution of the dataset of chains
    '''
    Z = get_Z(chains, J)
    P = []
    for chain in chains:
        x = np.exp(-get_H(chain, J))
        P.append(x/Z)

    return P


def get_E(chains, P):
    '''
    Returns the expectation values for pairs of chains for the given probability distribution P
    '''
    
