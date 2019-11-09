import numpy as np


def calc_dist(data, prox_countries):
    return np.tensordot(data, prox_countries) / np.sum(data, axis=(0,1))

