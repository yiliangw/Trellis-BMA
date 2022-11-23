import numpy as np
from symbols import symbol2digit, digit2symbol


def bma(samples):
    mat = np.zeros([len(samples), len(samples[0])])
    for i, sample in enumerate(samples):
        mat[i, :] = seq2digits(sample)
    majorities = [np.argmax(np.bincount(mat[:, i])) for i in mat.shape[1]]
    return digits2seq(majorities)
    
