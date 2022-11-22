import numpy as np

MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
UNMAP = {v: k for k, v in MAP.items()}

def __map(letters):
    return [MAP[letter] for letter in letters]

def __unmap(digits):
    return [UNMAP[digit] for digit in digits]

def bma(samples):
    mat = np.zeros([len(samples), len(samples[0])])
    for i, sample in enumerate(samples):
        mat[i, :] = __map(sample)
    majorities = [np.argmax(np.bincount(mat[:, i])) for i in mat.shape[1]]
    return __unmap(majorities)
    
