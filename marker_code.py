import numpy as np
from symbols import symbol2digit, digit2symbol, all_symbols

def marker_encode(seq, markers, positions):
    res = ''
    lastp = 0
    for m, p in zip(markers, positions):
        res += seq[lastp:p]
        res += m
        lastp = p
    res += seq[lastp:]
    return res

def __forward_propogation():


def marker_decode(sample, original_len, markers, positions, ins_p, del_p, sub_p):
    # Convert the sample sequence to digits, which is more convenient
    sample = symbol2digit(sample)
    
    # Construct the target sequence, where markers are represented by the corresponding 
    # digits defined map.py, and regulare bases are specified by None
    length = original_len + len(''.join(markers))
    marker = [None for _ in len(length)]
    for m, p in zip(markers, positions):
        marker[p: p+len(m)] = symbol2digit(m)

    nsymbol = len(all_symbols())
    # Get the match(align) emission probabilities
    m_emissions = {}
    # markers'
    for i in range(nsymbol):
        e = np.zeros([nsymbol, nsymbol], dtype=np.float64)
        e[i, :] = sub_p / (nsymbol - 1)     # Assume that the different substitutions have
                                            # equal probabilities
        e[i, i] = 1 - sub_p
        m_emissions[i] = e
    # regular symbols'
    e = np.zeros([nsymbol, nsymbol], dtype=np.float64)
    e = sub_p / (nsymbol - 1) / nsymbol
    for i in range(nsymbol):
        e[i, i] = (1 - sub_p) / nsymbol
    m_emissions[None] = e

    # Get the insertion emission probability
    i_emission = np.full(nsymbol, 1 / nsymbol, dtype=np.float64)

    # Get the deletion emission probabilities
    d_emission = {}
    # markers'
    for i in range(nsymbol):
        e = np.zeros(nsymbol)
        e[i] = 1
        d_emission[i] = e
    # regular symbols'
    e = np.full(nsymbol, 1 / nsymbol, dtype=np.float64)

    





    





    def __forward_propogation():
        return None

    return ''

