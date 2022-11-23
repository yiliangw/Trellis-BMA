import numpy as np
import symbols
from symbols import symbol2digit, digit2symbol
import enum

def __add_markers(seq, markers: list(tuple)):
    res = ''
    lastp = 0
    for p, m in markers:
        res += seq[lastp:p]
        res += m
        lastp = p
    res += seq[lastp:]
    return res

def __remove_markers(seq, markers: list(tuple)):
    res = ''
    p = 0
    for pos, m in markers:
        res += seq[p: pos]
        p += len(m)
    res += seq[p:]
    return res

def __convert_markers(markers: dict) -> list(tuple):
    return sorted(list(markers.items()), key=lambda t: t[0])

def encode(seq: str, markers: dict(int, str)):
    return __add_markers(seq, __convert_markers(markers))

def __get_emission_probabilities(nsymbol, sub_p):
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
    d_emissions = {}
    # markers'
    for i in range(nsymbol):
        e = np.zeros(nsymbol)
        e[i] = 1
        d_emissions[i] = e
    # regular symbols'
    e = np.full(nsymbol, 1 / nsymbol, dtype=np.float64)
    d_emissions[None] = e

    return {'match': m_emissions, 'insertion': i_emission, 'deletion': d_emissions}

class Status(enum.Enum):
    MAT = 0 # match (no error or substitution)
    INS = 1 # insertion
    DEL = 2 # deletion

def __decode(out_beliefs, sample, marker_flags, transition, emission):
    
    target_len = len(marker_flags)
    sample_len = len(sample)
    assert(out_beliefs.shape == (target_len, len(symbols.all())))

    sample = [None] + sample
    marker_flags = [None] + marker_flags

    matrix_shape = (target_len+1, sample_len+1)
    dtype = np.float64
    
    # Forward message passing
    f_mat = np.zeros(matrix_shape, dtype=dtype)
    f_del = np.zeros(matrix_shape, dtype=dtype)
    f_ins = np.zeros(matrix_shape, dtype=dtype)

    # Initialization
    f_mat[0, 0] = 1
    f_ins[0, 0] = 0
    f_del[0, 0] = 0

    # Recursion
    for i in range(1, matrix_shape[0]):
        f_mat[i, 0] = 0
        f_ins[i, 0] = 0
        f_del[i, 0] = f_del[i-1, 0] * transition[Status.DEL, Status.DEL] * emission[Status.DEL[marker_flags[i]]]

    for i in range(1, matrix_shape[1]):
        f_mat[0, i] = 0
        f_ins[0, i] = f_ins[0, i-1] * transition[Status.INS, Status.INS] * emission[Status.INS[sample[i]]]
        f_del[0, i] = 0

    for i in range(1, matrix_shape[0]):
        for j in range(1, matrix_shape[1]):
            f_mat[i, j] = (
                f_mat[i-1, j-1] * transition[Status.MAT, Status.MAT] +
                f_ins[i-1, j-1] * transition[Status.INS, Status.MAT] +
                f_del[i-1, j-1] * transition[Status.DEL, Status.MAT]) *
                                emission[Status.INS]


    # Backward messages
    b_mat = np.zeros(matrix_shape, dtype=dtype)
    b_del = np.zeros(matrix_shape, dtype=dtype)
    b_ins = np.zeros(matrix_shape, dtype=dtype)





def decode(samples, original_len, markers, sub_p, del_p, ins_p):
    # Convert the markers from dict to sorted list(tuple)
    markers = __convert_markers(markers)
    length = original_len + len(''.join(markers)) # The length of the target sequence with markers
    # Construct the target sequence, where marker bases are specified by the corresponding 
    # digits defined by symbols.py, and regular bases are specified by None
    marker_flags = [None for _ in range(length)]
    for pos, m in markers:
        marker_flags[pos: pos+len(m)] = symbol2digit(m)

    # Convert the sample sequences to list of digits, which is more convenient for manipulation
    _samples = [symbol2digit(s) for s in samples]
    samples = _samples

    # Calculate the emission probabilities    
    nsymbol = len(symbols.all())
    emissions = __get_emission_probabilities(nsymbol, sub_p)

    # Calculate the beliefs from each sample
    beliefs = np.zeros([len(samples), length, nsymbol], dtype=np.float64)
    for i, sample in enumerate(samples):
        __decode(beliefs[i, :], sample, marker_flags, ins_p, del_p, sub_p)

    # Decode the sequence (with markers) based on BMA (soft vote)
    beliefs = np.sum(beliefs, axis=0)
    seq_with_markers = np.argmax(beliefs, axis=-1)

    # Remove the markers
    decoded_seq = __remove_markers(seq_with_markers, markers)

    return decoded_seq
