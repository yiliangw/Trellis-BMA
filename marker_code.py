import numpy as np
import symbols
from symbols import symbol2digit, digit2symbol
import enum

class Status(enum.Enum):
    MAT = 0 # match (no error or substitution)
    INS = 1 # insertion
    DEL = 2 # deletion


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

DTYPE=np.float64

def __get_emission_probabilities(marker_flags, nsymbol, sub_p):
    # Match emission probabilities
    symbol_emissions = {}
    ## Markers'
    for i in range(nsymbol):
        e = np.full(nsymbol, (1 - sub_p) / (nsymbol - 1), dtype=DTYPE)
        e[i] = 1 - sub_p
        symbol_emissions[i] = e
    ## Regular symbols'
    symbol_emissions[None] = np.full(nsymbol, 1 / nsymbol)
    ## Match emission conditioned on each template symbols
    m_emission = np.zeros((len(marker_flags), nsymbol), dtype=DTYPE)
    for i in range(1, len(marker_flags)):
        m_emission[i][:] = symbol_emissions[marker_flags[i]]
    
    # Insertion emission probability
    i_emission = np.full(nsymbol, 1 / nsymbol, dtype=np.float64)

    # Deletion emission probabilities == 1
    
    return {Status.MAT: m_emission, Status.INS: i_emission}


def __decode(out_beliefs, sample, template_flag, transition, emission):
    t_len = len(template_flag) # Template's length
    s_len = len(sample) # Sample's length

    mat_e = emission[Status.MAT]
    ins_e = emission[Status.INS]

    MAT, INS, DEL = Status.MAT, Status.INS, Status.DEL

    # Forward recursion
    f = np.zeros([3, t_len, s_len], dtype=DTYPE)
    # f[MAT, :] are match forward messages
    # f[INS, :] are insertion forward messages
    # f[DEL, :] are deletion forward messages

    ## Initialization
    f[MAT, 0, 0] = 1
    f[INS, 0, 0] = 0
    f[DEL, 0, 0] = 0

    ## Message passing
    for i in range(1, t_len):
        f[MAT, i, 0] = 0
        f[INS, i, 0] = 0
        f[DEL, i, 0] = f[DEL, i-1, 0] * transition[DEL, DEL]

    for i in range(1, s_len):
        f[MAT, 0, i] = 0
        f[INS,0, i] = f[INS, 0, i-1] * transition[INS, INS] * ins_e[sample[i]]
        f[DEL, 0, i] = 0

    for i in range(1, t_len):
        for j in range(1, s_len):
            # f_mat[i, j] = (
            #     f_mat[i-1, j-1] * transition[Status.MAT, Status.MAT] +
            #     f_ins[i-1, j-1] * transition[Status.INS, Status.MAT] +
            #     f_del[i-1, j-1] * transition[Status.DEL, Status.MAT]
            # ) * mat_e[i] [sample[j]]
            # f_ins[i, j] = (
            #     f_mat[i, j-1] * transition[Status.MAT, Status.INS] +
            #     f_ins[i, j-1] * transition[Status.INS, Status.INS] +
            #     f_del[i, j-1] * transition[Status.DEL, Status.INS]
            # ) * ins_e[sample[j]]
            # f_del[i, j] = (
            #     f_mat[i-1, j] * transition[Status.MAT, Status.DEL] +
            #     f_ins[i-1, j] * transition[Status.INS, Status.DEL] +
            #     f_del[i-1, j] * transition[Status.DEL, Status.DEL]
            # )
            f[MAT, i, j] = np.sum(f[:, i-1, j-1] * transition[:, MAT]) * mat_e[i][sample[j]]
            f[INS, i, j] = np.sum(f[:, i, j-1] * transition[:, INS]) * ins_e[sample[j]]
            f[DEL, i, j] = np.sum(f[:, i-1, j] * transition[:, DEL])

    # Backward recursion
    b = np.zeros([3, t_len, s_len], dtype=DTYPE)
    # b[MAT, :] are match backward messages
    # b[INS, :] are insertion backward messages
    # b[DEL, :] are deletion backward messages

    ## Initialization
    b[MAT, t_len-1, s_len-1] = 1
    b[INS, t_len-1, s_len-1] = 1
    b[DEL, t_len-1, s_len-1] = 1

    ## Message passing
    for i in range(1, t_len, -1):
        b_mat[i, sample_len] = b_del[i+1, sample_len] * emission[Status.DEL][marker_flags[i+1]] * transition[Status.MAT, Status.DEL]
        b_ins[i, sample_len] = 0
        b_del[i, sample_len] = b_del[i+1, sample_len] * emission[Status.DEL][marker_flags[i+1]] * transition[Status.DEL, Status.DEL]

    for i in range(1, sample_len-1, -1):
        b_mat[target_len, i] = b_ins[target_len, i+1] * emission[Status.INS][sample[i+1]] * transition[Status.MAT, Status.INS]
        b_ins[target_len, i] = b_ins[target_len, i+1] * emission[Status.INS][sample[i+1]] * transition[Status.INS, Status.INS]
        b_del = 0

    for i in range(1, target_len-1, -1):
        for j in range(1, sample_len-1, -1):
            mat = b_mat[i+1, j+1] * emission[Status.MAT][marker_flags[i+1]] [sample[j+1]]
            ins = b_ins[i, j+1] * emission[Status.INS][sample[j+1]]
            dele = b_del[i+1, j] * emission[Status.DEL][marker_flags[i+1]]
            vec = np.array([mat, ins, dele], dtype=dtype)

            b_mat[i, j] = transition[Status.MAT] * vec
            b_ins[i, j] = transition[Status.INS] * vec
            b_del[i, j] = transition[Status.DEL] * vec
            
    p_full = f_mat[target_len, sample_len] + f_ins[target_len, sample_len] + f_del[target_len, sample_len]

    p_mat = f_mat[1:, 1:] * b_mat[1:, 1:] / p_full
    p_ins = f_ins[1:, 1:] * b_ins[1:, 1:] / p_full
    p_del = f_del[1:, 1:] * b_del[1:, 1:] / p_full

    out_beliefs = np.sum(p_mat, axis=1) + np.sum(p_del, axis=1)



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
