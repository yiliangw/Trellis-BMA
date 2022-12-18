import numpy as np
import symbols
from symbols import symbol2digit, digit2symbol
from pathlib import Path
import json
import tqdm

class Status():
    MAT = 0 # match (no error or substitution)
    INS = 1 # insertion
    DEL = 2 # deletion

# Data type for numpy arrays
DTYPE = np.float64

def __add_markers(seq, markers: list):
    res = ''
    lastp = 0
    for p, m in markers:
        res += seq[lastp:p]
        res += m
        lastp = p
    res += seq[lastp:]
    return res


def __remove_markers(seq, markers: list):
    res = ''
    cumulative_marker_len = 0
    lastpos = 0
    for p, m in markers:
        pos = p + cumulative_marker_len
        cumulative_marker_len += len(m)
        res += seq[lastpos: pos]
        lastpos = pos + len(m)
    res += seq[lastpos:]
    return res


'''
Convert markers from {position: str} (interface for user) 
to [(position, str)] (sorted with position as key)
'''
def __sort_markers(markers: list) -> list:
    return sorted(markers, key=(lambda t: t[0]))


def encode_sequence(seq: str, markers: list):
    return __add_markers(seq, __sort_markers(markers))


def encode(sequence_path: str, marker_path: str, encoded_path: str, config_path: str):
    pseq = Path(sequence_path)
    pmarker = Path(marker_path)
    pencoded = Path(encoded_path)
    pconfig = Path(config_path)

    pencoded.parent.mkdir(parents=True, exist_ok=True)
    pconfig.parent.mkdir(parents=True, exist_ok=True)

    markers = []
    with pmarker.open('r') as fmarker:
        for line in fmarker:
            line = line.split()
            markers.append([ int(line[0]), line[1] ])
    markers = __sort_markers(markers)

    length = 0;
    with pseq.open('r') as fseq, pencoded.open('w') as fencoded:
        length = len(fseq.readline().strip())   # All the sequences should have a common length
        fseq.seek(0)
        encoded_seqs = []
        for line in fseq:
            seq = line.strip()
            if len(seq) != length:
                fencoded.truncate(0)
                raise Exception("Detected sequences with different lengths.")
            encoded = __add_markers(seq, markers)
            try:    # In case of memory overflow
                encoded_seqs.append(encoded + '\n')
            except MemoryError:
                fencoded.writelines(encoded_seqs)
                encoded_seqs.clear()
                encoded_seqs.append(encoded + '\n')
        fencoded.writelines(encoded_seqs)

    cfg = {
        'original_length': length,
        'markers': markers
    }
    with pconfig.open('w') as fconfig:
        json.dump(cfg, fconfig, indent=2)
    

def decode_sequence(samples: list, original_length: int, markers: list, ins_p, del_p, sub_p):

    length = original_length + sum(len(m[1]) for m in markers)

    markers = __sort_markers(markers)

    # Construct template's marker flag, where marker bases are specified by the corresponding 
    # symbol, and regular bases are specified by None
    tp_marker_flag = [None for _ in range(length)]
    cumulative_marker_len = 0
    for p, m in markers:
        pos = p + cumulative_marker_len
        cumulative_marker_len += len(m)
        tp_marker_flag[pos: pos+len(m)] = symbol2digit(m)
        
    rev_tp_marker_flag = list(reversed(tp_marker_flag))

    # Convert the samples to list of digits for easy manipulation
    _samples = [symbol2digit(s) for s in samples]
    samples = _samples

    sub_p = sub_p / (1 - del_p - ins_p) # Update sub_p conditioned on Status.MAT
    transition_p = __get_transition_p(ins_p, del_p)
    emission_p = __get_initial_emission_p(tp_marker_flag, sub_p)
    template_p = __get_template_p(sub_p)

    # Calculate the beliefs from each sample
    nsymbol = symbols.num()
    beliefs = np.zeros([len(samples), length, nsymbol], dtype=np.float64)
    rev_beliefs = np.zeros([len(samples), length, nsymbol], dtype=np.float64)
    
    tp_marker_flag = [-1] + tp_marker_flag  # We add a dummy start symbol at the beginning of the template
    rev_tp_marker_flag = [-1] + rev_tp_marker_flag
    for i, s in enumerate(samples):
        # We add a dummy start symbol at the beginning of each sample
        sample = [-1] + s
        rev_sample = [-1] + list(reversed(s))
        __decode_sample(beliefs[i, :], sample, tp_marker_flag, transition_p, emission_p, template_p)
        __decode_sample(rev_beliefs[i, :], rev_sample, rev_tp_marker_flag, transition_p, emission_p, template_p)
        
    # Decode the sequence (with markers) based on BMA (soft vote)
    belief = np.zeros([length, nsymbol], dtype=DTYPE)
    rbeliefs = np.flip(rev_beliefs, axis=1)
    
    half = int(length/2)
    belief[:half] = np.sum(beliefs[:, 0:half, :], axis=0)
    belief[half:] = np.sum(rbeliefs[:, half:, :], axis=0)
    
    belief = belief / np.sum(belief, axis=1)[:, None]
    
    seq_with_markers = np.argmax(belief, axis=1)
    seq_with_markers = digit2symbol(seq_with_markers)

    # Remove the markers
    decoded_seq = __remove_markers(seq_with_markers, markers)

    return seq_with_markers, decoded_seq


def decode(cluster_path: str, config_path: str, decoded_path: str, ins_p: float, del_p: float, sub_p: float,
    cluster_seperator='=', decoded_with_marker_path=None):

    with_marker = decoded_with_marker_path != None

    p_cluster = Path(cluster_path)
    p_config  = Path(config_path)
    p_decoded = Path(decoded_path)
    p_decoded.parent.mkdir(parents=True, exist_ok=True)
    if with_marker:
        p_with_marker = Path(decoded_with_marker_path)
        p_with_marker.parent.mkdir(parents=True, exist_ok=True)

    with p_config.open('r') as f:
        cfg = json.load(f)

    markers = cfg['markers']
    original_length = cfg['original_length']

    # Count the number of the clusters
    with p_cluster.open('r') as f:
        cluster_num = sum(1 for line in f if line.startswith(cluster_seperator))

    decoded_seq = []
    if with_marker:
        decoded_seq_with_marker = []

    f_cluster = p_cluster.open('r')
    f_decoded = p_decoded.open('w')
    if with_marker:
        f_with_marker = p_with_marker.open('w')

    for _ in tqdm.tqdm(range(cluster_num), desc="Decoding"):
        samples = []
        while True:
            line = f_cluster.readline().strip()
            if line.startswith(cluster_seperator):
                break
            samples.append(line)
        with_marker, decoded = decode_sequence(
            samples         = samples,
            original_length = original_length,
            markers         = markers,
            ins_p           = ins_p,
            del_p           = del_p,
            sub_p           = sub_p
        )
        
        try:
            decoded_seq.append(decoded + '\n')
        except MemoryError:
            f_decoded.writeline(decoded_seq)
            decoded_seq.clear()
            decoded_seq.append(decoded + '\n')

        if with_marker:
            try:
                decoded_seq_with_marker.append(with_marker + '\n')
            except MemoryError:
                f_with_marker.writelines(decoded_seq_with_marker)
                decoded_seq_with_marker.clear()
                decoded_seq_with_marker.append(with_marker + '\n')

    f_cluster.close()

    f_decoded.writelines(decoded_seq)
    f_decoded.close()

    if with_marker:
        f_with_marker.writelines(decoded_seq_with_marker)
        f_with_marker.close()
    

'''
Get the emission probability of a sample symbol given the prior of 
the template(sub_p and marker_flags), the status
'''
def __get_initial_emission_p(tp_marker_flag, sub_p):
    nsymbol = symbols.num()

    # Match emission probabilities
    m_emission = {}
    ## Markers'
    for i in range(nsymbol):
        e = np.full(nsymbol, sub_p / (nsymbol - 1), dtype=DTYPE)
        e[i] = 1 - sub_p
        m_emission[i] = e
    ## Regular symbols'
    m_emission[None] = np.full(nsymbol, 1 / nsymbol)
    
    # The insertion emission probability
    i_emission = np.full(nsymbol, 1 / nsymbol, dtype=DTYPE)

    # The deletion emission probability is always 1 (do not emit a specific symbol to sample)
    
    return {Status.MAT: m_emission, Status.INS: i_emission}


def __get_transition_p(ins_p, del_p, full_transition=False):
    mat_p = 1 - ins_p - del_p
    p = np.zeros((3, 3), dtype=DTYPE)
    
    if full_transition:
        p[:, Status.MAT] = mat_p
        p[:, Status.INS] = ins_p
        p[:, Status.DEL] = del_p
    else:
        p[Status.MAT, :] = [mat_p, ins_p, del_p]
        p[Status.INS, :] = [mat_p, ins_p, 0]
        p[Status.DEL, :] = [mat_p, 0, del_p]
        p /= np.sum(p, axis=1)
        
    return p


'''
Get the probability of a template symbol conditioned on the type of 
path (and the corresponding sample symbol is the path is a match)
'''
def __get_template_p(sub_p):
    nsymbol = symbols.num()

    p_mat = np.zeros([nsymbol, nsymbol], dtype=DTYPE)
    for sym in range(nsymbol):
        p_mat[sym, :] = sub_p / (nsymbol - 1)
        p_mat[sym, sym] = 1 - sub_p

    p_del = np.full(nsymbol, 1 / nsymbol, dtype=DTYPE)

    return {Status.MAT: p_mat, Status.DEL: p_del}


'''
Decode a single sample
'''
def __decode_sample(out, sample, tp_marker_flag, transition_p, emission_p, template_p):
    
    tp_flag = tp_marker_flag
    t_len = len(tp_flag) # Template's length (including the dummy start symbol)
    s_len = len(sample) # Sample's length (inluding the dummy start symbol)
    nsymbol = symbols.num()

    assert(out.shape == (t_len-1, nsymbol))
    assert(t_len >= 2)

    # Scaling factors
    c = np.zeros(s_len, dtype=DTYPE)

    MAT, INS, DEL = Status.MAT, Status.INS, Status.DEL
    tran = transition_p
    mat_e = emission_p[MAT]
    ins_e = emission_p[INS]

    # Forward recursion
    def forward():
        # f[MAT, :] are match forward messages
        # f[INS, :] are insertion forward messages
        # f[DEL, :] are deletion forward messages
        f = np.zeros([3, s_len, t_len], dtype=DTYPE)
        # Initialize row 0
        f[[MAT, INS, DEL], 0, 0] = [1, 0, 0]
        for j in range(1, t_len):
            f[[MAT, INS, DEL], 0, j] = [0, 0, np.sum(f[:, 0, j-1] * tran[:, DEL])] # Actually redundant
        c[0] = 1
        # Row 1..M-1
        for i in range(1, s_len-1):
            # MATCH & INSERTION
             ## Column 0
            f[[MAT, INS], i, 0] = [0, np.sum(f[:, i-1, 0] * tran[:, INS]) * ins_e[sample[i]]]
             ## Column 1..N-1
            for j in range(1, t_len-1):
                f[[MAT, INS], i, j] = [
                    np.sum(f[:, i-1, j-1] * tran[:, MAT]) * mat_e[tp_flag[j]] [sample[i]],
                    np.sum(f[:, i-1, j] * tran[:, INS]) * ins_e[sample[i]],
                ]
             ## Column N
            j = t_len-1
            f[[MAT, INS], i, j] = [
                np.sum(f[:, i-1, j-1] * tran[:, MAT]) * mat_e[tp_flag[j]] [sample[i]],
                np.sum(f[:, i-1, j]) * ins_e[sample[i]],
            ]
             ## Calculate the scaling factor
            c[i] = np.sum(f[[MAT, INS], i, :], axis=None)
             ## Scale MATCH & INSERTION
            f[[MAT, INS], i, :] /= c[i]
            
            # DELETION
            f[DEL, i, 0] = 0
            for j in range(1, t_len):
                f[DEL, i, j] = np.sum(f[:, i, j-1] * tran[:, DEL])

       # Row M
        i = s_len - 1
        # MATCH & INSERTION
         ## Column 0
        f[[MAT, INS], i, 0] = [0, np.sum(f[:, i-1, 0] * tran[:, INS]) * ins_e[sample[i]]]
         ## Column 1..N-1
        for j in range(1, t_len-1):
            f[[MAT, INS], i, j] = [
                np.sum(f[:, i-1, j-1] * tran[:, MAT]) * mat_e[tp_flag[j]] [sample[i]],
                np.sum(f[:, i-1, j] * tran[:, INS]) * ins_e[sample[i]],
            ]
         ## Column N
        j = t_len-1
        f[[MAT, INS], i, j] = [
            np.sum(f[:, i-1, j-1] * tran[:, MAT]) * mat_e[tp_flag[j]] [sample[i]],
            np.sum(f[:, i-1, j]) * ins_e[sample[i]],
        ]
        c[i] = np.sum(f[[MAT, INS], i, :], axis=None)
        f[[MAT, INS], i, :] /= c[i]
        
        # DELETION
        f[DEL, i, 0] = 0
        for j in range(1, t_len):
            f[DEL, i, j] = np.sum(f[:, i, j-1])
            
        return f
    # /forward()
    f = forward()

    # Backward recursion
    def backward():
        # b[MAT, :] are match backward messages
        # b[INS, :] are insertion backward messages
        # b[DEL, :] are deletion backward messages
        b = np.zeros([3, s_len, t_len], dtype=DTYPE)
        # Initialize row M
        b[[MAT, INS, DEL], s_len-1, :] = 1
        # Message passing
        for i in range(s_len-2, -1, -1):
            # Column N
            b[[MAT, INS, DEL], i, t_len-1] = b[INS, i+1, t_len-1] * ins_e[sample[i+1]] / c[i+1]
            # Column 0..N-1
            for j in range(t_len-2, -1, -1):
                vec = np.array(
                    [
                        b[MAT, i+1, j+1] * mat_e[tp_flag[j+1]][sample[i+1]],
                        b[INS, i+1, j] * ins_e[sample[i+1]],
                        c[i+1] * b[DEL, i, j+1]
                    ],
                    dtype=DTYPE
                )
                b[[MAT, INS, DEL], i, j] = np.sum(vec[None, :] * tran[[MAT, INS, DEL], :], axis=1) / c[i+1]
                
        return b
    # /backward()
    b = backward()

    # Calculate the beliefs of different paths
    p = f * b

    # Check that the probabilities of MATCH and DELETION paths in a column sum up to 1
    for j in range(1, t_len):
        assert(np.isclose(np.sum(p[[MAT, DEL], :, j]), 1))

    # Calculate the posterior of each template symbol conditioned on the beliefs of all possible 
    # paths and the sample.
    mat_tp = template_p[MAT]
    del_tp = template_p[DEL]

    p_mat = np.zeros(nsymbol, dtype=DTYPE) # Accumulated probability for each types of match paths
    p_del = 0 # Accumulated probability for deletion paths
    for j in range(1, t_len):
        p_mat[:] = 0
        p_del = 0
        # Sum over all match and insertion paths
        for i in range(1, s_len):
            p_mat[sample[i]] += p[MAT, i, j]
        p_del = np.sum(p[DEL, :, j])
        out[j-1, :] = np.sum(p_mat[:, None] * mat_tp, axis=0) + p_del * del_tp

    return

