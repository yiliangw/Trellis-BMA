import symbols
import simulation
import marker_code
import evaluation
from pathlib import Path
from tqdm import tqdm
import numpy as np


def main():

    SIMULATION = True               # Refactoring: Only simulation is guranteed to work for the moment
    
    # Metadata for the CNR dataset
    global DATASET_SIZE, DATASET_SEQUENCE_LEN
    DATASET_SIZE = 10000
    DATASET_SEQUENCE_LEN = 110    
    
    ############################
    ###### Configurations ######
    ############################
    global SUB_P, DEL_P, INS_P
    SUB_P = 0.03
    DEL_P = 0.03
    INS_P = 0.03
    MARKER_NUM = 4                  # Numver of marker in each sequence (uniformly distributed)
    MARKER_LEN = 2                  # The length of each marker
    SIM_CLUSTER_NUM     = 20        # The number of clusters to process
    SIM_RANDOM_SEED     = 6219      # Random seed for IDS channel
    SIM_SEQUENCE_LEN    = 102       # Length for each sequence
    SIM_CLUSTER_SIZE    = 6         # Size of each cluster (coverage)
    ############################
    ############################
    
    if SIMULATION:
        run_with_simulation(
            marker_num  = MARKER_NUM,
            marker_len  = MARKER_LEN,
            cluster_num = SIM_CLUSTER_NUM,
            random_seed = SIM_RANDOM_SEED,
            seq_len     = SIM_SEQUENCE_LEN,
            nsample     = SIM_CLUSTER_SIZE
        )
    else:
        run_with_dataset(
            marker_num  = MARKER_NUM, 
            marker_len  = MARKER_LEN,
            cluster_num = DATASET_SIZE,
        )

    return


def run_with_simulation(marker_num, marker_len, cluster_num, random_seed, seq_len, nsample):

    ins_p = INS_P
    del_p = DEL_P
    sub_p = SUB_P

    symbols.init()
    data_path = Path(__file__).resolve().parent / 'data'
    encode_path = data_path / 'encode'
    
    sequence_path_str   = str(encode_path / 'input/sequences.txt')
    marker_path_str     = str(encode_path / 'input/markers.txt')
    encoded_path_str    = str(encode_path / 'output/encoded_sequences.txt')
    config_path_str     = str(encode_path / 'output/marker_config.json')

    decode_path = data_path / 'decode'
    cluster_path_str    = str(decode_path / 'input/clusters.txt')
    decoded_path_str    = str(decode_path / 'output/decoded.txt')
    decoded_with_mark_path_str \
                        = str(decode_path / 'output/decoded_with_marker.txt')

    evaluation_path_str = str(data_path / 'evaluation')

    SYMBOLS = ['A', 'C', 'G', 'T']
    cluster_seperator = ('=' * 20)

    # Generate simulation data
    simulation.generate_simulation_data(
        seq_num        = cluster_num,
        seq_len        = seq_len,
        marker_num     = marker_num,
        marker_len     = marker_len,
        sequence_path  = sequence_path_str,
        marker_path    = marker_path_str,
        seed           = random_seed
    )

    # Encode
    encoder = marker_code.Encoder()
    encoder.encode(
        sequence_path   = sequence_path_str,
        marker_path     = marker_path_str,
        encoded_path    = encoded_path_str,
        config_path     = config_path_str
    )

    # Simulate a IDS channel
    simulation.simulate_IDS_channel(
        encoded_path    = encoded_path_str,
        output_path     = cluster_path_str,
        ins_p           = ins_p,
        del_p           = del_p,
        sub_p           = sub_p,
        sample_num      = nsample,
        seed            = random_seed,
        seperator       = cluster_seperator
    )

    # Decode
    decoder = marker_code.Decoder(ins_p=ins_p, del_p=del_p, sub_p=sub_p, symbols=SYMBOLS, np_dtype=np.float64)
    decoder.decode(
        cluster_path             = cluster_path_str,
        config_path              = config_path_str,
        decoded_path             = decoded_path_str,
        decoded_with_marker_path = decoded_with_mark_path_str,
        cluster_seperator        = cluster_seperator,
    )

    # Evaluate with markers
    evaluation.report(
        truth_path      = encoded_path_str,
        result_path     = decoded_with_mark_path_str,
        output_dir      = evaluation_path_str,
    )

    print("\nSimualtion done.")

    return


def run_with_dataset(marker_num, marker_len, cluster_num):

    print("CNR dataset not supported to run with at the time")
    return
    
    seq_len = DATASET_SEQUENCE_LEN
    marker_info = [ (int(seq_len/(marker_num+1)*i)-int(marker_len/2), marker_len) for i in range(1, marker_num+1) ]
    
    def seperate_markers(sequence):
        seq = ''
        markers = {}
        cumulative_marker_len = 0
        lastpos = 0
        for pos, length in marker_info:
            seq += sequence[lastpos: pos]
            markers[pos-cumulative_marker_len] = sequence[pos: pos+length]
            cumulative_marker_len += length
            lastpos = pos + length
        seq += sequence[lastpos: ]
        return seq, markers 

    def process_cluster(center, samples):
        assert(len(center) == seq_len)
        gold, markers = seperate_markers(center)
        decoded_with_marker, _ = marker_code.decode(samples[:5], len(gold), markers, SUB_P, DEL_P, INS_P)
        return decoded_with_marker
    
    
    fname_results = 'results.txt'
    f_results = open(OUTPUT_PATH + '/' + fname_results, 'w+')
    f_centers = open(INPUT_PATH + '/Centers.txt', 'r')
    f_clusters = open(INPUT_PATH + '/Clusters.txt', 'r')
    
    f_clusters.readline()   # Skip the first line

    def get_one_cluster():
        center = f_centers.readline().strip()
        if len(center) == 0:
            return None, None
        samples = []
        while True:
            s = f_clusters.readline().strip()
            if len(s) == 0 or s.startswith('='):
                break
            samples.append(s)
        return center, samples
    # /get_one_cluster()

    
    for i in tqdm(range(cluster_num), desc="Decoding"):
        center, samples = get_one_cluster()
        assert(len(center) == seq_len)
        if center == None:
            cluster_num = i + 1
            break
        decoded = process_cluster(center, samples)
        f_results.write(decoded + '\n')
        
    f_centers.seek(0)
    f_results.seek(0)
    template_seqs = []
    decoded_seqs = []
    for i in range(cluster_num):
        template_seqs.append(f_centers.readline().strip())
        decoded_seqs.append(f_results.readline().strip())
    output_statistics(template_seqs, decoded_seqs)

    f_centers.close()
    f_clusters.close()
    f_results.close()
    
    print(fname_results + ' saved to ' + OUTPUT_PATH)

    return
    

'''
Add placeholder to sequence without markers to align it to sequence with markers
'''
def add_marker_placeholder(sequence, markers):
    markers = sorted([(p, m) for p, m in markers.items()], key=(lambda t: t[0]))
    out = ''
    lp = 0
    for p, m in markers:
        out += sequence[lp:p]
        out += '-' * len(m)
        lp = p
    out += sequence[lp:]
    return out


if __name__ == "__main__":
    main()
    