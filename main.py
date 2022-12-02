import symbols
import IDS_channel
import marker_code
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm


def main():

    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    
    # Metadata for the CNR dataset
    global DATASET_SIZE, DATASET_SEQUENCE_LEN
    DATASET_SIZE = 10000
    DATASET_SEQUENCE_LEN = 110    
    
    ############################
    ###### Configurations ######
    ############################
    # Shared configurations
    global OUTPUT_PATH
    OUTPUT_PATH = ROOT_PATH + '/output'
    global SUB_P, DEL_P, INS_P
    SUB_P = 0.01
    DEL_P = 0.01
    INS_P = 0.01
    CLUSTER_NUM = DATASET_SIZE      # The number of clusters to process
    MARKER_NUM = 4                  # Numver of marker in each sequence (uniformly distributed)
    MARKER_LEN = 2                  # The length of each marker
    SIMULATION = False              # True: run with CNR dataset; False: run with simulated IDS channel
    # Dataset configurations
    global INPUT_PATH
    INPUT_PATH = ROOT_PATH + '/dataset'
    # Simulation configurations
    SIM_RANDOM_SEED     = 6219      # Random seed for IDS channel
    SIM_SEQUENCE_LEN    = 110       # Length for each sequence
    SIM_CLUSTER_SIZE    = 6         # Size of each cluster (coverage)
    ############################
    ############################
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    symbols.init(['A', 'G', 'C', 'T'])

    if SIMULATION:
        run_with_simulation(
            nmarker     = MARKER_NUM,
            marker_len  = MARKER_LEN,
            ncluster    = CLUSTER_NUM,
            random_seed = SIM_RANDOM_SEED,
            seq_len     = SIM_SEQUENCE_LEN,
            nsample     = SIM_CLUSTER_SIZE
        )
    else:
        run_with_dataset(
            nmarker=MARKER_NUM, 
            marker_len=MARKER_LEN,
            ncluster=CLUSTER_NUM
        )

    return


def run_with_dataset(nmarker, marker_len, ncluster):
    
    seq_len = DATASET_SEQUENCE_LEN
    marker_info = [ (int(seq_len/(nmarker+1)*i)-int(marker_len/2), marker_len) for i in range(1, nmarker+1) ]
    
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

    
    for i in tqdm(range(ncluster), desc="Decoding"):
        center, samples = get_one_cluster()
        assert(len(center) == seq_len)
        if center == None:
            ncluster = i + 1
            break
        decoded = process_cluster(center, samples)
        f_results.write(decoded + '\n')
        
    f_centers.seek(0)
    f_results.seek(0)
    template_seqs = []
    decoded_seqs = []
    for i in range(ncluster):
        template_seqs.append(f_centers.readline().strip())
        decoded_seqs.append(f_results.readline().strip())
    output_statistics(template_seqs, decoded_seqs)

    f_centers.close()
    f_clusters.close()
    f_results.close()
    
    print(fname_results + ' saved to ' + OUTPUT_PATH)

    return


def run_with_simulation(nmarker, marker_len, ncluster, random_seed, seq_len, nsample):

    fname_results = 'results.txt'
    f_results = open(OUTPUT_PATH + '/' + fname_results, 'w')
    
    random.seed(random_seed)

    gold_seqs = []
    decoded_seqs = []
    for i in tqdm(range(ncluster), desc="Decoding"):
        gold = ''.join(random.choices(symbols.all(), k=116))
        markers = { int(seq_len/(nmarker+1)*i): 'A'*marker_len for i in range(1, nmarker+1)}

        # Add markers to the template
        encoded = marker_code.encode(gold, markers)
        # Pass the encoded template through IDS channel to get noisy samples 
        samples = IDS_channel.generate_noisy_samples(encoded, nsample, SUB_P, DEL_P, INS_P)
        # Decode the template
        decoded_with_marker, decoded = marker_code.decode(samples, len(gold), markers, SUB_P, DEL_P, INS_P)

        gold_seqs.append(gold)
        decoded_seqs.append(decoded)

        cluster_str = 'Cluster-{}\n'.format(i)
        cluster_str += \
            '  template: {}\n'.format(add_marker_placeholder(gold, markers)) + \
            'm-template: {}\n'.format(encoded) + \
            ' m-decoded: {}\n'.format(decoded_with_marker) + \
            '   decoded: {}\n'.format(add_marker_placeholder(decoded, markers)) + '\n'

        f_results.write(cluster_str)

    output_statistics(gold_seqs, decoded_seqs)

    f_results.close()
    print(fname_results + " saved to " + OUTPUT_PATH)

    return


def output_statistics(template_seqs, decoded_seqs):
    np.set_printoptions(threshold=sys.maxsize)  # To print ndarray in full length

    assert(len(template_seqs) == len(decoded_seqs))
    ncluster = len(template_seqs)
    length = len(template_seqs[0])
    
    fname_statistics = 'statistics.txt'

    f = open(OUTPUT_PATH + '/' + fname_statistics, 'w')

    accuracies = np.zeros(ncluster, np.float64)
    all_error = np.zeros(length, dtype=np.int32)
    seq_error = np.zeros(length, dtype=np.int8)
    for i in range(ncluster):
        template, decoded = template_seqs[i], decoded_seqs[i]
        seq_error[:] = 0
        for pos in range(length):
            seq_error[pos] = int(template[pos] != decoded[pos])
        all_error += seq_error
        accuracies[i] = 1 - (np.float64(np.sum(seq_error)) / length)

    hr = '=' * 20 + '\n'
    acc_str = hr + "ACCURACY\n" + hr
    acc_str += " max: {:.1%}\n".format(np.amax(accuracies))
    percents = np.array([75, 50, 25])
    for p, pt in zip(percents, np.percentile(accuracies, percents)):
        acc_str += " {:02d}%: {:.1%}\n".format(p, pt)
    acc_str += " min: {:.1%}\n".format(np.amin(accuracies))
    acc_str += "mean: {:.1%}\n".format(np.mean(accuracies))

    print(acc_str)

    acc_str += '\nclusters\' accuracies:\n'
    acc_str += np.array2string(accuracies, separator=', ') + '\n'

    idx_accuracies = 1 - all_error / np.float64(ncluster)
    acc_str += '\npositional accuracies:\n'
    acc_str += np.array2string(idx_accuracies, separator=', ') + '\n'
    
    f.write(acc_str + '\n')

    plt.plot(list(range(length)), idx_accuracies)
    plt.xlim([0, length])
    plt.ylim([max(np.amin(idx_accuracies) * 0.85, 0), np.amax(idx_accuracies)*1.15])
    plt.title('Positional Accuracy')
    plt.xlabel('Base Index')
    plt.ylabel('Accuracy')
    fname_index_acc = 'positional_accuracy.png'
    plt.savefig(OUTPUT_PATH + '/' + fname_index_acc, format='png')
    plt.clf()
    print(fname_index_acc + ' saved to ' + OUTPUT_PATH)
        
    sum = np.sum(all_error)
    err_distribution = np.full(length, 1 / length) if sum == 0 else all_error / sum
    distr_str = hr + 'ERROR DISTRIBUTION\n' + hr + np.array2string(err_distribution, precision=3, separator=', ')
    f.write(distr_str + '\n')

    plt.bar(list(range(length)), err_distribution * 100)
    plt.xlim([0, length])
    plt.ylim([0, min(np.amax(err_distribution)*100, 100)*1.15])
    plt.title('Positional Error Distribution')
    plt.xlabel('Base Index')
    plt.ylabel('Probability Mass (%)')
    fname_err_distr = 'positional_error_distribution.png'
    plt.savefig(OUTPUT_PATH + '/' + fname_err_distr, format='png')
    plt.clf()
    print(fname_err_distr + ' saved to ' + OUTPUT_PATH)

    f.close()
    print(fname_statistics + ' saved to ' + OUTPUT_PATH)

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
    