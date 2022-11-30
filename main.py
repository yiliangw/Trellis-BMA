import symbols
import IDS_channel
import marker_code
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def main():

    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    CLUSTER_NUM_MAX = 10000     # There are 10000 clusters in the CNR dataset
    ###### Configurations ######
    global INPUT_PATH, OUTPUT_PATH
    INPUT_PATH = ROOT_PATH + '/dataset'
    OUTPUT_PATH = ROOT_PATH + '/output'
    global SUB_P, DEL_P, INS_P
    SUB_P = 0.01
    DEL_P = 0.01
    INS_P = 0.01
    CLUSTER_NUM = 5
    SIMULATION = False
    # Simulation related configurations
    RANDOM_SEED = 6219        # The random seed for IDS channel
    SIM_CLUSTER_SIZE = 6      # The size of each cluster (coverage)
    ############################
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    symbols.init(['A', 'G', 'C', 'T'])

    if SIMULATION:
        run_with_simulation(random_seed=RANDOM_SEED, ncluster=CLUSTER_NUM, nsample=SIM_CLUSTER_SIZE)
    else:
        run_with_dataset(ncluster=CLUSTER_NUM)

    return


def run_with_dataset(ncluster=5):

    def seperate_markers(sequence, marker_info: list):
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
        marker_info = [(int(len(center)*1/4), 2), (int(len(center)*2/4), 2), (int(len(center)*3/4), 2)]
        gold, markers = seperate_markers(center, marker_info)
        decoded_with_marker, decoded = marker_code.decode(samples[:5], len(gold), markers, SUB_P, DEL_P, INS_P)
        return decoded_with_marker
    
    fname_results = 'results.txt'
    f_results = open(OUTPUT_PATH + '/' + fname_results, 'w+')
    f_centers = open(INPUT_PATH + '/Centers.txt', 'r')
    f_clusters = open(INPUT_PATH + '/Clusters.txt', 'r')
    
    f_clusters.readline().strip()   # Skip the first line

    seq_len = len(f_centers.readline().strip())
    f_centers.seek(0)

    def get_one_cluster():
        center = f_centers.readline().strip()
        if len(center) == 0:
            return None, None
        samples = []
        while True:
            s = f_clusters.readline().strip()
            if s.startswith('='):
                break
            samples.append(s)
        return center, samples
    # /get_one_cluster()

    template_seqs = []
    decoded_seqs = []
    for i in tqdm(range(ncluster), desc="Decoding"):
        center, samples = get_one_cluster()
        assert(len(center) == seq_len)
        if center == None:
            ncluster = i + 1
            break
        decoded = process_cluster(center, samples)
        template_seqs.append(center)
        decoded_seqs.append(decoded)
        f_results.write(decoded + '\n')

    output_statistics(template_seqs, decoded_seqs)

    f_centers.close()
    f_clusters.close()
    f_results.close()
    
    print(fname_results + ' saved to ' + OUTPUT_PATH)

    return


def run_with_simulation(random_seed=6219, ncluster=5, nsample=5):
    fname_results = 'results.txt'
    f_results = open(OUTPUT_PATH + '/' + fname_results, 'w')
    
    random.seed(random_seed)

    gold_seqs = []
    decoded_seqs = []
    for i in tqdm(range(ncluster), desc="Decoding"):
        gold = ''.join(random.choices(symbols.all(), k=116))
        markers = {int(len(gold)*1/3): 'AA', int(len(gold)*2/3): 'AA'}

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

        
    hr = '-' * 20 + '\n'
    acc_str = hr + "ACCURACY\n" + hr
    acc_str += " max: {:.1%}\n".format(np.amax(accuracies))
    percents = np.array([75, 50, 25])
    for p, pt in zip(percents, np.percentile(accuracies, percents)):
        acc_str += " {:02d}%: {:.1%}\n".format(p, pt)
    acc_str += " min: {:.1%}\n".format(np.amin(accuracies))
    acc_str += "mean: {:.1%}\n".format(np.mean(accuracies))

    print(acc_str)

    acc_str += '\ncluster-wise accuracies:\n'
    for i in range(ncluster):
        acc_str += "cluster-{:05d}:\t{:.1%}\n".format(i, accuracies[i])

    base_accuracies = 1 - all_error / np.float64(ncluster)
    acc_str += '\nbase-wise accuracies:\n'
    acc_str += np.array2string(base_accuracies, separator=', ') + '\n'
    
    f.write(acc_str + '\n')

    plt.plot(list(range(length)), base_accuracies)
    plt.xlim([0, length])
    plt.ylim([max(np.amin(base_accuracies) * 0.85, 0), np.amax(base_accuracies)*1.1])
    plt.title('Base-wise Accuracy')
    plt.xlabel('Base Index')
    plt.ylabel('Accuracy')
    fname_base_acc = 'basewise_accuracy.png'
    plt.savefig(OUTPUT_PATH + '/' + fname_base_acc, format='png')
        
    sum = np.sum(all_error)
    err_distribution = np.full(length, 1 / length) if sum == 0 else all_error / sum
    distr_str = hr + 'ERROR DISTRIBUTION\n' + hr + np.array2string(err_distribution, precision=3, separator=', ')
    f.write(distr_str + '\n')

    plt.bar(list(range(length)), err_distribution * 100)
    plt.xlim([0, length])
    plt.ylim([0, min(np.amax(err_distribution)*100, 100)])
    plt.title('Error Distribution')
    plt.xlabel('Base Index')
    plt.ylabel('Probability Density (%)')
    fname_err_distr = 'error_distribution.png'
    plt.savefig(OUTPUT_PATH + '/' + fname_err_distr, format='png')

    f.close()

    print(fname_err_distr + ' saved to ' + OUTPUT_PATH)
    print(fname_base_acc + ' saved to ' + OUTPUT_PATH)
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
    