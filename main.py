import symbols
import IDS_channel
import marker_code
import random
import numpy as np
import matplotlib.pyplot as plt
import os


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


def output_statistics(template_seqs, decoded_seqs):
    assert(len(template_seqs) == len(decoded_seqs))
    ncluster = len(template_seqs)
    length = len(template_seqs[0])

    f = open(OUTPUT_PATH + '/statistics.txt', 'w')

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

    accstr = \
        "Average accuracy: {:.0%}\n".format(np.average(accuracies)) + \
        "Medium accuracy:  {:.0%}\n".format(np.average(accuracies)) + \
        "Minimum accuracy: {:.0%}\n".format(np.amin(accuracies))

    print(accstr)

    accstr += '\nAccuracies for each cluster:\n'
    for i in range(ncluster):
        accstr += "Cluster-{}:\t{:.0%}\n".format(i, accuracies[i])

    f.write(accstr)
        
    sum = np.sum(all_error)
    err_distribution = all_error if sum == 0 else all_error / sum
    err_distribution * 100

    plt.plot(list(range(length)), err_distribution)
    plt.xlim([0, length-1])
    plt.ylim([0, 100])
    plt.title('Error Distribution')
    plt.xlabel('Base Index')
    plt.ylabel('Probability Density (%)')
    plt.savefig(OUTPUT_PATH + '/error_distribution.png', format='png')

    f.close()
    print('statistics.txt saved to ' + OUTPUT_PATH)
    print('error_distribution.png saved to ' + OUTPUT_PATH)

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
        marker_info = [(int(len(center)/3), 2), (int(len(center)*2/3), 2)]
        gold, markers = seperate_markers(center, marker_info)
        decoded_with_marker, decoded = marker_code.decode(samples[:5], len(gold), markers, SUB_P, DEL_P, INS_P)
        return decoded_with_marker

    f_centers = open(INPUT_PATH + '/Centers.txt', 'r')
    f_clusters = open(INPUT_PATH + '/Clusters.txt', 'r')
    f_results = open(OUTPUT_PATH + '/results.txt', 'w+')
    
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
    for i in range(ncluster):
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

    return


def run_with_simulation(random_seed=6219, ncluster=5, nsample=5):

    f_results = open(OUTPUT_PATH + '/results.txt', 'w')
    
    random.seed(random_seed)

    gold_seqs = []
    decoded_seqs = []
    for i in range(ncluster):
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
    print("results.txt saves to " + OUTPUT_PATH)

    return


def main():

    global ROOT_PATH, INPUT_PATH, OUTPUT_PATH
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = ROOT_PATH + '/data/input'
    OUTPUT_PATH = ROOT_PATH + '/data/output'

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    global SUB_P, DEL_P, INS_P
    SUB_P = 0.01
    DEL_P = 0.01
    INS_P = 0.01

    SIMULATION = False

    symbols.init(['A', 'G', 'C', 'T'])

    if SIMULATION:
        run_with_simulation(random_seed=6219, ncluster=5, nsample=5)
    else:
        run_with_dataset(ncluster=5)

    return


if __name__ == "__main__":
    main()