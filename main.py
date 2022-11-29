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


def get_statistics(gold, decoded):
    assert(len(gold) == len(decoded))
    error = np.zeros(len(gold), dtype=np.int32)
    for i in range(len(gold)):
        error[i] == int(gold[i] != decoded[i])
    accuracy = 1 - (np.float64(np.sum(error)) / error.shape[0])
    return accuracy, error


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


def run_with_dataset(ncluster=5):

    f_centers = open(INPUT_PATH + '/Centers.txt', 'r')
    f_clusters = open(INPUT_PATH + '/Clusters.txt', 'r')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    f_results = open(OUTPUT_PATH + '/results.txt', 'w+')
    
    f_clusters.readline().strip()   # Skip the first line

    seq_len = len(f_centers.readline().strip())
    f_centers.seek(0)

    symbols.init('AGCT')

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

    for i in range(ncluster):
        center, samples = get_one_cluster()
        assert(len(center) == seq_len)
        if center == None:
            ncluster = i + 1
            break
        decoded = process_cluster(center, samples)
        f_results.write(decoded + '\n')

    # Get statistics
    accuracies = np.zeros(ncluster, dtype=np.float64)
    err_cnt = np.zeros(seq_len, dtype=np.int32)
    f_centers.seek(0)
    f_results.seek(0)
    for i in range(ncluster):
        center = f_centers.readline().strip()
        result = f_results.readline().strip()
        accuracy, error = get_statistics(center, result)
        accuracies[i] = accuracy
        err_cnt += error
    
    print("Average accuracy:\t{:.0%}".format(np.average(accuracies)))
    print("Medium accuracy:\t{:.0%}".format(np.median(accuracies)))
    print("Minimum accuracy:\t{:.0%}".format(np.amin(accuracies)))

    sum = np.sum(err_cnt)
    err_distribution = err_cnt
    if sum != 0:
        err_distribution /= sum
    err_distribution * 100
    plt.plot(list(range(seq_len)), err_distribution)
    plt.xlim([0, seq_len-1])
    plt.ylim([0, 100])
    plt.title('Error Distribution')
    plt.xlabel('Base Index')
    plt.ylabel('Probability Density (%)')
    plt.savefig(OUTPUT_PATH + '/error_distribution.png', format='png')
    print('error_distribution.png saved to ' + OUTPUT_PATH)
    print('results.txt saved to ' + OUTPUT_PATH)

    f_centers.close()
    f_clusters.close()
    f_results.close()

    return


def run_with_simulation(random_seed=6219, nsample=5):
    
    random.seed(random_seed)
    symbols.init('AGCT')
    
    gold = ''.join(random.choices(symbols.all(), k=114))
    markers = {int(len(gold)*1/3): 'AA', int(len(gold)*2/3): 'AA'}

    # Add markers to the template
    encoded = marker_code.encode(gold, markers)
    # Pass the encoded template through IDS channel to get noisy samples 
    samples = IDS_channel.generate_noisy_samples(encoded, nsample, SUB_P, DEL_P, INS_P)
    # Decode the template
    decoded_with_marker, decoded = marker_code.decode(samples, len(gold), markers, SUB_P, DEL_P, INS_P)

    # Statistics
    print("withmarker:\t{}".format(decoded_with_marker))
    print("encoded:\t{}".format(encoded))
    print("gold:\t\t{}".format(add_marker_placeholder(gold, markers)))
    print("decoded:\t{}".format(add_marker_placeholder(decoded, markers)))

    accuracy, errs = get_statistics(gold, decoded)
    print("\nAccuracy = {:.0%}".format(accuracy))
    plt.bar(list(range(len(gold))), errs)
    plt.xlim([0, len(gold)-1])
    plt.ylim([0, 1.5])
    plt.title('Error Distribution')
    plt.xlabel('Base Index')
    plt.ylabel('Error')
    plt.savefig(OUTPUT_PATH + '/error_distribution.png', format='png')
    print('error_distribution.png saved to ' + OUTPUT_PATH)
    return


def main():

    global ROOT_PATH, INPUT_PATH, OUTPUT_PATH
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = ROOT_PATH + '/data/input'
    OUTPUT_PATH = ROOT_PATH + '/data/output'

    global SUB_P, DEL_P, INS_P
    SUB_P = 0.01
    DEL_P = 0.01
    INS_P = 0.01

    SIMULATION = False

    if SIMULATION:
        run_with_simulation(random_seed=6219, nsample=5)
    else:
        run_with_dataset(ncluster=10)

    return


if __name__ == "__main__":
    main()