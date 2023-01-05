from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import json




def report(truth_path: str, result_path: str, output_dir: str, std_output=True):

    def __print_std(msg: str):
        if std_output:
            print(msg)

    # Input files
    p_truth = Path(truth_path)
    p_result = Path(result_path)
    size, length = __check_data(p_truth, p_result)
    if size == None or length == None:
        print("Bad data for report")
        return -1

    # Output files
    p_output = Path(output_dir)
    p_statistics = p_output / 'statistics.json'
    p_pos_err_rate = p_output / 'positional_error_rate.png'
    p_output.mkdir(parents=True, exist_ok=True)

    # Calculate the statistics
    norm_hamdist = np.zeros(size, np.float64)
    all_error = np.zeros(length, dtype=np.int32)
    seq_error = np.zeros(length, dtype=np.int8)
    with p_truth.open('r') as f_t, p_result.open('r') as f_r:
        for i in range(size):
            truth, result = f_t.readline().strip(), f_r.readline().strip()

            seq_error[:] = 0
            for pos in range(length):
                seq_error[pos] = int(truth[pos] != result[pos])
            all_error += seq_error
            norm_hamdist[i] = np.float64(np.sum(seq_error)) / length

    statistics_dict = {}

    # Global normallized hamming distance        
    norm_hamdist_dict = {
        'max': np.amax(norm_hamdist),
        'min': np.amin(norm_hamdist),
        'mean': np.mean(norm_hamdist),
    }
    percents = np.array([75, 50, 25])
    for p, pt in zip(percents, np.percentile(norm_hamdist, percents)):
        norm_hamdist_dict["{:02d}%".format(p)] = pt
    statistics_dict['normalized_hamming_distance'] = norm_hamdist_dict

    
    hr = '=' * 30
    __print_std(hr + "\nNormalized Hamming Distance\n" + hr)
    for field, val in norm_hamdist_dict.items():
        __print_std(field + ': {:.1%}'.format(val))

    # Positional error rates
    positional_err_rates = all_error / np.float64(size)
    statistics_dict['positional_error_rates'] = positional_err_rates.tolist()

    with p_statistics.open('w') as f:
        json.dump(statistics_dict, f, indent=2)

    __print_std(p_statistics.name + 'saved to ' + str(p_statistics.parent))

    plt.clf()
    plt.plot(list(range(length)), positional_err_rates)
    plt.xlim([0, length])
    plt.ylim([max(np.amin(positional_err_rates) * 0.85, 0), np.amax(positional_err_rates)*1.15])
    plt.title('Positional Error Rate')
    plt.xlabel('position')
    plt.ylabel('error rate')
    plt.savefig(p_pos_err_rate, format='png')

    __print_std(p_pos_err_rate.name + ' saved to ' + str(p_pos_err_rate.parent))
    __print_std(hr + '\n')

    return 0


'''
Check whether the ground truth and result files are valid and compatible
'''
def __check_data(ground_truth: Path, result: Path):

    err = (None, None)

    try:
        with ground_truth.open('r') as f:
            length = len(f.readline())
            f.seek(0)
            truth_size = 0
            for line in f:
                if len(line) != length:
                    return err
                truth_size += 1
    except EnvironmentError:
        return err

    try:
        with result.open('r') as f:
            res_size = 0
            for line in f:
                if len(line) != length:
                    return err
                res_size += 1
    except EnvironmentError:
        return err

    if truth_size == res_size:
        return truth_size, length - 1   # The length above includes \n
    else:
        return err
