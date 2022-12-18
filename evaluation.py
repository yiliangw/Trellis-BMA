from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

def report(truth_path: str, result_path: str, output_dir: str):
    # Input files
    p_truth = Path(truth_path)
    p_result = Path(result_path)
    size, length = __check_data(p_truth, p_result)
    if size == None or length == None:
        print("Bad data for report")
        return -1

    # Output files
    p_output = Path(output_dir)
    p_output.mkdir(parents=True, exist_ok=True)
    fname_statistics    = 'statistics.txt'
    fname_err_distr     = 'positional_error_distribution.png'
    fname_pos_acc       = 'positional_accuracy.png'

    # Configure numpy to print arrays in full length
    np.set_printoptions(threshold=sys.maxsize)

    accuracies = np.zeros(size, np.float64)
    all_error = np.zeros(length, dtype=np.int32)
    seq_error = np.zeros(length, dtype=np.int8)

    with p_truth.open('r') as f_t, p_result.open('r') as f_r:
        for i in range(size):
            truth, result = f_t.readline().strip(), f_r.readline().strip()

            seq_error[:] = 0
            for pos in range(length):
                seq_error[pos] = int(truth[pos] != result[pos])
            all_error += seq_error
            accuracies[i] = 1 - (np.float64(np.sum(seq_error)) / length)

    f = (p_output / fname_statistics).open('w')

    hr = '=' * 20 + '\n'
    acc_str = hr + "ACCURACY\n" + hr
    acc_str += " max: {:.1%}\n".format(np.amax(accuracies))
    percents = np.array([75, 50, 25])
    for p, pt in zip(percents, np.percentile(accuracies, percents)):
        acc_str += " {:02d}%: {:.1%}\n".format(p, pt)
    acc_str += " min: {:.1%}\n".format(np.amin(accuracies))
    acc_str += "mean: {:.1%}\n".format(np.mean(accuracies))

    print(acc_str)

    acc_str += '\ncluster accuracies:\n'
    acc_str += np.array2string(accuracies, separator=', ') + '\n'

    idx_accuracies = 1 - all_error / np.float64(size)
    acc_str += '\npositional accuracies:\n'
    acc_str += np.array2string(idx_accuracies, separator=', ') + '\n'

    f.write(acc_str + '\n')
    
    plt.plot(list(range(length)), idx_accuracies)
    plt.xlim([0, length])
    plt.ylim([max(np.amin(idx_accuracies) * 0.85, 0), np.amax(idx_accuracies)*1.15])
    plt.title('Positional Accuracy')
    plt.xlabel('Base Index')
    plt.ylabel('Accuracy')
    
    plt.savefig(p_output / fname_pos_acc, format='png')
    plt.clf()
    print(fname_pos_acc + ' saved to ' + output_dir)
        
    sum = np.sum(all_error)
    err_distribution = np.full(length, 1 / length) if sum == 0 else all_error / sum
    distr_str = hr + 'ERROR DISTRIBUTION\n' + hr + np.array2string(err_distribution, precision=3, separator=', ')
    f.write(distr_str + '\n')

    plt.plot(list(range(length)), err_distribution * 100)
    plt.xlim([0, length])
    plt.ylim([0, min(np.amax(err_distribution)*100, 100)*1.15])
    plt.title('Positional Error Distribution')
    plt.xlabel('Base Index')
    plt.ylabel('Probability Mass (%)')
    plt.savefig(p_output / fname_err_distr, format='png')
    plt.clf()
    print(fname_err_distr + ' saved to ' + output_dir)

    f.close()
    print(fname_statistics + ' saved to ' + output_dir)

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
