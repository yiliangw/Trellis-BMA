import symbols
from noise import gen_noisy_samples
import marker_code
import random

SUB_P = 0.01
DEL_P = 0.01
INS_P = 0.01

def main():
    random.seed(62122)
    symbols.init('AGCT')
    
    gold = ''.join(random.choices(symbols.all(), k=4))

    markers = {int(len(gold)/2): 'AA'}
    encoded = marker_code.encode(gold, markers)

    samples = gen_noisy_samples(gold, 4, SUB_P, DEL_P, INS_P)
    # samples = [encoded]

    # print("samples: {}".format(samples))

    decoded = marker_code.decode(samples, len(gold), markers, SUB_P, DEL_P, INS_P)

    print("gold:\t\t{}\nencoded:\t{}\ndecoded:\t{}".format(gold, encoded, decoded))
    print("decoded equals to gold: {}".format(decoded == gold))
    return


if __name__ == "__main__":
    main()