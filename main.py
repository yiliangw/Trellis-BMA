from symbols import symbol_init
from noise import gen_noisy_samples
import marker_code
from random import random
from bma import bma

SUB_P = 0.05
DEL_P = 0.05
INS_P = 0.05

def main():
    random.seed(100)
    symbol_init('ACGT')
    
    gold = 'ACGTTCATTACGGCTA'

    markers = {len(gold)/2: 'AA'}
    encoded = marker_code.encode(gold, markers)

    samples = gen_noisy_samples(gold, 6, SUB_P, DEL_P, INS_P)

    decoded = marker_code.decode(samples, len(gold), markers, SUB_P, DEL_P, INS_P)

    print("gold: {}\ndecoded: {}".format(gold, decoded))
    return


if __name__ == "__main__":
    main()