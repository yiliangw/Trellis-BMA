from symbols import symbol_init
from noise import gen_noisy_samples
from marker_code import marker_encode, marker_decode
from random import random
from bma import bma

SUB_P = 0.05
DEL_P = 0.05
INS_P = 0.05

def main():
    symbol_init('ACGT')
    
    gold = 'ACGTTCATTACGGCTA'

    markers = ['AA']
    positions = [len(gold)/2]

    random.seed(100)

    encoded = marker_encode(gold, 'AA', positions)

    samples = gen_noisy_samples(gold, 6, SUB_P, DEL_P, INS_P)

    decoded = [marker_decode(sample, len(gold), markers, positions) for sample in samples]

    majority = bma(decoded)

    print("gold: {}\ndecoded: {}".format(gold, majority))

    return


if __name__ == "__main__":
    main()