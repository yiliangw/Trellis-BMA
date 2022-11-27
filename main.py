import symbols
from noise import gen_noisy_samples
import marker_code
import random

SUB_P = 0.05
DEL_P = 0.05
INS_P = 0.05

def main():
    random.seed(99)
    symbols.init('ACGT')
    
    gold = ''.join(random.choices(symbols.all(), k=96))

    markers = {int(len(gold)/2): 'AAAA'}
    encoded = marker_code.encode(gold, markers)

    samples = gen_noisy_samples(gold, 10, SUB_P, DEL_P, INS_P)
    # print("samples: {}".format(samples))

    decoded = marker_code.decode(samples, len(gold), markers, SUB_P, DEL_P, INS_P)

    print("gold:\t\t{}\nencoded:\t{}\ndecoded:\t{}".format(gold, encoded, decoded))
    print("decoded equals to gold: {}".format(decoded == gold))
    return


if __name__ == "__main__":
    main()