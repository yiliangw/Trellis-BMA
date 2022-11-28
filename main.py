import symbols
from noise import gen_noisy_samples
import marker_code
import random

SUB_P = 0.01
DEL_P = 0.01
INS_P = 0.01

def main():
    # random.seed()
    symbols.init('AGCT')
    
    gold = ''.join(random.choices(symbols.all(), k=116))

    # markers = {int(len(gold)/2): 'AAAA'}
    markers = {}
    encoded = marker_code.encode(gold, markers)

    samples = gen_noisy_samples(gold, 6, SUB_P, DEL_P, INS_P)
    # samples = [encoded]

    decoded_marker, decoded = marker_code.decode(samples, len(gold), markers, SUB_P, DEL_P, INS_P)

    print("withmarker:\t{}\nencoded:\t{}\ngold:\t\t{}\ndecoded:\t{}".format( decoded_marker, encoded, gold, decoded))
    print("decoded equals to gold: {}".format(decoded == gold))
    return


if __name__ == "__main__":
    main()