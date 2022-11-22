from noise import gen_noisy_samples
from marker_code import marker_encode, markder_decode

SUB_P = 0.05
DEL_P = 0.05
INS_P = 0.05

def main():
    gold = 'ACGTTCATTACGGCTA'

    markers = ['AA']
    positions = [len(gold)/2]

    encoded = marker_encode(gold, 'AA', positions)

    samples = gen_noisy_samples(gold, 6, SUB_P, DEL_P, INS_P)

    decoded = markder_decode(samples, len(gold), markers, positions)

    print("gold: {}\ndecoded: {}".format(gold, decoded))

    return


if __name__ == "__main__":
    main()