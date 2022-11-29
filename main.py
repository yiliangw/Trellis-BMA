import symbols
import IDS_channel
import marker_code
import random

SUB_P = 0.01
DEL_P = 0.01
INS_P = 0.01


def align_sequence(sequence, markers):
    markers = sorted([(p, m) for p, m in markers.items()], key=(lambda t: t[0]))
    out = ''
    lp = 0
    for p, m in markers:
        out += sequence[lp:p]
        out += '-' * len(m)
        lp = p
    out += sequence[lp:]
    return out


def main():
    # random.seed()
    symbols.init('AGCT')
    
    gold = ''.join(random.choices(symbols.all(), k=116))
    markers = {int(len(gold)*1/3): 'AAA', int(len(gold)*2/3): 'AAA'}

    # Add markers to the sequence
    encoded = marker_code.encode(gold, markers)

    # Pass the encoded sequence through IDS channel to get noisy samples 
    samples = IDS_channel.generate_noisy_samples(encoded, 5, SUB_P, DEL_P, INS_P)

    decoded_with_marker, decoded = marker_code.decode(samples, len(gold), markers, SUB_P, DEL_P, INS_P)

    print("withmarker:\t{}\nencoded:\t{}\ngold:\t\t{}\ndecoded:\t{}".format(decoded_with_marker, encoded, align_sequence(gold, markers), align_sequence(decoded, markers)))
    print("decoded equals to gold: {}".format(decoded == gold))
    return


if __name__ == "__main__":
    main()