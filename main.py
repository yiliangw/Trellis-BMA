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


def calculate_accuracy(gold, decoded):
    assert(len(gold) == len(decoded))
    cnt = 0
    for g, d in zip(gold, decoded):
        if g == d:
            cnt += 1
    return float(cnt) / len(gold)


def main():
    
    random.seed(6219)
    symbols.init('AGCT')
    
    gold = ''.join(random.choices(symbols.all(), k=114))
    markers = {int(len(gold)*1/3): 'AAA', int(len(gold)*2/3): 'AAA'}

    # Add markers to the sequence
    encoded = marker_code.encode(gold, markers)

    # Pass the encoded sequence through IDS channel to get noisy samples 
    samples = IDS_channel.generate_noisy_samples(encoded, 5, SUB_P, DEL_P, INS_P)

    decoded_with_marker, decoded = marker_code.decode(samples, len(gold), markers, SUB_P, DEL_P, INS_P)

    print("withmarker:\t{}\nencoded:\t{}\ngold:\t\t{}\ndecoded:\t{}".format(decoded_with_marker, encoded, align_sequence(gold, markers), align_sequence(decoded, markers)))

    print("Accuracy = {:.0%}".format(calculate_accuracy(gold, decoded)))
    return


if __name__ == "__main__":
    main()